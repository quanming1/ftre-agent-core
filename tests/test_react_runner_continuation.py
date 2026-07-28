"""
ReActRunner continuation / retry 逻辑测试（状态机重构版）。

length 截断续写已删除。空响应重试和 on_stop hook 由新状态机处理。
"""
import logging

import pytest

from ftre_agent_core.agent import EventType, ReActAgent
from ftre_agent_core.event import ReplyFinishedReason
from ftre_agent_core.llm import ReasoningDelta, StepFinish, TextDelta, ToolCall
from ftre_agent_core.tool import tool, ToolRegistry


def make_agent(tools=None, max_iterations=3):
    registry = ToolRegistry()
    for t in (tools or []):
        registry.register(t)
    return ReActAgent(
        model="fake",
        api_key="fake",
        system_prompt="test",
        tool_registry=registry,
        max_iterations=max_iterations,
    )


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def test_step_finish_defaults_to_unknown():
    assert StepFinish().finish_reason == "unknown"


@pytest.mark.asyncio
async def test_reasoning_only_turn_is_treated_as_empty_response_retry():
    agent = make_agent()
    calls = 0

    async def fake_stream(messages, tools=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield ReasoningDelta(text="need more work")
            yield StepFinish(finish_reason="stop")
        else:
            yield TextDelta(text="done")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    events = [event async for event in agent.run("start")]

    assert calls == 2
    assert agent.state.done_reason == ReplyFinishedReason.COMPLETED


@pytest.mark.asyncio
async def test_empty_response_retries_then_requests_finalization_without_tools():
    @tool(description="Echo text")
    def echo(text: str) -> str:
        return f"echo:{text}"

    agent = make_agent(tools=[echo], max_iterations=6)
    calls = 0

    async def fake_stream(messages, tools=None):
        nonlocal calls
        calls += 1
        if calls <= 3:
            # 前三轮都返回空响应（带工具）
            if tools is None:
                raise RuntimeError(f"tools should not be None on call {calls}")
            if calls < 3:
                yield StepFinish(finish_reason="stop")
                yield  # make it an async generator
            else:
                yield TextDelta(text=" \n ")
                yield StepFinish(finish_reason="stop")
        else:
            # 第四轮：最终化（不带工具）
            if tools is not None:
                raise RuntimeError("tools should be None on finalization call")
            if messages[-1]["role"] != "user":
                raise RuntimeError("last message should be user")
            content = _content_text(messages[-1]["content"])
            if "直接给出回复用户的最终内容" not in content:
                raise RuntimeError(f"finalization prompt not found: {content}")
            yield TextDelta(text="final")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    events = [event async for event in agent.run("start")]

    assert calls == 4
    assert agent.state.done_reason == ReplyFinishedReason.COMPLETED


@pytest.mark.asyncio
async def test_unknown_finish_with_text_completes(caplog):
    """finish_reason=unknown + 有文本 → 直接完成（新行为）。"""
    agent = make_agent(max_iterations=3)
    calls = 0

    async def fake_stream(messages, tools=None):
        nonlocal calls
        calls += 1
        yield TextDelta(text="hello")
        yield StepFinish(finish_reason="unknown")

    agent.runner.llm.stream = fake_stream

    with caplog.at_level(logging.INFO):
        events = [event async for event in agent.run("start")]

    assert calls == 1
    assert agent.state.done_reason == ReplyFinishedReason.COMPLETED


@pytest.mark.asyncio
async def test_unknown_finish_with_empty_response_continues_without_finalization():
    agent = make_agent(max_iterations=5)
    calls = 0

    async def fake_stream(messages, tools=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield StepFinish(finish_reason="unknown")
        elif calls == 2:
            yield TextDelta(text="done")
            yield StepFinish(finish_reason="stop")
        else:
            pytest.fail("should not reach 3rd call")

    agent.runner.llm.stream = fake_stream

    events = [event async for event in agent.run("start")]

    assert calls == 2


@pytest.mark.asyncio
async def test_tool_call_turn_produces_tool_result_events():
    @tool(description="Echo text")
    def echo(text: str) -> str:
        return f"echo:{text}"

    agent = make_agent(tools=[echo])
    calls = 0

    async def fake_stream(messages, tools=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield ToolCall(id="call_echo", name="echo", input={"text": "x"})
            yield StepFinish(finish_reason="tool_calls")
        else:
            yield TextDelta(text="finished")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    events = [event async for event in agent.run("start")]

    assert calls == 2
    tool_results = [e for e in events if e.type == EventType.TOOL_RESULT_END]
    assert len(tool_results) == 1
    assert agent.state.done_reason == ReplyFinishedReason.COMPLETED


@pytest.mark.asyncio
async def test_multi_tool_call_events_are_emitted_before_results():
    @tool(description="Echo text")
    def echo(text: str) -> str:
        return f"echo:{text}"

    @tool(description="Uppercase text")
    def upper(text: str) -> str:
        return text.upper()

    agent = make_agent(tools=[echo, upper])
    calls = 0

    async def fake_stream(messages, tools=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield ToolCall(id="call_echo", name="echo", input={"text": "x"})
            yield ToolCall(id="call_upper", name="upper", input={"text": "y"})
            yield StepFinish(finish_reason="tool_calls")
        else:
            yield TextDelta(text="finished")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    events = [event async for event in agent.run("start")]

    assert calls == 2
    tool_results = [e for e in events if e.type == EventType.TOOL_RESULT_END]
    assert len(tool_results) == 2


@pytest.mark.asyncio
async def test_single_reply_start_and_reply_end():
    """验证一次 run() 只产一对 ReplyStart/ReplyEnd。"""
    agent = make_agent(max_iterations=3)
    calls = 0

    async def fake_stream(messages, tools=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield ToolCall(id="c1", name="nonexistent", input={})
            yield StepFinish(finish_reason="tool_calls")
        else:
            yield TextDelta(text="done")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    events = [event async for event in agent.run("start")]

    types = [e.type for e in events]
    assert types.count(EventType.REPLY_START) == 1
    assert types.count(EventType.REPLY_END) == 1
