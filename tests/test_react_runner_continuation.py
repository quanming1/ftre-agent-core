"""
ReActRunner continuation / retry 逻辑测试。

原始 continuation_active / pending_user_messages 魔法键已删除，
continuation 逻辑由 on_stop hook 覆盖（见 test_hooks.py）。
本文件保留 length 截断续写、空响应重试等内置机制测试。
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
    # 验证最终状态
    assert agent.state.done_reason == ReplyFinishedReason.COMPLETED


@pytest.mark.asyncio
async def test_empty_response_retries_then_requests_finalization_without_tools():
    @tool(description="Echo text")
    def echo(text: str) -> str:
        return f"echo:{text}"

    agent = make_agent(tools=[echo], max_iterations=4)
    calls = 0

    async def fake_stream(messages, tools=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            assert tools is not None
            yield StepFinish(finish_reason="stop")
        elif calls == 2:
            assert tools is not None
            yield TextDelta(text=" \n ")
            yield StepFinish(finish_reason="stop")
        else:
            assert tools is None
            assert messages[-1]["role"] == "user"
            assert "直接给出回复用户的最终内容" in messages[-1]["content"]
            yield TextDelta(text="final")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    events = [event async for event in agent.run("start")]

    assert calls == 3
    assert agent.state.done_reason == ReplyFinishedReason.COMPLETED


@pytest.mark.asyncio
async def test_unknown_finish_with_text_logs_and_continues(caplog):
    agent = make_agent(max_iterations=3)
    calls = 0

    async def fake_stream(messages, tools=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield TextDelta(text="hello")
            yield StepFinish(finish_reason="unknown")
        else:
            yield TextDelta(text="done")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    with caplog.at_level(logging.INFO):
        events = [event async for event in agent.run("start")]

    assert calls == 2
    assert agent.state.done_reason == ReplyFinishedReason.COMPLETED


@pytest.mark.asyncio
async def test_unknown_finish_with_empty_response_continues_without_finalization():
    agent = make_agent(max_iterations=3)
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
async def test_tool_call_turn_does_not_emit_turn_end_before_followup_turn():
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
    # tool_result 事件应该存在（agent-core 不再产出 turn_end 事件）
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
async def test_length_finish_adds_hidden_user_continuation():
    agent = make_agent()
    calls = 0

    async def fake_stream(messages, tools=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield TextDelta(text="partial")
            yield StepFinish(finish_reason="length")
        else:
            assert messages[-1]["role"] == "user"
            assert "从刚才中断的位置继续" in messages[-1]["content"]
            yield TextDelta(text=" done")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    events = [event async for event in agent.run("start")]

    assert calls == 2
    # 续写提示应该是隐藏的 user message
    user_msgs = [m for m in agent.memory.messages if m["role"] == "user"]
    assert len(user_msgs) == 2  # original + continuation
    assert "从刚才中断的位置继续" in user_msgs[1]["content"]
