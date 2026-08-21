# tests/test_react_runner_new.py
"""ReActRunner 主循环集成测试。"""
import asyncio

import pytest
from fake_llm import ReasoningDelta, StepFinish, TextDelta, ToolCall, seq

from ftre_agent_core.agent import ReActAgent
from ftre_agent_core.event import EventType
from ftre_agent_core.message import Msg
from ftre_agent_core.message_context import MessageContext
from ftre_agent_core.tool import ToolRegistry, tool
from ftre_agent_core.types import ReplyFinishedReason


def make_agent(tools=None, max_iterations=5, max_retries=1):
    registry = ToolRegistry()
    for t in (tools or []):
        registry.register(t)
    return ReActAgent(
        model="fake", api_key="fake", system_prompt="test",
        tool_registry=registry, max_iterations=max_iterations,
        max_retries=max_retries, retry_delay=0.01,
    )


async def _collect_events(gen):
    return [e async for e in gen]


@pytest.mark.asyncio
async def test_simple_text_reply():
    agent = make_agent()

    async def fake_stream(messages, tools=None):
        for chunk in seq(
            TextDelta(text="hello world"),
            StepFinish(finish_reason="stop"),
        ):
            yield chunk

    agent.runner.llm.stream = fake_stream
    events = [e async for e in agent.run("hi")]

    types = [e.type for e in events]
    assert types.count(EventType.REPLY_START) == 1
    assert types.count(EventType.REPLY_END) == 1
    assert agent.run_state.done_reason == ReplyFinishedReason.COMPLETED


@pytest.mark.asyncio
async def test_tool_call_then_text():
    @tool(description="Echo")
    def echo(text: str) -> str:
        return f"echo:{text}"

    agent = make_agent(tools=[echo])
    call_count = 0

    async def fake_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            for chunk in seq(
                ToolCall(id="c1", name="echo", input={"text": "hi"}),
                StepFinish(finish_reason="tool_calls"),
            ):
                yield chunk
        else:
            for chunk in seq(
                TextDelta(text="done"),
                StepFinish(finish_reason="stop"),
            ):
                yield chunk

    agent.runner.llm.stream = fake_stream
    events = [e async for e in agent.run("echo hi")]

    assert call_count == 2
    types = [e.type for e in events]
    assert types.count(EventType.REPLY_START) == 1
    assert types.count(EventType.REPLY_END) == 1
    assert EventType.TOOL_RESULT_END in types
    assert agent.run_state.done_reason == ReplyFinishedReason.COMPLETED


@pytest.mark.asyncio
async def test_tool_call_turn_is_one_atomic_assistant_message():
    """同轮纯 reasoning + tool_calls 在下一次请求中不能拆开或丢失。"""
    @tool(description="Echo")
    def echo(text: str) -> str:
        return f"echo:{text}"

    agent = make_agent(tools=[echo])
    call_count = 0
    second_call_messages = None

    async def fake_stream(messages, tools=None):
        nonlocal call_count, second_call_messages
        call_count += 1
        if call_count == 1:
            for chunk in seq(
                ReasoningDelta(text="先调用 echo 获取结果"),
                ToolCall(id="c1", name="echo", input={"text": "hi"}),
                StepFinish(finish_reason="tool_calls"),
            ):
                yield chunk
        else:
            second_call_messages = messages
            for chunk in seq(
                TextDelta(text="done"),
                StepFinish(finish_reason="stop"),
            ):
                yield chunk

    agent.runner.llm.stream = fake_stream
    events = [event async for event in agent.run("echo hi")]

    assert second_call_messages is not None
    call_messages = [
        message
        for message in second_call_messages
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]
    assert len(call_messages) == 1
    assert call_messages[0]["content"] == ""
    assert call_messages[0]["reasoning_content"] == "先调用 echo 获取结果"
    assert call_messages[0]["tool_calls"][0]["id"] == "c1"

    call_index = second_call_messages.index(call_messages[0])
    assert second_call_messages[call_index + 1]["role"] == "tool"
    assert not any(
        message.get("role") == "assistant"
        and message.get("reasoning_content") == "先调用 echo 获取结果"
        and not message.get("tool_calls")
        for message in second_call_messages
    )

    # Core 运行态与 FTRE 事件投影采用同一种结构：整次 reply 只有一个
    # assistant Msg，内部按真实时序保存 reasoning → tool_call → result → text。
    reply_id = next(
        event.reply_id for event in events if event.type == EventType.REPLY_START
    )
    assistant_replies = [
        message for message in agent.state.context if message.role == "assistant"
    ]
    assert len(assistant_replies) == 1
    assert assistant_replies[0].id == reply_id
    assert [block.type for block in assistant_replies[0].content] == [
        "thinking",
        "tool_call",
        "tool_result",
        "text",
    ]

    # state.json 往返后生成的 Provider 消息必须与实时运行态逐字一致。
    live_messages = MessageContext.messages(agent.state.context)
    restored_context = [
        Msg.model_validate(message.model_dump(mode="json"))
        for message in agent.state.context
    ]
    assert MessageContext.messages(restored_context) == live_messages


@pytest.mark.asyncio
async def test_max_iterations():
    agent = make_agent(max_iterations=2)

    counter = 0

    async def fake_stream(messages, tools=None):
        nonlocal counter
        counter += 1
        for chunk in seq(
            ToolCall(id=f"c{counter}", name="nonexistent", input={}),
            StepFinish(finish_reason="tool_calls"),
        ):
            yield chunk

    agent.runner.llm.stream = fake_stream
    [e async for e in agent.run("loop")]

    assert agent.run_state.done_reason == ReplyFinishedReason.EXCEED_MAX_ITERS


@pytest.mark.asyncio
async def test_cancel_via_task_cancel():
    agent = make_agent()

    async def fake_stream(messages, tools=None):
        await asyncio.sleep(10)
        for chunk in seq(
            TextDelta(text="never"),
            StepFinish(finish_reason="stop"),
        ):
            yield chunk
        yield  # make it an async generator

    agent.runner.llm.stream = fake_stream

    task = asyncio.create_task(_collect_events(agent.run("hi")))
    await asyncio.sleep(0.1)
    agent.cancel_nowait()

    events = await task
    assert agent.run_state.done_reason == ReplyFinishedReason.INTERRUPTED
    types = [e.type for e in events]
    assert EventType.REPLY_END in types


@pytest.mark.asyncio
async def test_concurrent_run_raises():
    agent = make_agent()

    async def fake_stream(messages, tools=None):
        await asyncio.sleep(10)
        for chunk in seq(
            StepFinish(finish_reason="stop"),
        ):
            yield chunk
        yield  # make it an async generator

    agent.runner.llm.stream = fake_stream

    task1 = asyncio.create_task(_collect_events(agent.run("first")))
    await asyncio.sleep(0.05)

    with pytest.raises(RuntimeError, match="already running"):
        async for _ in agent.run("second"):
            pass

    task1.cancel()
    try:
        await task1
    except asyncio.CancelledError:
        pass
