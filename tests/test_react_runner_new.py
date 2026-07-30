# tests/test_react_runner_new.py
"""ReActRunner 主循环集成测试。"""
import asyncio
import pytest
from ftre_agent_core.agent import ReActAgent
from ftre_agent_core.event import EventType
from ftre_agent_core.llm import TextDelta, ToolCall, StepFinish, LLMError
from ftre_agent_core.tool import tool, ToolRegistry
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
        yield TextDelta(text="hello world")
        yield StepFinish(finish_reason="stop")

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
            yield ToolCall(id="c1", name="echo", input={"text": "hi"})
            yield StepFinish(finish_reason="tool_calls")
        else:
            yield TextDelta(text="done")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream
    events = [e async for e in agent.run("echo hi")]

    assert call_count == 2
    types = [e.type for e in events]
    assert types.count(EventType.REPLY_START) == 1
    assert types.count(EventType.REPLY_END) == 1
    assert EventType.TOOL_RESULT_END in types
    assert agent.run_state.done_reason == ReplyFinishedReason.COMPLETED


@pytest.mark.asyncio
async def test_max_iterations():
    agent = make_agent(max_iterations=2)

    counter = 0

    async def fake_stream(messages, tools=None):
        nonlocal counter
        counter += 1
        yield ToolCall(id=f"c{counter}", name="nonexistent", input={})
        yield StepFinish(finish_reason="tool_calls")

    agent.runner.llm.stream = fake_stream
    events = [e async for e in agent.run("loop")]

    assert agent.run_state.done_reason == ReplyFinishedReason.EXCEED_MAX_ITERS


@pytest.mark.asyncio
async def test_cancel_via_task_cancel():
    agent = make_agent()

    async def fake_stream(messages, tools=None):
        await asyncio.sleep(10)
        yield TextDelta(text="never")
        yield StepFinish(finish_reason="stop")
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
        yield StepFinish(finish_reason="stop")
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
