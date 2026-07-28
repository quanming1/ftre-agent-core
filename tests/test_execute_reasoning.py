# tests/test_execute_reasoning.py
"""ReasoningExecutor 单元测试。"""
import pytest
from ftre_agent_core.agent.runner._execute_reasoning import ReasoningExecutor
from ftre_agent_core.agent.runner._state import Reasoning, TurnResult
from ftre_agent_core.agent.runner._state import RunState
from ftre_agent_core.agent.react import ReActAgent
from ftre_agent_core.llm import TextDelta, ToolCall, StepFinish, LLMError
from ftre_agent_core.event import EventType


def make_agent():
    return ReActAgent(
        model="fake", api_key="fake", system_prompt="test",
        max_iterations=5, max_retries=1, retry_delay=0.01,
    )


def make_state():
    s = RunState()
    s.runtime_context = {"session_id": "s1", "max_iterations": 5}
    s.start()
    s.reply_id = "r1"
    return s


@pytest.mark.asyncio
async def test_text_only_turn():
    agent = make_agent()
    state = make_state()

    async def fake_stream(messages, tools=None):
        yield TextDelta(text="hello")
        yield StepFinish(finish_reason="stop", usage={"prompt_tokens": 10, "completion_tokens": 5})

    agent.runner.llm.stream = fake_stream

    executor = ReasoningExecutor(agent, state, agent.runner.llm, agent.hook_manager)
    events = [e async for e in executor.stream(Reasoning())]

    assert executor.result.text == "hello"
    assert executor.result.tool_calls == []
    assert executor.result.finish_reason == "stop"
    assert executor.result.error is None
    types = [e.type for e in events]
    assert EventType.MODEL_CALL_START in types
    assert EventType.TEXT_BLOCK_START in types
    assert EventType.TEXT_BLOCK_DELTA in types
    assert EventType.TEXT_BLOCK_END in types
    assert EventType.MODEL_CALL_END in types
    assert len(agent.memory.messages) == 1
    assert agent.memory.messages[0]["role"] == "assistant"


@pytest.mark.asyncio
async def test_tool_call_turn():
    agent = make_agent()
    state = make_state()

    async def fake_stream(messages, tools=None):
        yield ToolCall(id="c1", name="echo", input={"text": "hi"})
        yield StepFinish(finish_reason="tool_calls")

    agent.runner.llm.stream = fake_stream

    executor = ReasoningExecutor(agent, state, agent.runner.llm, agent.hook_manager)
    events = [e async for e in executor.stream(Reasoning())]

    assert executor.result.text == ""
    assert len(executor.result.tool_calls) == 1
    assert executor.result.tool_calls[0].id == "c1"
    assert executor.result.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_hint_written_to_memory_before_llm_call():
    agent = make_agent()
    state = make_state()

    hint_seen = None

    async def fake_stream(messages, tools=None):
        nonlocal hint_seen
        hint_seen = messages[-1]
        yield TextDelta(text="done")
        yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    executor = ReasoningExecutor(agent, state, agent.runner.llm, agent.hook_manager)
    events = [e async for e in executor.stream(Reasoning(hint="test hint"))]

    assert hint_seen is not None
    assert hint_seen["role"] == "user"
    # content 可能是 list[dict] 或 str
    content = hint_seen["content"]
    if isinstance(content, list):
        text = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    else:
        text = str(content)
    assert "test hint" in text


@pytest.mark.asyncio
async def test_force_no_tools_passes_none():
    agent = make_agent()
    state = make_state()

    tools_received = "not_called"

    async def fake_stream(messages, tools=None):
        nonlocal tools_received
        tools_received = tools
        yield TextDelta(text="final")
        yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    executor = ReasoningExecutor(agent, state, agent.runner.llm, agent.hook_manager)
    events = [e async for e in executor.stream(Reasoning(force_no_tools=True))]

    assert tools_received is None


@pytest.mark.asyncio
async def test_retry_on_rate_limit():
    agent = make_agent()
    state = make_state()

    call_count = 0

    async def fake_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise LLMError(message="rate limited", code="rate_limit")
        yield TextDelta(text="success")
        yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    executor = ReasoningExecutor(agent, state, agent.runner.llm, agent.hook_manager)
    events = [e async for e in executor.stream(Reasoning())]

    assert call_count == 2
    assert executor.result.text == "success"
    retry_events = [e for e in events if e.type == EventType.RETRY]
    assert len(retry_events) == 1


@pytest.mark.asyncio
async def test_retry_exhausted_returns_error():
    agent = make_agent()
    state = make_state()

    async def fake_stream(messages, tools=None):
        raise LLMError(message="rate limited", code="rate_limit")
        yield  # noqa: unreachable — makes it an async generator

    agent.runner.llm.stream = fake_stream

    executor = ReasoningExecutor(agent, state, agent.runner.llm, agent.hook_manager)
    events = [e async for e in executor.stream(Reasoning())]

    assert executor.result.error is not None
    assert executor.result.error.code == "rate_limit"


@pytest.mark.asyncio
async def test_unretryable_error_returns_error_immediately():
    agent = make_agent()
    state = make_state()

    call_count = 0

    async def fake_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        raise LLMError(message="bad request", code="bad_request")
        yield  # noqa: unreachable — makes it an async generator

    agent.runner.llm.stream = fake_stream

    executor = ReasoningExecutor(agent, state, agent.runner.llm, agent.hook_manager)
    events = [e async for e in executor.stream(Reasoning())]

    assert call_count == 1
    assert executor.result.error is not None
    assert executor.result.error.code == "bad_request"
