# tests/test_execute_reasoning.py
"""ReasoningExecutor 单元测试。"""
import pytest
from fake_llm import StepFinish, TextDelta, ToolCall, seq

from ftre_agent_core.agent.react import ReActAgent
from ftre_agent_core.agent.runner._execute_reasoning import ReasoningExecutor
from ftre_agent_core.agent.runner._state import Reasoning, RunState
from ftre_agent_core.event import EventType
from ftre_agent_core.llm import LLMError


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
        for chunk in seq(
            TextDelta(text="hello"),
            StepFinish( finish_reason="stop", usage={ "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, }, ),
        ):
            yield chunk

    agent.runner.llm.stream = fake_stream

    executor = ReasoningExecutor(agent, state, agent.runner.llm, agent.hooks)
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
    assert len(agent.messages) == 1
    assert agent.messages[0]["role"] == "assistant"


@pytest.mark.asyncio
async def test_tool_call_turn():
    agent = make_agent()
    state = make_state()

    async def fake_stream(messages, tools=None):
        for chunk in seq(
            ToolCall(id="c1", name="echo", input={"text": "hi"}),
            StepFinish( finish_reason="tool_calls", usage={ "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, }, ),
        ):
            yield chunk

    agent.runner.llm.stream = fake_stream

    executor = ReasoningExecutor(agent, state, agent.runner.llm, agent.hooks)
    [e async for e in executor.stream(Reasoning())]

    assert executor.result.text == ""
    assert len(executor.result.tool_calls) == 1
    assert executor.result.tool_calls[0].id == "c1"
    assert executor.result.finish_reason == "tool-calls"


@pytest.mark.asyncio
async def test_max_tokens_drops_truncated_tool_calls():
    """max-tokens 截断时 tool-call 参数不完整，整体丢弃不可执行（对齐 DSH）；
    text / reasoning 保留。"""
    agent = make_agent()
    state = make_state()

    async def fake_stream(messages, tools=None):
        for chunk in seq(
            TextDelta(text="partial"),
            ToolCall(id="c1", name="echo", input={"text": "hi"}),
            StepFinish( finish_reason="length", usage={ "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, }, ),
        ):
            yield chunk

    agent.runner.llm.stream = fake_stream

    executor = ReasoningExecutor(agent, state, agent.runner.llm, agent.hooks)
    [e async for e in executor.stream(Reasoning())]

    assert executor.result.text == "partial"
    assert executor.result.tool_calls == []
    assert executor.result.finish_reason == "max-tokens"
    # 被丢弃的 tool-call 不写入 reply（context 中无 ToolCallBlock）
    assistant_msg = agent.messages[-1]
    assert assistant_msg["role"] == "assistant"
    content = assistant_msg["content"]
    block_types = [b.get("type") for b in content] if isinstance(content, list) else []
    assert "tool-call" not in block_types


@pytest.mark.asyncio
async def test_hint_written_to_memory_before_llm_call():
    agent = make_agent()
    state = make_state()

    hint_seen = None

    async def fake_stream(messages, tools=None):
        nonlocal hint_seen
        hint_seen = messages[-1]
        for chunk in seq(
            TextDelta(text="done"),
            StepFinish( finish_reason="stop", usage={ "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, }, ),
        ):
            yield chunk

    agent.runner.llm.stream = fake_stream

    executor = ReasoningExecutor(agent, state, agent.runner.llm, agent.hooks)
    _events = [e async for e in executor.stream(Reasoning(hint="test hint"))]

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
        for chunk in seq(
            TextDelta(text="final"),
            StepFinish( finish_reason="stop", usage={ "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, }, ),
        ):
            yield chunk

    agent.runner.llm.stream = fake_stream

    executor = ReasoningExecutor(agent, state, agent.runner.llm, agent.hooks)
    _events = [e async for e in executor.stream(Reasoning(force_no_tools=True))]

    assert tools_received is None


@pytest.mark.asyncio
async def test_incomplete_usage_does_not_emit_model_call_end_or_update_state(caplog):
    agent = make_agent()
    state = make_state()

    async def fake_stream(messages, tools=None):
        for chunk in seq(
            TextDelta(text="done"),
            StepFinish( finish_reason="stop", usage={"prompt_tokens": 10, "completion_tokens": 5}, ),
        ):
            yield chunk

    agent.runner.llm.stream = fake_stream

    executor = ReasoningExecutor(agent, state, agent.runner.llm, agent.hooks)
    with caplog.at_level("WARNING"):
        events = [e async for e in executor.stream(Reasoning())]

    assert EventType.MODEL_CALL_END not in [event.type for event in events]
    assert state.token_usage == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    assert "未返回完整 token usage" in caplog.text


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
        for chunk in seq(
            TextDelta(text="success"),
            StepFinish( finish_reason="stop", usage={ "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, }, ),
        ):
            yield chunk

    agent.runner.llm.stream = fake_stream

    executor = ReasoningExecutor(agent, state, agent.runner.llm, agent.hooks)
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
        yield

    agent.runner.llm.stream = fake_stream

    executor = ReasoningExecutor(agent, state, agent.runner.llm, agent.hooks)
    _events = [e async for e in executor.stream(Reasoning())]

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
        yield

    agent.runner.llm.stream = fake_stream

    executor = ReasoningExecutor(agent, state, agent.runner.llm, agent.hooks)
    [e async for e in executor.stream(Reasoning())]

    assert call_count == 1
    assert executor.result.error is not None
    assert executor.result.error.code == "bad_request"
