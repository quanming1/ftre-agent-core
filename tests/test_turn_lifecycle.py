"""
Turn Lifecycle Ownership 测试 — 验证 agent-core 不再产出 Step 事件，
turn_id 从 runtime_context 读取，done_reason/error_code 通过 RunState 暴露。
"""
import pytest

from ftre_agent_core.agent import ReActAgent
from ftre_agent_core.event import ReplyFinishedReason
from ftre_agent_core.agent.runner import RunStatus
from fake_llm import seq, TextDelta, StepFinish, ToolCall
from ftre_agent_core.llm import LLMError
from ftre_agent_core.tool import tool


def make_agent(max_iterations=5, max_retries=5):
    return ReActAgent(
        model="fake",
        api_key="fake",
        system_prompt="test",
        max_iterations=max_iterations,
        max_retries=max_retries,
    )


# ── Step 9: turn_id 从 runtime_context 传入 ──────────────────────────────

@pytest.mark.asyncio
async def test_turn_id_from_runtime_context():
    """ftre 传入的 turn_id 被使用，不自动生成。"""
    agent = make_agent()

    async def fake_stream(messages, tools=None):
        for chunk in seq(
            TextDelta(text="hello"),
            StepFinish(finish_reason="stop"),
        ):
            yield chunk

    agent.runner.llm.stream = fake_stream
    events = [e async for e in agent.run("hi", runtime_context={"turn_id": "turn_test123"})]

    assert agent.run_state.turn_id == "turn_test123"


# ── Step 10: 无 turn_id 时自动生成 ────────────────────────────────────────

@pytest.mark.asyncio
async def test_turn_id_auto_generated():
    """runtime_context 无 turn_id 时，agent-core 兜底生成。"""
    agent = make_agent()

    async def fake_stream(messages, tools=None):
        for chunk in seq(
            TextDelta(text="hello"),
            StepFinish(finish_reason="stop"),
        ):
            yield chunk

    agent.runner.llm.stream = fake_stream
    events = [e async for e in agent.run("hi")]

    assert agent.run_state.turn_id.startswith("turn_")
    assert len(agent.run_state.turn_id) > 5


# ── Step 11: 正常完成 → done_reason=COMPLETED ─────────────────────────────

@pytest.mark.asyncio
async def test_done_reason_completed():
    """正常文本输出完成时 done_reason=COMPLETED。"""
    agent = make_agent()

    async def fake_stream(messages, tools=None):
        for chunk in seq(
            TextDelta(text="done"),
            StepFinish(finish_reason="stop"),
        ):
            yield chunk

    agent.runner.llm.stream = fake_stream
    events = [e async for e in agent.run("start")]

    assert agent.run_state.done_reason == ReplyFinishedReason.COMPLETED
    assert agent.run_state.status == RunStatus.COMPLETED


# ── Step 12: max_iterations 耗尽 → done_reason=MAX_ITERATIONS ────────────

@pytest.mark.asyncio
async def test_done_reason_max_iterations():
    """迭代次数耗尽时 done_reason=MAX_ITERATIONS。"""
    @tool(description="echo")
    def echo(text: str) -> str:
        return f"echo:{text}"

    agent = make_agent(max_iterations=1)
    agent._registry.register(echo)

    async def fake_stream(messages, tools=None):
        for chunk in seq(
            ToolCall(id="c1", name="echo", input={"text": "hi"}),
            StepFinish(finish_reason="tool_calls"),
        ):
            yield chunk

    agent.runner.llm.stream = fake_stream
    events = [e async for e in agent.run("start")]

    assert agent.run_state.done_reason == ReplyFinishedReason.EXCEED_MAX_ITERS
    assert agent.run_state.status == RunStatus.COMPLETED


# ── Step 13: LLM 错误 → done_reason=ERROR + error_code ───────────────────

@pytest.mark.asyncio
async def test_done_reason_error():
    """不可重试错误时 done_reason=ERROR + error_code 被设置。"""
    agent = make_agent(max_retries=0)

    async def fake_stream(messages, tools=None):
        raise LLMError(message="bad request", code="bad_request")
        yield  # pragma: no cover — makes this an async generator

    agent.runner.llm.stream = fake_stream
    events = [e async for e in agent.run("start")]

    assert agent.run_state.done_reason == ReplyFinishedReason.ERROR
    assert agent.run_state.status == RunStatus.ERROR
    assert agent.run_state.error_code == "bad_request"
    assert agent.run_state.error is not None
    assert "bad_request" in agent.run_state.error




