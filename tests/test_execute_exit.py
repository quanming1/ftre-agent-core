# tests/test_execute_exit.py
"""ExitExecutor 单元测试。"""
import pytest
from ftre_agent_core.agent.runner._execute_exit import ExitExecutor
from ftre_agent_core.agent.runner._actions import Exit, ExitOutcome
from ftre_agent_core.agent.runner._state import RunState
from ftre_agent_core.agent.react import ReActAgent
from ftre_agent_core.event import EventType
from ftre_agent_core.types import ReplyFinishedReason


def make_agent():
    return ReActAgent(
        model="fake", api_key="fake", system_prompt="test",
        max_iterations=5,
    )


def make_state():
    s = RunState()
    s.runtime_context = {"session_id": "s1", "max_iterations": 5}
    s.start()
    s.reply_id = "r1"
    return s


@pytest.mark.asyncio
async def test_completed_exit_yields_reply_end():
    agent = make_agent()
    state = make_state()

    executor = ExitExecutor(agent, state, agent.hook_manager)
    events = [e async for e in executor.stream(Exit(finished_reason=ReplyFinishedReason.COMPLETED))]

    types = [e.type for e in events]
    assert EventType.REPLY_END in types
    assert executor.outcome.should_continue is False
    assert state.done_reason == ReplyFinishedReason.COMPLETED


@pytest.mark.asyncio
async def test_error_exit_no_on_stop_hook():
    agent = make_agent()
    state = make_state()

    executor = ExitExecutor(agent, state, agent.hook_manager)
    events = [e async for e in executor.stream(
        Exit(finished_reason=ReplyFinishedReason.ERROR, error="boom", error_code="test")
    )]

    assert executor.outcome.should_continue is False
    assert state.done_reason == ReplyFinishedReason.ERROR
    assert state.error == "boom"
    assert state.error_code == "test"


@pytest.mark.asyncio
async def test_on_stop_block_returns_continue():
    from ftre_agent_core.hooks import ON_STOP, StopInput, HookOutput

    agent = make_agent()
    state = make_state()

    def block_hook(inp: StopInput):
        return HookOutput(decision="block", reason="keep working")

    agent.hook_manager.register(ON_STOP, block_hook)

    executor = ExitExecutor(agent, state, agent.hook_manager)
    events = [e async for e in executor.stream(Exit(finished_reason=ReplyFinishedReason.COMPLETED))]

    assert executor.outcome.should_continue is True
    assert executor.outcome.continue_hint == "keep working"
    types = [e.type for e in events]
    assert EventType.REPLY_END not in types
    assert EventType.HINT_BLOCK in types
    msgs = agent.memory.messages
    assert any("keep working" in str(m.get("content", "")) for m in msgs)


@pytest.mark.asyncio
async def test_on_stop_allow_exits_normally():
    from ftre_agent_core.hooks import ON_STOP, StopInput, HookOutput

    agent = make_agent()
    state = make_state()

    def allow_hook(inp: StopInput):
        return HookOutput(decision="allow")

    agent.hook_manager.register(ON_STOP, allow_hook)

    executor = ExitExecutor(agent, state, agent.hook_manager)
    events = [e async for e in executor.stream(Exit(finished_reason=ReplyFinishedReason.COMPLETED))]

    assert executor.outcome.should_continue is False
    assert state.done_reason == ReplyFinishedReason.COMPLETED
