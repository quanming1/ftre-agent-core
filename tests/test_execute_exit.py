"""ExitExecutor stop decision tests."""

import asyncio

import pytest

from ftre_agent_core.agent import ReActAgent
from ftre_agent_core.agent.runner._execute_acting import ExitExecutor
from ftre_agent_core.agent.runner._state import Exit, RunState
from ftre_agent_core.types import ReplyFinishedReason


class Dispatcher:
    def __init__(self, result):
        self.result = result

    async def dispatch(self, spec, payload, *, context=None):
        return self.result


def make_state():
    state = RunState()
    state.runtime_context = {
        "session_id": "s1",
        "max_iterations": 5,
        "cancellation": asyncio.Event(),
        "max_continuations": 3,
    }
    state.start()
    state.reply_id = "r1"
    return state


@pytest.mark.asyncio
async def test_completed_exit_yields_reply_end_without_hook():
    agent = ReActAgent(model="fake", api_key="fake")
    state = make_state()
    executor = ExitExecutor(agent, state)
    events = [
        event
        async for event in executor.stream(
            Exit(finished_reason=ReplyFinishedReason.COMPLETED)
        )
    ]
    assert events
    assert executor.outcome.should_continue is False
    assert state.done_reason == ReplyFinishedReason.COMPLETED


@pytest.mark.asyncio
async def test_error_exit_skips_stop_hook():
    dispatcher = Dispatcher(RuntimeError("must not run"))
    agent = ReActAgent(model="fake", api_key="fake", hooks=dispatcher)
    state = make_state()
    executor = ExitExecutor(agent, state, dispatcher)
    events = [
        event
        async for event in executor.stream(
            Exit(
                finished_reason=ReplyFinishedReason.ERROR,
                error="boom",
                error_code="test",
            )
        )
    ]
    assert events
    assert state.done_reason == ReplyFinishedReason.ERROR
    assert state.error == "boom"
