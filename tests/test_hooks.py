"""Core-facing Hook contracts and dispatcher injection tests."""
from __future__ import annotations

import asyncio

import pytest

from ftre_agent_core.agent import ReActAgent
from ftre_agent_core.agent.runner._execute_acting import ExitExecutor
from ftre_agent_core.agent.runner._state import Exit, RunState
from ftre_agent_core.hooks import (
    AGENT_TURN_STOPPING_SPEC,
    ContinueTurn,
    HookMode,
    StopTurn,
    ToolCallIdentity,
    ToolPreExecutePayload,
    TurnStoppingPayload,
)
from ftre_agent_core.types import ReplyFinishedReason


class RecordingDispatcher:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    async def dispatch(self, spec, payload, *, context=None):
        self.calls.append((spec, payload, context))
        if spec is AGENT_TURN_STOPPING_SPEC:
            return self.result or StopTurn()
        return spec.default(payload) if spec.default else None


def make_state():
    state = RunState()
    state.runtime_context = {
        "session_id": "s1",
        "request_id": "q1",
        "max_iterations": 5,
        "cancellation": asyncio.Event(),
        "max_continuations": 2,
    }
    state.start()
    state.reply_id = "r1"
    return state


def test_specs_are_typed_and_waterfall():
    assert AGENT_TURN_STOPPING_SPEC.mode is HookMode.WATERFALL
    assert isinstance(
        ToolPreExecutePayload(ToolCallIdentity("c1", "echo"), {}, asyncio.Event()),
        ToolPreExecutePayload,
    )


@pytest.mark.asyncio
async def test_agent_accepts_host_dispatcher_without_registry():
    dispatcher = RecordingDispatcher()
    agent = ReActAgent(model="fake", api_key="fake", hooks=dispatcher, hook_context="scope")
    assert agent.hooks is dispatcher
    assert agent.hook_context == "scope"


@pytest.mark.asyncio
async def test_turn_stopping_continue_injects_hint_before_finalize():
    dispatcher = RecordingDispatcher(ContinueTurn("继续检查结果", source="test"))
    agent = ReActAgent(model="fake", api_key="fake", hooks=dispatcher)
    state = make_state()
    executor = ExitExecutor(agent, state, dispatcher)

    _events = [
        event async for event in executor.stream(
            Exit(finished_reason=ReplyFinishedReason.COMPLETED)
        )
    ]
    assert executor.outcome.should_continue is True
    assert executor.outcome.continue_hint == "继续检查结果"
    assert state.done_reason is None
    assert any("继续检查结果" in str(message.get("content", "")) for message in agent.messages)
    assert isinstance(dispatcher.calls[0][1], TurnStoppingPayload)


@pytest.mark.asyncio
async def test_turn_stopping_budget_exhaustion_finalizes():
    dispatcher = RecordingDispatcher(ContinueTurn("继续"))
    agent = ReActAgent(model="fake", api_key="fake", hooks=dispatcher)
    state = make_state()
    state.runtime_context["continuation_count"] = 2
    executor = ExitExecutor(agent, state, dispatcher)
    events = [
        event async for event in executor.stream(
            Exit(finished_reason=ReplyFinishedReason.COMPLETED)
        )
    ]
    assert executor.outcome.should_continue is False
    assert state.done_reason == ReplyFinishedReason.COMPLETED
    assert events


@pytest.mark.asyncio
async def test_default_dispatcher_is_not_created():
    agent = ReActAgent(model="fake", api_key="fake")
    assert agent.hooks is None
    assert agent.runner.tool_handler.hooks is None
