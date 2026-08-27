"""C6 llm/stream attempt 元数据与重试边界回归。"""

from __future__ import annotations

import asyncio

import pytest
from fake_llm import StepFinish, TextDelta, seq

from ftre_agent_core.agent import ReActAgent
from ftre_agent_core.agent.runner._execute_reasoning import ReasoningExecutor
from ftre_agent_core.agent.runner._state import Reasoning, RunState
from ftre_agent_core.hooks import LLM_STREAM_SPEC, LLMStreamPayload
from ftre_agent_core.llm import LLMError


def _state() -> RunState:
    state = RunState()
    state.runtime_context = {"session_id": "session-c6", "max_iterations": 3}
    state.start()
    state.reply_id = "reply-c6"
    return state


class StreamRecorder:
    def __init__(self) -> None:
        self.payloads: list[LLMStreamPayload] = []

    async def dispatch(self, spec, payload, *, context=None):
        del context
        if spec is LLM_STREAM_SPEC:
            self.payloads.append(payload)
        return await spec.default(payload)


@pytest.mark.asyncio
async def test_payload_attempt_increments_for_each_core_retry() -> None:
    recorder = StreamRecorder()
    agent = ReActAgent(
        model="fake",
        api_key="fake",
        max_retries=1,
        retry_delay=0,
        hooks=recorder,
    )
    calls = 0

    async def stream(messages, tools=None):
        nonlocal calls
        del messages, tools
        calls += 1
        if calls == 1:
            raise LLMError(message="temporary", code="timeout")
        for chunk in seq(TextDelta(text="ok"), StepFinish(finish_reason="stop")):
            yield chunk

    agent.runner.llm.stream = stream
    executor = ReasoningExecutor(agent, _state(), agent.runner.llm, recorder)
    _events = [event async for event in executor.stream(Reasoning())]

    assert [payload.attempt for payload in recorder.payloads] == [1, 2]
    assert {payload.max_attempts for payload in recorder.payloads} == {2}


def test_payload_defaults_keep_direct_construction_compatible() -> None:
    payload = LLMStreamPayload(
        agent_id="a",
        session_id="s",
        turn_id="t",
        model="m",
        messages=(),
        tools=(),
        cancellation=asyncio.Event(),
        invoke=lambda: (),
    )
    assert payload.attempt == 1
    assert payload.max_attempts == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attempt": 0},
        {"max_attempts": 0},
        {"attempt": 2, "max_attempts": 1},
    ],
)
def test_payload_rejects_invalid_attempt_coordinates(kwargs) -> None:
    values = {
        "agent_id": "a",
        "session_id": "s",
        "turn_id": "t",
        "model": "m",
        "messages": (),
        "tools": (),
        "cancellation": asyncio.Event(),
        "invoke": lambda: (),
        **kwargs,
    }
    with pytest.raises(ValueError):
        LLMStreamPayload(**values)
