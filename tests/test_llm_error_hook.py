"""C5 llm/error 契约与 Core RetryExecutor 接线回归。"""

from __future__ import annotations

import asyncio

import pytest
from fake_llm import StepFinish, TextDelta, seq

from ftre_agent_core.agent import ReActAgent
from ftre_agent_core.agent.runner._execute_reasoning import ReasoningExecutor
from ftre_agent_core.agent.runner._state import Reasoning, RunState
from ftre_agent_core.event import EventType
from ftre_agent_core.hooks import LLM_ERROR_SPEC, LLMErrorDecision
from ftre_agent_core.llm import LLMError


def _state() -> RunState:
    state = RunState()
    state.runtime_context = {"session_id": "session-1", "max_iterations": 3}
    state.start()
    state.reply_id = "reply-1"
    return state


def _error_then_text(error_code: str, failures: int, text: str):
    calls = 0

    async def stream(messages, tools=None):
        nonlocal calls
        del messages, tools
        if calls < failures:
            calls += 1
            raise LLMError(message="failed", code=error_code)
        for chunk in seq(
            TextDelta(text=text),
            StepFinish(
                finish_reason="stop",
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            ),
        ):
            yield chunk

    return stream


class DecisionDispatcher:
    def __init__(self, decision=None, *, raises: bool = False) -> None:
        self.decision = decision
        self.raises = raises
        self.payloads = []

    async def dispatch(self, spec, payload, *, context=None):
        del context
        if spec is LLM_ERROR_SPEC:
            self.payloads.append(payload)
            if self.raises:
                raise RuntimeError("policy unavailable")
            return self.decision
        return await spec.default(payload)


@pytest.mark.asyncio
async def test_llm_error_hook_runs_before_next_core_attempt():
    dispatcher = DecisionDispatcher(LLMErrorDecision("retry", delay=0))
    agent = ReActAgent(
        model="fake",
        api_key="fake",
        max_iterations=3,
        max_retries=1,
        retry_delay=0,
        hooks=dispatcher,
    )
    agent.runner._llm.stream = _error_then_text("rate_limit", failures=1, text="ok")

    executor = ReasoningExecutor(agent, _state(), agent.runner.llm, dispatcher)
    events = [event async for event in executor.stream(Reasoning())]

    assert executor.result is not None
    assert executor.result.text == "ok"
    assert [payload.attempt for payload in dispatcher.payloads] == [1]
    assert dispatcher.payloads[0].max_attempts == 2
    assert len([event for event in events if event.type == EventType.RETRY]) == 1


@pytest.mark.asyncio
async def test_llm_error_stop_prevents_retry_even_for_retryable_error():
    dispatcher = DecisionDispatcher(LLMErrorDecision("stop", reason="policy stop"))
    agent = ReActAgent(
        model="fake", api_key="fake", max_retries=3, retry_delay=0, hooks=dispatcher
    )
    async def always_error(messages, tools=None):
        del messages, tools
        raise LLMError(message="failed", code="rate_limit")
        yield

    agent.runner._llm.stream = always_error

    executor = ReasoningExecutor(agent, _state(), agent.runner.llm, dispatcher)
    events = [event async for event in executor.stream(Reasoning())]

    assert executor.result is not None
    assert executor.result.error is not None
    assert len(dispatcher.payloads) == 1
    assert not [event for event in events if event.type == EventType.RETRY]


@pytest.mark.asyncio
async def test_llm_error_hook_failure_falls_back_to_core_default(caplog):
    dispatcher = DecisionDispatcher(raises=True)
    agent = ReActAgent(
        model="fake", api_key="fake", max_retries=1, retry_delay=0, hooks=dispatcher
    )
    agent.runner._llm.stream = _error_then_text("rate_limit", failures=1, text="ok")

    executor = ReasoningExecutor(agent, _state(), agent.runner.llm, dispatcher)
    events = [event async for event in executor.stream(Reasoning())]

    assert executor.result is not None
    assert executor.result.text == "ok"
    assert len([event for event in events if event.type == EventType.RETRY]) == 1
    assert "[llm/error] listener failed" in caplog.text


def test_llm_error_payload_is_bounded_by_attempt_limit():
    from ftre_agent_core.hooks import LLMErrorPayload

    with pytest.raises(ValueError):
        LLMErrorPayload(
            session_id="s",
            turn_id="t",
            iteration=1,
            model="m",
            error_code="timeout",
            error_message="failed",
            attempt=3,
            max_attempts=2,
            cancellation=asyncio.Event(),
        )
