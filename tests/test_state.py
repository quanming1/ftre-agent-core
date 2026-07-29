# tests/test_state.py
"""RunState / RunStatus / CancelledError 单元测试。"""
import asyncio
import pytest
from ftre_agent_core.agent.runner._state import RunState, RunStatus, CancelledError
from ftre_agent_core.types import ReplyFinishedReason


def test_run_status_values():
    assert RunStatus.IDLE == "idle"
    assert RunStatus.RUNNING == "running"
    assert RunStatus.COMPLETED == "completed"
    assert RunStatus.ERROR == "error"
    assert RunStatus.CANCELLED == "cancelled"


def test_cancelled_error_is_exception():
    err = CancelledError("test")
    assert isinstance(err, Exception)


def test_run_state_defaults():
    state = RunState()
    assert state.status == RunStatus.IDLE
    assert state.iteration == 0
    assert state.done_reason is None
    assert state.error is None
    assert state.error_code is None
    assert state.trace_span is None
    assert state.runtime_context == {}
    assert state.reply_id == ""
    assert state.turn_id == ""
    assert state.empty_retries == 0
    assert state.in_finalization is False
    assert state.token_usage["prompt_tokens"] == 0
    assert state.token_usage["completion_tokens"] == 0
    assert state.token_usage["total_tokens"] == 0


def test_run_state_start_resets_fields():
    state = RunState()
    state.iteration = 5
    state.empty_retries = 3
    state.in_finalization = True
    state.error = "old error"
    state.runtime_context = {"session_id": "s1", "turn_id": "t1"}
    state.start()
    assert state.status == RunStatus.RUNNING
    assert state.iteration == 0
    assert state.empty_retries == 0
    assert state.in_finalization is False
    assert state.error is None
    assert state.error_code is None
    assert state.done_reason is None
    assert state.turn_id == "t1"  # 从 runtime_context 继承


def test_run_state_is_cancelled():
    state = RunState()
    assert state.is_cancelled is False
    state.status = RunStatus.CANCELLED
    assert state.is_cancelled is True


def test_run_state_is_done():
    state = RunState()
    assert state.is_done is False
    for status in (RunStatus.COMPLETED, RunStatus.ERROR, RunStatus.CANCELLED):
        state.status = status
        assert state.is_done is True
    state.status = RunStatus.RUNNING
    assert state.is_done is False
