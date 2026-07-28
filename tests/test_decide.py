# tests/test_decide.py
"""_decide() 纯决策函数单元测试。"""
import pytest
from ftre_agent_core.agent.runner.react_runner import decide, MAX_EMPTY_RESPONSE_RETRIES, FINALIZATION_RETRY_PROMPT, EMPTY_FINAL_RESPONSE_MESSAGE
from ftre_agent_core.agent.runner._state import Reasoning, Acting, Exit, TurnResult, ExitOutcome
from ftre_agent_core.agent.runner._state import RunState
from ftre_agent_core.llm import ToolCall, LLMError
from ftre_agent_core.types import ReplyFinishedReason


def make_state(iteration=0, empty_retries=0, in_finalization=False, max_iterations=10):
    s = RunState()
    s.iteration = iteration
    s.empty_retries = empty_retries
    s.in_finalization = in_finalization
    s.runtime_context = {"max_iterations": max_iterations}
    return s


def make_turn(text="", tool_calls=None, finish_reason="stop", error=None):
    return TurnResult(
        text=text,
        reasoning="",
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        error=error,
    )


# --- 优先级 1: error 非空 → Exit(ERROR) ---
def test_error_in_turn_result_returns_exit_error():
    state = make_state()
    prev = make_turn(error=LLMError(message="boom", code="unknown"))
    action = decide(state, prev)
    assert isinstance(action, Exit)
    assert action.finished_reason == ReplyFinishedReason.ERROR
    assert "boom" in action.error


# --- 优先级 2: 有工具调用 → Acting ---
def test_tool_calls_returns_acting():
    state = make_state()
    tc = ToolCall(id="c1", name="echo", input={})
    prev = make_turn(tool_calls=[tc])
    action = decide(state, prev)
    assert isinstance(action, Acting)
    assert len(action.tool_calls) == 1


# --- 优先级 3: 有文本无工具 → Exit(COMPLETED) ---
def test_text_no_tools_returns_exit_completed():
    state = make_state()
    prev = make_turn(text="hello world")
    action = decide(state, prev)
    assert isinstance(action, Exit)
    assert action.finished_reason == ReplyFinishedReason.COMPLETED


# --- 优先级 4: 空响应 + in_finalization → Exit(ERROR) ---
def test_empty_in_finalization_returns_exit_error():
    state = make_state(in_finalization=True)
    prev = make_turn(text="")
    action = decide(state, prev)
    assert isinstance(action, Exit)
    assert action.finished_reason == ReplyFinishedReason.ERROR
    assert action.error_code == "empty_response"
    assert EMPTY_FINAL_RESPONSE_MESSAGE in action.error


# --- 优先级 5: 空响应 + retries < MAX → Reasoning() ---
def test_empty_retry_increments_counter():
    state = make_state(empty_retries=0)
    prev = make_turn(text="")
    action = decide(state, prev)
    assert isinstance(action, Reasoning)
    assert action.hint is None
    assert action.force_no_tools is False
    assert state.empty_retries == 1


def test_empty_retry_at_max_minus_one():
    state = make_state(empty_retries=MAX_EMPTY_RESPONSE_RETRIES - 1)
    prev = make_turn(text="")
    action = decide(state, prev)
    assert isinstance(action, Reasoning)
    assert state.empty_retries == MAX_EMPTY_RESPONSE_RETRIES


# --- 优先级 6: 空响应 + retries 耗尽 → 最终化 ---
def test_empty_retries_exhausted_enters_finalization():
    state = make_state(empty_retries=MAX_EMPTY_RESPONSE_RETRIES)
    prev = make_turn(text="")
    action = decide(state, prev)
    assert isinstance(action, Reasoning)
    assert action.force_no_tools is True
    assert FINALIZATION_RETRY_PROMPT in (action.hint or "")
    assert state.in_finalization is True


# --- 优先级 7: 达到 max_iterations → Exit(EXCEED_MAX_ITERS) ---
def test_max_iterations_returns_exit_exceed():
    state = make_state(iteration=5, max_iterations=5)
    prev = None
    action = decide(state, prev)
    assert isinstance(action, Exit)
    assert action.finished_reason == ReplyFinishedReason.EXCEED_MAX_ITERS


# --- 优先级 8: 默认 → Reasoning() ---
def test_prev_none_returns_reasoning():
    state = make_state(iteration=0, max_iterations=10)
    action = decide(state, None)
    assert isinstance(action, Reasoning)
    assert action.hint is None


def test_prev_none_after_acting_returns_reasoning():
    state = make_state(iteration=1, max_iterations=10)
    action = decide(state, None)
    assert isinstance(action, Reasoning)
