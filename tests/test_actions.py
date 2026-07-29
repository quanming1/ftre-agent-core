# tests/test_actions.py
"""动作模型与数据载体单元测试。"""
import pytest
from ftre_agent_core.agent.runner._state import (
    Reasoning, Acting, Exit, TurnResult, ExitOutcome,
)
from ftre_agent_core.llm import ToolCall
from ftre_agent_core.types import ReplyFinishedReason


def test_reasoning_defaults():
    r = Reasoning()
    assert r.hint is None
    assert r.tool_choice is None
    assert r.force_no_tools is False


def test_reasoning_with_fields():
    r = Reasoning(hint="test hint", tool_choice="my_tool", force_no_tools=True)
    assert r.hint == "test hint"
    assert r.tool_choice == "my_tool"
    assert r.force_no_tools is True


def test_acting():
    tc = ToolCall(id="c1", name="echo", input={"text": "hi"})
    a = Acting(tool_calls=[tc])
    assert len(a.tool_calls) == 1
    assert a.tool_calls[0].id == "c1"


def test_exit_completed():
    e = Exit(finished_reason=ReplyFinishedReason.COMPLETED)
    assert e.finished_reason == ReplyFinishedReason.COMPLETED
    assert e.exit_msg is None
    assert e.error is None
    assert e.error_code is None


def test_exit_error():
    e = Exit(
        finished_reason=ReplyFinishedReason.ERROR,
        error="something broke",
        error_code="empty_response",
    )
    assert e.finished_reason == ReplyFinishedReason.ERROR
    assert e.error == "something broke"
    assert e.error_code == "empty_response"


def test_turn_result_defaults():
    tr = TurnResult(text="hello", reasoning="", tool_calls=[], finish_reason="stop")
    assert tr.text == "hello"
    assert tr.reasoning == ""
    assert tr.tool_calls == []
    assert tr.finish_reason == "stop"
    assert tr.usage is None
    assert tr.error is None


def test_exit_outcome_defaults():
    eo = ExitOutcome()
    assert eo.should_continue is False
    assert eo.continue_hint is None


def test_exit_outcome_continue():
    eo = ExitOutcome(should_continue=True, continue_hint="keep going")
    assert eo.should_continue is True
    assert eo.continue_hint == "keep going"
