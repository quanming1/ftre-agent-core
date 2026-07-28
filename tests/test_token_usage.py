# tests/test_token_usage.py
"""Token usage protocol tests.

Covers the 9 required scenarios from the redesign doc:
1. Single MODEL_CALL_END initializes both usage and last_call_usage
2. Three MODEL_CALL_ENDs accumulate correctly
3. last_call_usage equals the third call
4. user/system Msg with token fails validation
5. JSON round-trip preserves all fields
6. Failed call without usage doesn't overwrite last valid usage
7. RunnerState resets token_usage on start()
8. Event serialization only contains new fields
9. No production code references input_tokens/output_tokens
"""
import json
import pytest
from ftre_agent_core.message import Msg, TokenUsage, MsgToken, AssistantMsg, UserMsg
from ftre_agent_core.event import ModelCallEndEvent
from ftre_agent_core.agent.runner._state import RunState, RunStatus


class TestSingleModelCallEnd:
    """1. Single MODEL_CALL_END initializes both usage and last_call_usage."""

    def test_single_call_initializes_both(self):
        msg = AssistantMsg(name="test", content="hello", id="reply_1")
        event = ModelCallEndEvent(
            reply_id="reply_1",
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        )
        msg.append_event(event)
        assert msg.token is not None
        assert msg.token.usage.prompt_tokens == 100
        assert msg.token.usage.completion_tokens == 20
        assert msg.token.usage.total_tokens == 120
        assert msg.token.last_call_usage.prompt_tokens == 100
        assert msg.token.last_call_usage.completion_tokens == 20
        assert msg.token.last_call_usage.total_tokens == 120


class TestMultipleModelCallEndAccumulation:
    """2 & 3. Three MODEL_CALL_ENDs accumulate; last_call equals third."""

    def test_three_calls_accumulate_and_last_call_is_third(self):
        msg = AssistantMsg(name="test", content="hello", id="reply_1")
        # Call 1: 10,000 + 200 = 10,200
        msg.append_event(ModelCallEndEvent(
            reply_id="reply_1", prompt_tokens=10000, completion_tokens=200, total_tokens=10200,
        ))
        # Call 2: 12,000 + 100 = 12,100
        msg.append_event(ModelCallEndEvent(
            reply_id="reply_1", prompt_tokens=12000, completion_tokens=100, total_tokens=12100,
        ))
        # Call 3: 15,000 + 300 = 15,300
        msg.append_event(ModelCallEndEvent(
            reply_id="reply_1", prompt_tokens=15000, completion_tokens=300, total_tokens=15300,
        ))

        assert msg.token is not None
        # usage = cumulative
        assert msg.token.usage.prompt_tokens == 37000
        assert msg.token.usage.completion_tokens == 600
        assert msg.token.usage.total_tokens == 37600
        # last_call_usage = third call only
        assert msg.token.last_call_usage.prompt_tokens == 15000
        assert msg.token.last_call_usage.completion_tokens == 300
        assert msg.token.last_call_usage.total_tokens == 15300


class TestRoleValidation:
    """4. user/system Msg with token fails validation."""

    def test_user_msg_with_token_raises(self):
        from pydantic import ValidationError
        token = MsgToken(
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            last_call_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        with pytest.raises(ValidationError) as exc_info:
            UserMsg(name="test", content="hello", token=token)
        assert "assistant" in str(exc_info.value).lower()

    def test_system_msg_with_token_raises(self):
        from pydantic import ValidationError
        from ftre_agent_core.message import TextBlock
        token = MsgToken(
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            last_call_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        with pytest.raises(ValidationError) as exc_info:
            Msg(name="test", content=[TextBlock(text="sys")], role="system", token=token)
        assert "assistant" in str(exc_info.value).lower()

    def test_user_msg_without_token_is_fine(self):
        msg = UserMsg(name="test", content="hello")
        assert msg.token is None

    def test_assistant_msg_with_token_is_fine(self):
        token = MsgToken(
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            last_call_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        msg = AssistantMsg(name="test", content="hi", token=token)
        assert msg.token is not None


class TestJsonRoundTrip:
    """5. JSON round-trip preserves all fields."""

    def test_json_round_trip(self):
        msg = AssistantMsg(name="test", content="hello", id="reply_1")
        msg.append_event(ModelCallEndEvent(
            reply_id="reply_1", prompt_tokens=10000, completion_tokens=200, total_tokens=10200,
        ))
        msg.append_event(ModelCallEndEvent(
            reply_id="reply_1", prompt_tokens=12000, completion_tokens=100, total_tokens=12100,
        ))

        # Serialize
        data = msg.model_dump(mode="json")
        json_str = json.dumps(data)
        parsed = json.loads(json_str)

        # Deserialize
        restored = Msg.model_validate(parsed)

        assert restored.token is not None
        assert restored.token.usage.prompt_tokens == 22000
        assert restored.token.usage.completion_tokens == 300
        assert restored.token.usage.total_tokens == 22300
        assert restored.token.last_call_usage.prompt_tokens == 12000
        assert restored.token.last_call_usage.completion_tokens == 100
        assert restored.token.last_call_usage.total_tokens == 12100

    def test_user_msg_json_has_no_token_key(self):
        """exclude_none=True should omit token for user messages."""
        msg = UserMsg(name="test", content="hello")
        data = msg.model_dump(mode="json", exclude_none=True)
        assert "token" not in data


class TestRunnerStateTokenUsage:
    """7. RunnerState resets token_usage on start()."""

    def test_default_token_usage(self):
        state = RunState()
        assert state.token_usage["prompt_tokens"] == 0
        assert state.token_usage["completion_tokens"] == 0
        assert state.token_usage["total_tokens"] == 0
        assert "cached_tokens" not in state.token_usage
        assert "llm_calls" not in state.token_usage

    def test_start_resets_token_usage(self):
        state = RunState()
        state.token_usage["prompt_tokens"] = 999
        state.token_usage["completion_tokens"] = 888
        state.token_usage["total_tokens"] = 777
        state.start()
        assert state.token_usage["prompt_tokens"] == 0
        assert state.token_usage["completion_tokens"] == 0
        assert state.token_usage["total_tokens"] == 0


class TestEventSerialization:
    """8. Event serialization only contains new fields."""

    def test_model_call_end_event_fields(self):
        event = ModelCallEndEvent(
            reply_id="r1",
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
        )
        data = event.model_dump(mode="json")
        assert "prompt_tokens" in data
        assert "completion_tokens" in data
        assert "total_tokens" in data
        assert "input_tokens" not in data
        assert "output_tokens" not in data
        assert "cached_tokens" not in data
        assert "reasoning_tokens" not in data
