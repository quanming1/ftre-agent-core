from ftre_agent_core.llm.wire.normalize import _normalize_chat_messages


def _assistant_call(*ids: str, content: str | None = None) -> dict:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "tool", "arguments": "{}"}}
            for call_id in ids
        ],
    }


def _result(call_id: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": "ok"}


def test_normalization_keeps_complete_adjacent_tool_call_group():
    messages = [_assistant_call("call-a", "call-b"), _result("call-a"), _result("call-b")]
    expected = [dict(messages[0], content=""), *messages[1:]]
    assert _normalize_chat_messages(messages) == expected


def test_normalization_drops_orphan_tool_result():
    messages = [{"role": "user", "content": "hello"}, _result("call-missing")]
    assert _normalize_chat_messages(messages) == [{"role": "user", "content": "hello"}]


def test_normalization_removes_calls_without_all_results_but_preserves_text():
    messages = [_assistant_call("call-a", content="I will check"), {"role": "user", "content": "continue"}]
    assert _normalize_chat_messages(messages) == [
        {"role": "assistant", "content": "I will check"},
        {"role": "user", "content": "continue"},
    ]


def test_normalization_drops_non_adjacent_result_and_empty_call_message():
    messages = [_assistant_call("call-a"), {"role": "user", "content": "continue"}, _result("call-a")]
    assert _normalize_chat_messages(messages) == [{"role": "user", "content": "continue"}]


def test_normalization_drops_empty_and_reasoning_only_assistant_messages():
    messages = [
        {"role": "assistant", "content": ""},
        {"role": "assistant", "reasoning_content": "private reasoning"},
        {"role": "user", "content": "continue"},
    ]
    assert _normalize_chat_messages(messages) == [
        {"role": "user", "content": "continue"}
    ]


def test_responses_normalization_preserves_reasoning_only_assistant():
    messages = [
        {"role": "assistant", "reasoning_content": "internal reasoning"},
        {"role": "user", "content": "continue"},
    ]

    assert _normalize_chat_messages(
        messages,
        preserve_reasoning_only=True,
    ) == [
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "internal reasoning",
        },
        {"role": "user", "content": "continue"},
    ]
