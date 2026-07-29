from ftre_agent_core.event import EventType, UserMessageEvent


def test_user_message_event_is_core_event_and_hides_projection_fields_on_wire():
    event = UserMessageEvent(
        reply_id="turn_1",
        data={"content": "hello"},
        content=[{"type": "text", "text": "hello"}],
        message_metadata={"hide": False},
    )

    payload = event.model_dump(mode="json")
    assert event.type == EventType.USER_MESSAGE
    assert payload["type"] == "USER_MESSAGE"
    assert payload["data"] == {"content": "hello"}
    assert "content" not in payload
    assert "message_metadata" not in payload
