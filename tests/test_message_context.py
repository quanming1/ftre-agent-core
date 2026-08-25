"""MessageContext stateless helper tests."""
from ftre_agent_core.message import (
    HintBlock,
    Msg,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolResultBlock,
)
from ftre_agent_core.message_context import MessageContext
from ftre_agent_core.state import AgentState


def test_message_context_updates_agent_state_context():
    state = AgentState()

    MessageContext.add_user(state.context, "hello")
    MessageContext.append_reply_blocks(
        state.context,
        "reply-1",
        [ThinkingBlock(thinking="thinking"), TextBlock(text="world")],
    )

    assert all(isinstance(message, Msg) for message in state.context)
    assert [message.role for message in state.context] == ["user", "assistant"]
    assert MessageContext.messages(state.context)[1]["reasoning_content"] == "thinking"


def test_get_messages_prepends_system_prompt_without_storing_it():
    state = AgentState()
    MessageContext.add_user(state.context, "hello")

    messages = MessageContext.get_messages(state.context, "system")

    assert messages[0] == {"role": "system", "content": "system"}
    assert len(messages) == 2
    assert len(state.context) == 1


def test_set_and_clear_messages_mutate_the_supplied_context():
    state = AgentState()

    MessageContext.set_messages(
        state.context,
        [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}],
    )
    assert [message.get_text_content() for message in state.context] == ["one", "two"]

    MessageContext.clear(state.context)
    assert state.context == []


def test_aggregated_reply_splits_tool_results_at_provider_boundary():
    context = [
        Msg(
            role="assistant",
            content=[
                TextBlock(text="before"),
                ToolCallBlock(id="call-1", name="bash", arguments={"command": "echo ok"}),
                ToolResultBlock(
                    id="call-1",
                    name="bash",
                    output=[{"type": "text", "text": "tool output"}],
                    state="success",
                ),
                TextBlock(text="after"),
            ],
        )
    ]

    messages = MessageContext.messages(context)

    assert [message["role"] for message in messages] == ["assistant", "tool", "assistant"]
    assert messages[0]["tool_calls"][0]["id"] == "call-1"
    assert messages[1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "tool output",
    }
    assert messages[2]["content"] == [{"type": "text", "text": "after"}]


def test_append_reply_blocks_keeps_one_msg_and_splits_hint_as_user():
    context = []

    MessageContext.append_reply_blocks(
        context,
        "reply-1",
        [TextBlock(text="before")],
    )
    MessageContext.append_reply_blocks(
        context,
        "reply-1",
        [HintBlock(hint="继续工作。", source="system"), TextBlock(text="after")],
    )

    assert len(context) == 1
    assert context[0].id == "reply-1"
    assert [block.type for block in context[0].content] == ["text", "hint", "text"]
    assert MessageContext.messages(context) == [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "before"}],
            "reasoning_content": "",
        },
        {"role": "user", "content": "继续工作。"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "after"}],
            "reasoning_content": "",
        },
    ]


def test_non_user_hook_context_does_not_duplicate_assistant_message_id():
    """system/assistant Hook mapping 没有正式 UserMessage 时仍复用当前 Msg。"""
    context = [
        Msg(id="assistant-a", role="assistant", content=[TextBlock(text="before")]),
        Msg(id="system-1", role="system", content=[TextBlock(text="hint")]),
    ]

    MessageContext.append_reply_blocks(
        context,
        "assistant-a",
        [TextBlock(text="after")],
    )

    assert [message.id for message in context] == ["assistant-a", "system-1"]
    assert context[0].get_text_content() == "before\nafter"
