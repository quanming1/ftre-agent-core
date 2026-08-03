from __future__ import annotations

from ftre_agent_core.message import TextBlock, ThinkingBlock, ToolCallBlock
from ftre_agent_core.message_context import MessageContext
from ftre_agent_core.state import AgentState


def test_message_context_keeps_reasoning_separate_from_content():
    state = AgentState()

    MessageContext.append_reply_blocks(
        state.context,
        "reply-1",
        [ThinkingBlock(thinking="thinking"), TextBlock(text="answer")],
    )

    assert MessageContext.messages(state.context) == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "answer"},
            ],
            "reasoning_content": "thinking",
        }
    ]


def test_reply_blocks_preserve_reasoning_with_tool_call():
    state = AgentState()
    MessageContext.append_reply_blocks(
        state.context,
        "reply-1",
        [
            ThinkingBlock(thinking="thinking"),
            ToolCallBlock(
                id="call_1",
                name="bash",
                arguments={"command": "pwd"},
            ),
        ],
    )
    msg = MessageContext.messages(state.context)[0]

    assert msg["content"] == ""
    assert msg["reasoning_content"] == "thinking"
    assert msg["tool_calls"][0]["id"] == "call_1"


def test_reply_blocks_preserve_visible_content_with_tool_call():
    state = AgentState()
    MessageContext.append_reply_blocks(
        state.context,
        "reply-1",
        [
            ThinkingBlock(thinking="thinking"),
            TextBlock(text="visible answer"),
            ToolCallBlock(
                id="call_1",
                name="bash",
                arguments={"command": "pwd"},
            ),
        ],
    )
    msg = MessageContext.messages(state.context)[0]

    assert msg["content"] == [{"type": "text", "text": "visible answer"}]
    assert msg["reasoning_content"] == "thinking"
    assert msg["tool_calls"][0]["id"] == "call_1"


def test_reply_blocks_keep_empty_reasoning_field_without_thinking():
    state = AgentState()
    MessageContext.append_reply_blocks(
        state.context,
        "reply-1",
        [
            TextBlock(text="visible answer"),
            ToolCallBlock(
                id="call_1",
                name="bash",
                arguments={"command": "pwd"},
            ),
        ],
    )
    msg = MessageContext.messages(state.context)[0]

    assert msg["content"] == [{"type": "text", "text": "visible answer"}]
    assert msg["reasoning_content"] == ""
    assert msg["tool_calls"][0]["id"] == "call_1"
