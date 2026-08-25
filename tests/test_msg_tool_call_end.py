"""Msg 事件重建的工具参数回归。

END 事件是工具参数的最终事实，防止实时 delta 丢失后持久化快照仍为空。
"""

from ftre_agent_core.event import (
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from ftre_agent_core.message import AssistantMsg


def test_tool_call_end_arguments_override_incomplete_deltas() -> None:
    msg = AssistantMsg(id="assistant-1")

    msg.append_event(
        ToolCallStartEvent(
            reply_id="reply-1",
            message_id="assistant-1",
            tool_call_id="call-1",
            tool_call_name="bash",
        )
    )
    msg.append_event(
        ToolCallDeltaEvent(
            reply_id="reply-1",
            message_id="assistant-1",
            tool_call_id="call-1",
            delta='{"command":"old',
        )
    )
    msg.append_event(
        ToolCallEndEvent(
            reply_id="reply-1",
            message_id="assistant-1",
            tool_call_id="call-1",
            arguments='{"command":"pnpm test","timeout":30}',
        )
    )

    block = msg.content[0]
    assert block.type == "tool_call"
    assert block.arguments == {"command": "pnpm test", "timeout": 30}


def test_tool_call_end_without_arguments_uses_legacy_delta_buffer() -> None:
    msg = AssistantMsg(id="assistant-1")

    msg.append_event(
        ToolCallStartEvent(
            reply_id="reply-1",
            message_id="assistant-1",
            tool_call_id="call-1",
            tool_call_name="bash",
        )
    )
    msg.append_event(
        ToolCallDeltaEvent(
            reply_id="reply-1",
            message_id="assistant-1",
            tool_call_id="call-1",
            delta='{"command":"pnpm test"}',
        )
    )
    msg.append_event(
        ToolCallEndEvent(
            reply_id="reply-1",
            message_id="assistant-1",
            tool_call_id="call-1",
        )
    )

    block = msg.content[0]
    assert block.type == "tool_call"
    assert block.arguments == {"command": "pnpm test"}
