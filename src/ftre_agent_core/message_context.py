"""Stateless helpers for reading and updating ``AgentState.context``."""
from typing import Any

from .message import (
    Msg,
    MsgName,
    TextBlock,
    ToolResultBlock,
    ToolResultState,
    from_openai_message,
    to_openai_message,
)


def _msg_to_dict(msg: Msg) -> dict:
    """Convert a typed message to the provider-compatible dictionary format."""
    has_tool_result = any(isinstance(block, ToolResultBlock) for block in msg.content)
    role = None if has_tool_result else msg.role
    result = to_openai_message(msg.content, role=role)
    reasoning = msg.metadata.get("reasoning_content")
    if reasoning and "reasoning_content" not in result:
        result["reasoning_content"] = reasoning
    return result


class MessageContext:
    """Operate on a caller-owned ``list[Msg]`` without storing state itself."""

    @staticmethod
    def messages(context: list[Msg]) -> list[dict]:
        return [_msg_to_dict(message) for message in context]

    @staticmethod
    def get_messages(context: list[Msg], system_prompt: str = "") -> list[dict]:
        return [
            {"role": "system", "content": system_prompt},
            *MessageContext.messages(context),
        ]

    @staticmethod
    def add_user(context: list[Msg], content: str) -> None:
        context.append(
            Msg(name=MsgName.DEFAULT, content=[TextBlock(text=content)], role="user")
        )

    @staticmethod
    def add_assistant(
        context: list[Msg],
        content: str,
        usage: dict | None = None,
        reasoning: str | None = None,
    ) -> None:
        blocks = [TextBlock(text=content)] if content else []
        metadata: dict[str, Any] = {}
        if reasoning:
            metadata["reasoning_content"] = reasoning
        context.append(
            Msg(name=MsgName.DEFAULT, content=blocks, role="assistant", metadata=metadata)
        )

    @staticmethod
    def add_tool_result(
        context: list[Msg], tool_call_id: str, content: str, **kwargs
    ) -> None:
        context.append(
            Msg(
                name=MsgName.DEFAULT,
                content=[
                    ToolResultBlock(
                        id=tool_call_id,
                        name=kwargs.get("name", ""),
                        output=content,
                        state=ToolResultState.SUCCESS,
                    )
                ],
                role="assistant",
            )
        )

    @staticmethod
    def add_raw(context: list[Msg], message: Any, usage=None) -> None:
        if isinstance(message, Msg):
            context.append(message)
            return
        if isinstance(message, dict):
            blocks = from_openai_message(message)
            role = message.get("role", "user")
            if role not in ("user", "assistant", "system"):
                role = "assistant"
            context.append(Msg(name=MsgName.DEFAULT, content=blocks, role=role))
            return
        context.append(message)

    @staticmethod
    def clear(context: list[Msg]) -> None:
        context.clear()

    @staticmethod
    def set_messages(context: list[Msg], messages: list[dict | Msg]) -> None:
        context.clear()
        for message in messages:
            MessageContext.add_raw(context, message)
