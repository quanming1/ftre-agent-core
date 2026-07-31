"""Stateless helpers for reading and updating ``AgentState.context``."""
from typing import Any

from .message import (
    Msg,
    MsgName,
    TextBlock,
    ToolCallBlock,
    ToolCallState,
    ToolResultBlock,
    ToolResultState,
    from_openai_message,
    to_openai_message,
)


def _msg_to_dicts(msg: Msg) -> list[dict]:
    """Expand one typed Msg into provider-compatible protocol messages.

    A persisted assistant reply may aggregate several reasoning/tool rounds in one
    ``Msg``. Provider protocols require every ``ToolResultBlock`` to be a separate
    ``role=tool`` message, so split the aggregate while preserving block order.
    """
    messages: list[dict] = []
    assistant_blocks: list = []
    metadata_reasoning = msg.metadata.get("reasoning_content")
    metadata_reasoning_used = False

    def flush_assistant() -> None:
        nonlocal metadata_reasoning_used
        if not assistant_blocks:
            return
        result = to_openai_message(assistant_blocks, role=msg.role)
        if not metadata_reasoning_used:
            if metadata_reasoning and "reasoning_content" not in result:
                result["reasoning_content"] = metadata_reasoning
            metadata_reasoning_used = True
        messages.append(result)
        assistant_blocks.clear()

    for block in msg.content:
        if isinstance(block, ToolResultBlock):
            flush_assistant()
            messages.append(to_openai_message([block], role="tool"))
        else:
            assistant_blocks.append(block)
    flush_assistant()

    # Preserve empty messages for compatibility; the provider boundary may drop
    # truly empty assistant messages when required by a concrete API.
    if not messages:
        result = to_openai_message([], role=msg.role)
        if metadata_reasoning:
            result["reasoning_content"] = metadata_reasoning
        messages.append(result)
    return messages


class MessageContext:
    """Operate on a caller-owned ``list[Msg]`` without storing state itself."""

    @staticmethod
    def messages(context: list[Msg]) -> list[dict]:
        return [item for message in context for item in _msg_to_dicts(message)]

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
        context: list[Msg],
        tool_call_id: str,
        content: str,
        state: ToolResultState = ToolResultState.SUCCESS,
        **kwargs,
    ) -> None:
        context.append(
            Msg(
                name=MsgName.DEFAULT,
                content=[
                    ToolResultBlock(
                        id=tool_call_id,
                        name=kwargs.get("name", ""),
                        output=content,
                        state=state,
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
    def set_tool_call_state(
        context: list[Msg], tool_call_id: str, state: ToolCallState
    ) -> bool:
        """把 context 里指定 tool_call 的 ToolCallBlock 状态改为 ``state``。

        权限流程用它标记工具调用的生命周期：挂起时置 ASKING，用户确认后置
        ALLOWED，拒绝或了结后置 FINISHED。返回是否找到并修改了对应 block。
        """
        for message in context:
            for block in message.content:
                if isinstance(block, ToolCallBlock) and block.id == tool_call_id:
                    block.state = state
                    return True
        return False

    @staticmethod
    def tool_calls_in_state(
        context: list[Msg], state: ToolCallState
    ) -> list[ToolCallBlock]:
        """收集 context 里处于指定状态的全部 ToolCallBlock（保持出现顺序）。

        用于恢复阶段：读出仍 ASKING 的调用判断是否还需暂停，或读出 ALLOWED
        的调用交给 resume_execute 整批执行。
        """
        found: list[ToolCallBlock] = []
        for message in context:
            for block in message.content:
                if isinstance(block, ToolCallBlock) and block.state == state:
                    found.append(block)
        return found

    @staticmethod
    def clear(context: list[Msg]) -> None:
        context.clear()

    @staticmethod
    def set_messages(context: list[Msg], messages: list[dict | Msg]) -> None:
        context.clear()
        for message in messages:
            MessageContext.add_raw(context, message)
