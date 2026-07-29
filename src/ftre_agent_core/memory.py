"""
MemoryManager - 对话消息管理器

内部存 list[Msg]（对齐 AgentScope AgentState.context），
对外接口保持 OpenAI dict 兼容（get_messages/add_*/messages property）。
react_runner 调用接口零改动。
"""
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
    """Msg → OpenAI dict（保 ftre 现有格式）。

    用 to_openai_message 转换 content blocks，再补 reasoning_content（add_assistant
    把 reasoning 存 metadata，不进 content，避免 content 多 thinking part）。

    含 ToolResultBlock 时传 role=None，让 to_openai_message 自动拆回 {role:tool} 消息。
    """
    has_tool_result = any(isinstance(b, ToolResultBlock) for b in msg.content)
    role = None if has_tool_result else msg.role
    d = to_openai_message(msg.content, role=role)
    rc = msg.metadata.get("reasoning_content")
    if rc and "reasoning_content" not in d:
        d["reasoning_content"] = rc
    return d


class MemoryManager:
    """管理单次 ReAct 循环中的消息列表（内部 list[Msg]）。"""

    def __init__(self, options: dict = None):
        options = options or {}
        self._system_prompt = options.get("system_prompt", "你是一个有帮助的助手。")
        self._messages: list[Msg] = []

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        self._system_prompt = value

    @property
    def messages(self) -> list[dict]:
        """返回 OpenAI dict 列表（保兼容，react_runner/tests 在用）。"""
        return [_msg_to_dict(m) for m in self._messages]

    def get_messages(self) -> list[dict]:
        """获取完整消息列表（含 system），用于发送给 LLM"""
        return [{"role": "system", "content": self._system_prompt}] + self.messages

    def add_user(self, content: str) -> None:
        self._messages.append(
            Msg(name=MsgName.DEFAULT, content=[TextBlock(text=content)], role="user")
        )

    def add_assistant(
        self,
        content: str,
        usage: dict | None = None,
        reasoning: str | None = None,
    ) -> None:
        """添加 assistant 消息。reasoning 存 metadata（不进 content，保 ftre 格式）。"""
        blocks = [TextBlock(text=content)] if content else []
        metadata: dict[str, Any] = {}
        if reasoning:
            metadata["reasoning_content"] = reasoning
        self._messages.append(
            Msg(name=MsgName.DEFAULT, content=blocks, role="assistant", metadata=metadata)
        )

    def add_tool_result(self, tool_call_id: str, content: str, **kwargs) -> None:
        """工具结果 → Msg(content=[ToolResultBlock])，输出时 to_openai 拆回 {role:tool}。"""
        self._messages.append(
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

    def add_raw(self, message: Any, usage=None) -> None:
        """追加原始消息（dict 或 Msg）。

        dict → from_openai_message 转 blocks → Msg；
        Msg → 直接 append。
        """
        if isinstance(message, Msg):
            self._messages.append(message)
        elif isinstance(message, dict):
            blocks = from_openai_message(message)
            role = message.get("role", "user")
            # tool 等 non-standard role → assistant（content 含 ToolResultBlock，
            # to_openai_message 输出时会自动拆回 {role:tool})
            if role not in ("user", "assistant", "system"):
                role = "assistant"
            self._messages.append(
                Msg(name=MsgName.DEFAULT, content=blocks, role=role)
            )
        else:
            # 兜底：原样存（不应该走到，但保不崩）
            self._messages.append(message)

    def clear(self) -> None:
        self._messages = []

    def set_messages(self, messages: list[dict]) -> None:
        """设置完整消息列表（用于恢复历史上下文）。

        接受 OpenAI dict 列表，逐条转 Msg。
        """
        self._messages = []
        for msg_dict in messages:
            self.add_raw(msg_dict)

    def __len__(self) -> int:
        return len(self._messages)
