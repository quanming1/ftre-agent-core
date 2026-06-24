"""
MemoryManager - 对话消息管理器
"""
from typing import Any

from .reasoning import merge_reasoning_into_content


class MemoryManager:
    """管理单次 ReAct 循环中的消息列表。"""

    def __init__(self, options: dict = None):
        options = options or {}
        self._system_prompt = options.get("system_prompt", "你是一个有帮助的助手。")
        self._messages: list[dict] = []

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        self._system_prompt = value

    @property
    def messages(self) -> list[dict]:
        return self._messages

    def get_messages(self) -> list[dict]:
        """获取完整消息列表（含 system），用于发送给 LLM"""
        return [{"role": "system", "content": self._system_prompt}] + self._messages

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str, usage=None, reasoning: str | None = None) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        # 把 thinking 内容按 Zed 兼容格式合并进 assistant.content。
        merge_reasoning_into_content(msg, reasoning)
        self._messages.append(msg)

    def add_tool_result(self, tool_call_id: str, content: str, **kwargs) -> None:
        self._messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

    def add_raw(self, message: Any, usage=None) -> None:
        if hasattr(message, "model_dump"):
            self._messages.append(message.model_dump())
        else:
            self._messages.append(message)

    def clear(self) -> None:
        self._messages = []

    def set_messages(self, messages: list[dict]) -> None:
        """设置完整消息列表（用于恢复历史上下文）"""
        self._messages = list(messages)

    def __len__(self) -> int:
        return len(self._messages)
