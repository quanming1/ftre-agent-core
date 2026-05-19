"""
MemoryManager - 对话消息管理器

职责：
- 管理当前 ReAct 循环的消息列表（messages）
- 维护 system prompt
- 统计 token 使用
"""
from typing import Any
from .token import TokenUsage


DEFAULT_MAX_MESSAGES = 100


class MemoryManager:
    """
    对话消息管理器

    管理单次 ReAct 循环中的消息列表和 token 统计。
    """

    def __init__(self, options: dict = None):
        options = options or {}
        self._system_prompt = options.get("system_prompt", "你是一个有帮助的助手。")
        self._max_messages = options.get("max_messages", DEFAULT_MAX_MESSAGES)
        self._messages: list[dict] = []
        self.token = TokenUsage()

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        self._system_prompt = value

    @property
    def messages(self) -> list[dict]:
        """获取消息列表（不含 system）"""
        return self._messages

    def get_messages(self) -> list[dict]:
        """获取完整消息列表（含 system），用于发送给 LLM"""
        return [{"role": "system", "content": self._system_prompt}] + self._messages

    # ============================================================
    # 消息操作
    # ============================================================

    def add_user(self, content: str) -> None:
        """添加用户消息"""
        self._append({"role": "user", "content": content})

    def add_assistant(self, content: str, usage=None) -> None:
        """添加助手消息"""
        self._append({"role": "assistant", "content": content})

    def add_tool_result(self, tool_call_id: str, content: str, **kwargs) -> None:
        """添加工具结果"""
        self._append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

    def add_raw(self, message: Any, usage=None) -> None:
        """添加原始消息（OpenAI 返回的 message 对象）"""
        if hasattr(message, "model_dump"):
            self._append(message.model_dump())
        else:
            self._append(message)

    def clear(self) -> None:
        """清空消息"""
        self._messages = []
        self.token.clear()

    # ============================================================
    # 内部方法
    # ============================================================

    def _append(self, message: dict) -> None:
        """内部追加消息"""
        self._messages.append(message)

    def __len__(self) -> int:
        return len(self._messages)
