"""
Token 使用统计

累计 Agent 与 LLM 交互过程中的 token 消耗。
纯累加器，不承担消息传递职责。

典型用法：
    token = TokenUsage()
    token.add(response.usage)       # 每次 LLM 调用后累加
    print(token.total_tokens)       # 查看累计消耗
    snapshot = token.to_dict()      # 序列化（用于 Checkpoint）
    token.restore(snapshot)         # 从快照恢复
"""
from __future__ import annotations


class TokenUsage:
    """
    Token 使用累加器

    支持两种输入格式：
    - OpenAI response.usage 对象（有 prompt_tokens 属性）
    - 普通字典（如从 Checkpoint 恢复时）
    """

    def __init__(self):
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._request_count = 0
        self._last_prompt_tokens = 0  # 最后一次 LLM 调用的 prompt_tokens（上下文大小）

    @property
    def prompt_tokens(self) -> int:
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self._completion_tokens

    @property
    def total_tokens(self) -> int:
        return self._prompt_tokens + self._completion_tokens

    @property
    def request_count(self) -> int:
        return self._request_count

    def add(self, usage) -> None:
        """累加一次 LLM 请求的 token 使用"""
        if usage is None:
            return

        prompt = 0
        completion = 0
        if hasattr(usage, "prompt_tokens"):
            prompt = usage.prompt_tokens or 0
            completion = usage.completion_tokens or 0
        elif isinstance(usage, dict):
            prompt = usage.get("prompt_tokens", 0)
            completion = usage.get("completion_tokens", 0)

        self._prompt_tokens += prompt
        self._completion_tokens += completion
        self._last_prompt_tokens = prompt
        self._request_count += 1

    def clear(self) -> None:
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._request_count = 0
        self._last_prompt_tokens = 0

    def restore(self, data: dict) -> None:
        """从字典恢复"""
        self._prompt_tokens = data.get("prompt_tokens", 0)
        self._completion_tokens = data.get("completion_tokens", 0)
        self._request_count = data.get("request_count", 0)
        self._last_prompt_tokens = data.get("last_prompt_tokens", 0)

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self._request_count,
            "last_prompt_tokens": self._last_prompt_tokens,
        }

    def __repr__(self) -> str:
        return (
            f"TokenUsage(prompt={self._prompt_tokens}, "
            f"completion={self._completion_tokens}, "
            f"total={self.total_tokens}, "
            f"requests={self._request_count})"
        )
