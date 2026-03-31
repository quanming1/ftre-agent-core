"""
MemoryManager - 对话记忆管理器

职责：
- 管理对话消息列表（messages）
- 维护 system prompt
- 统计 token 使用
- 通过 CheckpointManager 管理快照

Checkpoint state key 约定：
- "messages":    list[dict]  — 消息列表快照
- "token_usage": dict        — token 统计快照
"""
from typing import Any
from .types import MemoryOptions
from .token import TokenUsage
from packages.core.checkpoint import CheckpointManager, Checkpoint, CheckpointType


DEFAULT_MAX_MESSAGES = 100

# Checkpoint state 中的 key 常量，避免魔法字符串
STATE_KEY_MESSAGES = "messages"
STATE_KEY_TOKEN_USAGE = "token_usage"


class MemoryManager:
    """
    对话记忆管理器

    管理对话消息、token 统计、checkpoint 快照。
    checkpoint 的状态收集和恢复逻辑集中在这里，
    CheckpointManager 本身只负责存储和检索。
    """

    def __init__(self, options: MemoryOptions = None):
        options = options or {}
        self._system_prompt = options.get("system_prompt", "你是一个有帮助的助手。")
        self._max_messages = options.get("max_messages", DEFAULT_MAX_MESSAGES)
        self._messages: list[dict] = []
        self.token = TokenUsage()
        self.checkpoints = CheckpointManager()

        # 对话轮次计数（每次 add_user 时 +1）
        self._turn: int = 0

    @property
    def turn(self) -> int:
        """当前对话轮次"""
        return self._turn

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        self._system_prompt = value

    @property
    def messages(self) -> list[dict]:
        """获取消息列表（不含 system），用于快照等内部操作"""
        return self._messages

    def get_messages(self) -> list[dict]:
        """获取完整消息列表（含 system），用于发送给 LLM"""
        return [{"role": "system", "content": self._system_prompt}] + self._messages

    # ============================================================
    # 消息操作
    # ============================================================

    def add_user(self, content: str) -> None:
        """添加用户消息，同时递增轮次"""
        self._turn += 1
        self._append({"role": "user", "content": content})

    def add_assistant(self, content: str, usage=None) -> None:
        """添加助手消息"""
        self._append({"role": "assistant", "content": content})

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """添加工具结果"""
        self._append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

    def add_raw(self, message: Any, usage=None) -> None:
        """添加原始消息（OpenAI 返回的 message 对象）"""
        if hasattr(message, "model_dump"):
            self._append(message.model_dump())
        else:
            self._append(message)

    # ============================================================
    # Checkpoint 操作
    # ============================================================

    def _collect_state(self) -> dict:
        """
        收集当前状态为字典（用于创建 checkpoint）

        这里定义了"哪些东西需要被快照"。
        未来新增状态只需在这里加 key。
        """
        return {
            STATE_KEY_MESSAGES: self._messages,
            STATE_KEY_TOKEN_USAGE: self.token.to_dict(),
        }

    def _apply_state(self, state: dict) -> None:
        """
        从状态字典恢复（用于 restore checkpoint）

        与 _collect_state 对应，定义了"怎么从快照恢复"。
        """
        self._messages = state.get(STATE_KEY_MESSAGES, [])

        token_data = state.get(STATE_KEY_TOKEN_USAGE)
        self.token.clear()
        if token_data:
            self.token.restore(token_data)

    def save_checkpoint(
        self,
        label: str = "",
        type: CheckpointType = CheckpointType.AUTO,
        metadata: dict | None = None,
    ) -> Checkpoint:
        """
        保存当前状态为快照

        Args:
            label:    可选标签（方便识别）
            type:     快照类型（auto/manual/interrupt）
            metadata: 可选元数据

        Returns:
            新创建的 Checkpoint
        """
        return self.checkpoints.save(
            turn=self._turn,
            state=self._collect_state(),
            type=type,
            label=label,
            metadata=metadata,
        )

    def restore_checkpoint(self, checkpoint_id: str) -> Checkpoint:
        """
        恢复到指定快照

        会截断消息列表和后续快照，token 统计恢复到快照时的值。

        Args:
            checkpoint_id: 快照 ID

        Returns:
            恢复使用的 Checkpoint

        Raises:
            ValueError: 快照不存在
        """
        cp = self.checkpoints.restore(checkpoint_id)
        self._turn = cp.turn
        self._apply_state(cp.state)
        return cp

    # ============================================================
    # 内部方法
    # ============================================================

    def _append(self, message: dict) -> None:
        """内部追加消息"""
        self._messages.append(message)

    def after_loop(self) -> None:
        """循环结束钩子（默认空操作，子类可覆盖）"""
        pass

    def clear(self, clear_token: bool = False, clear_checkpoints: bool = False) -> None:
        """清空消息"""
        self._messages = []
        self._turn = 0
        if clear_token:
            self.token.clear()
        if clear_checkpoints:
            self.checkpoints.clear()
    def __len__(self) -> int:
        return len(self._messages)
