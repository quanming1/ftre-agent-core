"""
Memory 协议定义

定义 Runner 对 Memory 管理器的最小依赖接口。
任何实现了此协议的类都可以作为 Agent 的 memory。

默认实现: MemoryManager (内存版, 原有逻辑)
自定义实现: 业务层可以基于 MongoDB/FtreMessage 等实现自己的版本
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .token import TokenUsage
from ftre_agent_core.checkpoint import Checkpoint, CheckpointType


@runtime_checkable
class MemoryProtocol(Protocol):
    """
    Memory 管理器协议

    Runner 只依赖这个接口，不依赖具体实现。
    """

    token: TokenUsage

    @property
    def system_prompt(self) -> str: ...

    @system_prompt.setter
    def system_prompt(self, value: str) -> None: ...

    @property
    def turn(self) -> int: ...

    @property
    def messages(self) -> list: ...

    def get_messages(self) -> list[dict]:
        """获取完整消息列表 (含 system)，用于发送给 LLM"""
        ...

    def add_user(self, content: str) -> None: ...

    def add_assistant(self, content: str, usage=None) -> None: ...

    def add_tool_result(self, tool_call_id: str, content: str, **kwargs) -> None:
        """
        添加工具结果

        kwargs 允许传入额外参数 (如 tool_name)，
        默认实现可以忽略，自定义实现按需使用。
        """
        ...

    def add_raw(self, message: Any, usage=None) -> None: ...

    def save_checkpoint(
        self,
        label: str = "",
        type: CheckpointType = CheckpointType.AUTO,
        metadata: dict | None = None,
    ) -> Checkpoint: ...

    def restore_checkpoint(self, checkpoint_id: str) -> Checkpoint: ...

    def after_loop(self) -> None:
        """
        ReAct 循环结束后调用

        用于触发后处理逻辑（如中间件链的 on_loop_end）。
        默认实现可以为空操作。
        """
        ...

    def compact(self) -> bool:
        """
        上下文压缩 (Compaction)

        检测 token 用量是否超过阈值，超过则生成摘要消息替代旧历史。

        Returns:
            True = 执行了压缩，False = 未触发
        """
        ...

    def clear(self, clear_token: bool = False, clear_checkpoints: bool = False) -> None: ...
