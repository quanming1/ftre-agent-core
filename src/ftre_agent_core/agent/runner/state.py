"""
RunState - 运行时状态管理

记录 ReAct 循环的执行位置，支持中断/恢复。

中断时保存的关键信息：
- iteration:          当前迭代轮次
- pending_tool_calls: LLM 返回的待执行工具列表
- tool_call_index:    已执行到第几个工具（resume 时从这里继续）
"""
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Any

from packages.core.tool_system import CancellationToken, ToolCancelledError

class RunStatus(str, Enum):
    """运行状态"""
    IDLE = "idle"              # 空闲，未开始
    RUNNING = "running"        # 运行中
    INTERRUPTED = "interrupted"  # 已中断，等待用户确认
    COMPLETED = "completed"    # 已完成
    ERROR = "error"            # 出错
    CANCELLED = "cancelled"    # 用户主动取消

class CancelledError(Exception):
    """用户主动取消时抛出，由最外层统一捕获处理"""
    pass

@dataclass
class RunState:
    """
    运行时状态

    记录当前执行的位置和状态，支持中断/恢复。
    中断时 pending_tool_calls 和 tool_call_index 保留现场，
    resume 时从 tool_call_index 处继续执行。
    """
    status: RunStatus = RunStatus.IDLE
    iteration: int = 0
    pending_tool_calls: list[Any] = field(default_factory=list)
    tool_call_index: int = 0
    interrupt_checkpoint_id: str | None = None
    error: str | None = None
    cancel_token: CancellationToken = field(default_factory=CancellationToken)
    _done_event: threading.Event = field(default_factory=threading.Event)

    @property
    def is_idle(self) -> bool:
        return self.status == RunStatus.IDLE

    @property
    def is_running(self) -> bool:
        return self.status == RunStatus.RUNNING

    @property
    def is_interrupted(self) -> bool:
        return self.status == RunStatus.INTERRUPTED

    @property
    def is_cancelled(self) -> bool:
        return self.status == RunStatus.CANCELLED

    @property
    def is_done(self) -> bool:
        return self.status in (RunStatus.COMPLETED, RunStatus.ERROR, RunStatus.CANCELLED)

    def start(self) -> None:
        self.status = RunStatus.RUNNING
        self.iteration = 0
        self.pending_tool_calls = []
        self.tool_call_index = 0
        self.interrupt_checkpoint_id = None
        self.error = None
        self.cancel_token = CancellationToken()
        self._done_event.clear()

    def next_iteration(self) -> None:
        self.iteration += 1
        self.pending_tool_calls = []
        self.tool_call_index = 0

    def interrupt(self, checkpoint_id: str) -> None:
        self.status = RunStatus.INTERRUPTED
        self.interrupt_checkpoint_id = checkpoint_id

    def resume(self) -> None:
        self.status = RunStatus.RUNNING
        self.interrupt_checkpoint_id = None
        self.cancel_token = CancellationToken()
        self._done_event.clear()

    def complete(self) -> None:
        self.status = RunStatus.COMPLETED
        self._done_event.set()

    def cancel(self) -> None:
        if not self.is_running:
            self._done_event.set()
            return
        self.status = RunStatus.CANCELLED
        self.cancel_token.cancel("user_cancelled")

    def mark_done(self) -> None:
        self._done_event.set()

    def wait_done(self, timeout: float | None = None) -> bool:
        return self._done_event.wait(timeout)

    def fail(self, error: str) -> None:
        self.status = RunStatus.ERROR
        self.error = error
        self._done_event.set()

    def reset(self) -> None:
        self.status = RunStatus.IDLE
        self.iteration = 0
        self.pending_tool_calls = []
        self.tool_call_index = 0
        self.interrupt_checkpoint_id = None
        self.error = None
        self.cancel_token = CancellationToken()
        self._done_event.clear()

    def wait_or_cancelled(self, timeout: float) -> bool:
        return self.cancel_token.wait(timeout)

    def check_cancel(self) -> None:
        try:
            self.cancel_token.raise_if_cancelled()
        except ToolCancelledError as exc:
            raise CancelledError(str(exc)) from exc

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "iteration": self.iteration,
            "tool_call_index": self.tool_call_index,
            "interrupt_checkpoint_id": self.interrupt_checkpoint_id,
        }
