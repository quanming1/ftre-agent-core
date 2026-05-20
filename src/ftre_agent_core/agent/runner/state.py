"""
RunState - 运行时状态
"""
import threading
from enum import Enum
from dataclasses import dataclass, field

from ftre_agent_core.tool_system import CancellationToken, ToolCancelledError


class RunStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class CancelledError(Exception):
    """用户取消时抛出，由 _loop() 统一捕获"""
    pass


@dataclass
class RunState:
    status: RunStatus = RunStatus.IDLE
    iteration: int = 0
    error: str | None = None
    cancel_token: CancellationToken = field(default_factory=CancellationToken)
    _done_event: threading.Event = field(default_factory=threading.Event)

    @property
    def is_running(self) -> bool:
        return self.status == RunStatus.RUNNING

    @property
    def is_cancelled(self) -> bool:
        return self.status == RunStatus.CANCELLED

    @property
    def is_done(self) -> bool:
        return self.status in (RunStatus.COMPLETED, RunStatus.ERROR, RunStatus.CANCELLED)

    def start(self) -> None:
        """开始新一轮执行"""
        self.status = RunStatus.RUNNING
        self.iteration = 0
        self.error = None
        self.cancel_token = CancellationToken()
        self._done_event.clear()

    def next_iteration(self) -> None:
        self.iteration += 1

    def complete(self) -> None:
        self.status = RunStatus.COMPLETED
        self._done_event.set()

    def fail(self, error: str) -> None:
        self.status = RunStatus.ERROR
        self.error = error
        self._done_event.set()

    def cancel(self) -> None:
        if not self.is_running:
            self._done_event.set()
            return
        self.status = RunStatus.CANCELLED
        self.cancel_token.cancel("user_cancelled")

    def mark_done(self) -> None:
        """通知等待者善后已完成（取消场景用）"""
        self._done_event.set()

    def wait_done(self, timeout: float | None = None) -> bool:
        return self._done_event.wait(timeout)

    def wait_or_cancelled(self, timeout: float) -> bool:
        """等待指定时间，如果期间被取消返回 True"""
        return self.cancel_token.wait(timeout)

    def check_cancel(self) -> None:
        """检查取消信号，已取消则抛 CancelledError"""
        try:
            self.cancel_token.raise_if_cancelled()
        except ToolCancelledError as exc:
            raise CancelledError(str(exc)) from exc
