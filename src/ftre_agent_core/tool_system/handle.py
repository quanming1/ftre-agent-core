from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading

from .cancellation import CancellationToken
from .result import ToolResult

class ToolExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    FAILED = "failed"
    COMPLETED = "completed"

_FINAL_STATUSES = {
    ToolExecutionStatus.CANCELLED,
    ToolExecutionStatus.TIMED_OUT,
    ToolExecutionStatus.FAILED,
    ToolExecutionStatus.COMPLETED,
}

@dataclass
class ToolExecutionHandle:
    call_id: str
    name: str
    status: ToolExecutionStatus = ToolExecutionStatus.PENDING
    cancel_token: CancellationToken = field(default_factory=CancellationToken)
    result: ToolResult | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def transition_to(self, next_status: ToolExecutionStatus) -> None:
        with self._lock:
            if self.status in _FINAL_STATUSES:
                return
            self.status = next_status
            if next_status == ToolExecutionStatus.RUNNING and self.started_at is None:
                self.started_at = datetime.utcnow()
            if next_status in _FINAL_STATUSES and self.finished_at is None:
                self.finished_at = datetime.utcnow()

    def request_cancel(self, reason: str = "cancelled") -> None:
        with self._lock:
            if self.status in _FINAL_STATUSES:
                return
            self.status = ToolExecutionStatus.CANCELLING
        self.cancel_token.cancel(reason)

    def finish(self, result: ToolResult) -> ToolResult:
        with self._lock:
            if self.result is not None:
                return self.result
            self.result = result
            mapping = {
                "completed": ToolExecutionStatus.COMPLETED,
                "cancelled": ToolExecutionStatus.CANCELLED,
                "timed_out": ToolExecutionStatus.TIMED_OUT,
                "failed": ToolExecutionStatus.FAILED,
            }
            self.status = mapping[result.status]
            if self.started_at is None:
                self.started_at = datetime.utcnow()
            if self.finished_at is None:
                self.finished_at = datetime.utcnow()
        return result
