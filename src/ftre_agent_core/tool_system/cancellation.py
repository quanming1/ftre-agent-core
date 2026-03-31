from __future__ import annotations

import threading
from typing import Callable


class ToolCancelledError(Exception):
    def __init__(self, message: str = "Tool execution cancelled"):
        super().__init__(message)


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: dict[int, Callable[[str], None]] = {}
        self._next_callback_id = 0
        self._reason = "cancelled"

    def cancel(self, reason: str = "cancelled") -> None:
        callbacks: list[Callable[[str], None]] = []
        with self._lock:
            if self._event.is_set():
                return
            self._reason = reason
            self._event.set()
            callbacks = list(self._callbacks.values())
        for callback in callbacks:
            try:
                callback(reason)
            except Exception:
                continue

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise ToolCancelledError(self._reason)

    def on_cancel(self, callback: Callable[[str], None]) -> Callable[[], None]:
        with self._lock:
            if self._event.is_set():
                reason = self._reason
            else:
                callback_id = self._next_callback_id
                self._next_callback_id += 1
                self._callbacks[callback_id] = callback

                def unregister() -> None:
                    with self._lock:
                        self._callbacks.pop(callback_id, None)

                return unregister
        callback(reason)
        return lambda: None

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)
