from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Protocol

class ManagedResource(Protocol):
    resource_id: str
    kind: str

    def cancel(self, reason: str) -> None: ...
    def cleanup(self) -> None: ...
    def snapshot(self) -> dict[str, Any]: ...

@dataclass
class ProcessResource:
    resource_id: str
    proc: Any
    terminate_fn: Any
    kind: str = "process"

    def cancel(self, reason: str) -> None:
        self.terminate_fn(self.proc)

    def cleanup(self) -> None:
        try:
            if self.proc.poll() is None:
                self.terminate_fn(self.proc)
        except Exception:
            pass

    def snapshot(self) -> dict[str, Any]:
        pid = getattr(self.proc, "pid", None)
        return {"resource_id": self.resource_id, "kind": self.kind, "pid": pid}

class ResourceRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._resources: dict[str, ManagedResource] = {}

    def register(self, resource: ManagedResource) -> str:
        with self._lock:
            self._resources[resource.resource_id] = resource
        return resource.resource_id

    def unregister(self, resource_id: str) -> None:
        with self._lock:
            self._resources.pop(resource_id, None)

    def cancel_all(self, reason: str) -> None:
        with self._lock:
            resources = list(self._resources.values())
        for resource in resources:
            try:
                resource.cancel(reason)
            except Exception:
                continue

    def cleanup_all(self) -> None:
        with self._lock:
            resources = list(self._resources.values())
            self._resources.clear()
        for resource in resources:
            try:
                resource.cleanup()
            except Exception:
                continue

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            resources = list(self._resources.values())
        return [resource.snapshot() for resource in resources]
