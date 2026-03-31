from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

@dataclass
class ToolError:
    code: str
    message: str
    retryable: bool = False
    category: str = "tool"
    details: dict[str, Any] | None = None

@dataclass
class ToolOutput:
    preview: str = ""
    channels: dict[str, str] = field(default_factory=dict)
    truncated: bool = False
    full_ref: str | None = None

@dataclass
class ToolResult:
    status: Literal["completed", "cancelled", "timed_out", "failed"]
    value: Any | None
    error: ToolError | None
    output: ToolOutput
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def completed(
        cls,
        value: Any,
        output: ToolOutput,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            status="completed",
            value=value,
            error=None,
            output=output,
            metadata=metadata or {},
        )

    @classmethod
    def cancelled(cls, message: str = "Tool execution cancelled") -> "ToolResult":
        return cls(
            status="cancelled",
            value=None,
            error=ToolError(code="cancelled", message=message, category="cancelled"),
            output=ToolOutput(preview=""),
            metadata={},
        )

    @classmethod
    def timed_out(cls, message: str = "Tool execution timed out") -> "ToolResult":
        return cls(
            status="timed_out",
            value=None,
            error=ToolError(code="timed_out", message=message, category="timeout"),
            output=ToolOutput(preview=""),
            metadata={},
        )

    @classmethod
    def failed(
        cls,
        code: str,
        message: str,
        *,
        category: str = "tool",
        details: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            status="failed",
            value=None,
            error=ToolError(code=code, message=message, category=category, details=details),
            output=ToolOutput(preview=""),
            metadata={},
        )
