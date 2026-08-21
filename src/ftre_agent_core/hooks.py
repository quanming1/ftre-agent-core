"""宿主注入的类型化 Hook 协议。

Core 是无状态算法库：它不注册监听器，也不持有进程级 Hook 表。宿主（ftre）通过
``HookDispatcher`` 注入自己的生命周期和作用域实现；本模块只定义双方共享的静态
Spec、payload 和结果类型。
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal, Protocol


class HookMode(StrEnum):
    EMIT = "emit"
    PARALLEL = "parallel"
    SERIAL = "serial"
    BAIL = "bail"
    WATERFALL = "waterfall"


class HookFailurePolicy(StrEnum):
    OBSERVE = "observe"
    PROPAGATE = "propagate"


class HookScope(StrEnum):
    GLOBAL = "global"
    AGENT = "agent"


@dataclass(frozen=True, slots=True)
class HookSpec:
    """一个公开 Hook 的静态契约，不持有运行时监听器。"""

    name: str
    domain: str
    mode: HookMode
    failure_policy: HookFailurePolicy = HookFailurePolicy.PROPAGATE
    payload_type: type | tuple[type, ...] | None = None
    result_type: type | tuple[type, ...] | None = None
    default: Callable[..., Any] | None = None
    scope: HookScope = HookScope.GLOBAL

    def __post_init__(self) -> None:
        if not isinstance(self.mode, HookMode):
            raise TypeError("HookSpec.mode must be a HookMode")
        if not isinstance(self.failure_policy, HookFailurePolicy):
            raise TypeError("HookSpec.failure_policy must be a HookFailurePolicy")
        if not isinstance(self.scope, HookScope):
            raise TypeError("HookSpec.scope must be a HookScope")
        if not self.name or "/" not in self.name:
            raise ValueError("HookSpec.name must be a stable domain/name pair")
        if not self.domain or "/" in self.domain:
            raise ValueError("HookSpec.domain must be a simple domain name")
        if self.name.split("/", 1)[0] != self.domain:
            raise ValueError("HookSpec.name domain must match HookSpec.domain")
        if self.mode is HookMode.WATERFALL and self.default is None:
            raise ValueError("waterfall HookSpec requires an explicit default")

    def validate_payload(self, payload: Any) -> None:
        if self.payload_type is not None and not isinstance(payload, self.payload_type):
            raise TypeError(
                f"{self.name} payload must be {self.payload_type!r}, "
                f"got {type(payload).__name__}"
            )

    def validate_result(self, result: Any) -> None:
        if self.result_type is not None and not isinstance(result, self.result_type):
            raise TypeError(
                f"{self.name} result must be {self.result_type!r}, "
                f"got {type(result).__name__}"
            )


class HookDispatcher(Protocol):
    """宿主提供的异步调度端口。``context`` 是 opaque scope carrier。"""

    async def dispatch(
        self,
        spec: HookSpec,
        payload: Any,
        *,
        context: object | None = None,
    ) -> Any: ...


def _readonly_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


# ── Tool Hook contracts ────────────────────────────────────────────────

TOOLS_PRE_EXECUTE = "tools/pre-execute"
TOOLS_EXECUTE = "tools/execute"
TOOLS_POST_EXECUTE = "tools/post-execute"
TOOLS_RESULT = "tools/result"


@dataclass(frozen=True, slots=True)
class ToolCallIdentity:
    call_id: str
    name: str
    session_id: str = ""
    turn_id: str = ""
    agent_id: str = ""
    iteration: int = 0


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    output: str = ""
    status: Literal["completed", "failed", "cancelled"] = "completed"
    error: str | None = None
    metadata: Mapping[str, Any] = MappingProxyType({})
    value: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _readonly_mapping(self.metadata))
        if self.status not in {"completed", "failed", "cancelled"}:
            raise ValueError("ToolExecutionResult.status is invalid")


@dataclass(frozen=True, slots=True)
class ToolPreExecutePayload:
    call: ToolCallIdentity
    arguments: Mapping[str, Any]
    cancellation: asyncio.Event

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", _readonly_mapping(self.arguments))


@dataclass(frozen=True, slots=True)
class ToolAllow:
    pass


@dataclass(frozen=True, slots=True)
class ToolDeny:
    reason: str


@dataclass(frozen=True, slots=True)
class ToolArguments:
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", _readonly_mapping(self.arguments))


@dataclass(frozen=True, slots=True)
class ToolExecutePayload:
    call: ToolCallIdentity
    arguments: Mapping[str, Any]
    cancellation: asyncio.Event
    invoke: Callable[[], Awaitable[ToolExecutionResult]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", _readonly_mapping(self.arguments))


@dataclass(frozen=True, slots=True)
class ToolPostExecutePayload:
    call: ToolCallIdentity
    arguments: Mapping[str, Any]
    result: ToolExecutionResult
    cancellation: asyncio.Event

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", _readonly_mapping(self.arguments))


@dataclass(frozen=True, slots=True)
class ToolResultPayload:
    call: ToolCallIdentity
    arguments: Mapping[str, Any]
    result: ToolExecutionResult

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", _readonly_mapping(self.arguments))


async def _allow(_payload: ToolPreExecutePayload) -> ToolAllow:
    return ToolAllow()


async def _execute(payload: ToolExecutePayload) -> ToolExecutionResult:
    return await payload.invoke()


async def _accept(payload: ToolPostExecutePayload) -> ToolExecutionResult:
    return payload.result


def _observe(_payload: ToolResultPayload) -> None:
    return None


TOOLS_PRE_EXECUTE_SPEC = HookSpec(
    TOOLS_PRE_EXECUTE,
    "tools",
    HookMode.WATERFALL,
    payload_type=ToolPreExecutePayload,
    result_type=(ToolAllow, ToolDeny, ToolArguments),
    default=_allow,
    scope=HookScope.AGENT,
)
TOOLS_EXECUTE_SPEC = HookSpec(
    TOOLS_EXECUTE,
    "tools",
    HookMode.WATERFALL,
    payload_type=ToolExecutePayload,
    result_type=ToolExecutionResult,
    default=_execute,
    scope=HookScope.AGENT,
)
TOOLS_POST_EXECUTE_SPEC = HookSpec(
    TOOLS_POST_EXECUTE,
    "tools",
    HookMode.WATERFALL,
    payload_type=ToolPostExecutePayload,
    result_type=ToolExecutionResult,
    default=_accept,
    scope=HookScope.AGENT,
)
TOOLS_RESULT_SPEC = HookSpec(
    TOOLS_RESULT,
    "tools",
    HookMode.EMIT,
    failure_policy=HookFailurePolicy.OBSERVE,
    payload_type=ToolResultPayload,
    result_type=type(None),
    default=_observe,
    scope=HookScope.AGENT,
)


# ── LLM stream contract ────────────────────────────────────────────────

LLM_STREAM = "llm/stream"


def _readonly_sequence(value) -> tuple[Mapping[str, Any], ...]:
    return tuple(MappingProxyType(dict(item)) for item in (value or ()))


@dataclass(frozen=True, slots=True)
class LLMStreamPayload:
    agent_id: str
    session_id: str
    turn_id: str
    model: str
    messages: tuple[Mapping[str, Any], ...]
    tools: tuple[Mapping[str, Any], ...]
    cancellation: asyncio.Event
    invoke: Callable[[], AsyncIterator[Any]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", _readonly_sequence(self.messages))
        object.__setattr__(self, "tools", _readonly_sequence(self.tools))


async def _stream(payload: LLMStreamPayload) -> AsyncIterator[Any]:
    return payload.invoke()


LLM_STREAM_SPEC = HookSpec(
    LLM_STREAM,
    "llm",
    HookMode.WATERFALL,
    payload_type=LLMStreamPayload,
    result_type=AsyncIterator,
    default=_stream,
    scope=HookScope.AGENT,
)


# ── Agent turn-stopping contract ───────────────────────────────────────

AGENT_TURN_STOPPING = "agent/turn-stopping"


@dataclass(frozen=True, slots=True)
class TurnStoppingPayload:
    agent: object
    session_id: str
    turn_id: str
    status: str
    request_id: str
    cancellation: asyncio.Event
    last_assistant_text: str = ""
    finish_reason: str = ""
    iteration: int = 0
    continuation_count: int = 0
    max_continuations: int = 3


@dataclass(frozen=True, slots=True)
class StopTurn:
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ContinueTurn:
    prompt: str
    reason: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("ContinueTurn.prompt must be non-empty")


async def _stop_turn(_payload: TurnStoppingPayload) -> StopTurn:
    return StopTurn()


AGENT_TURN_STOPPING_SPEC = HookSpec(
    AGENT_TURN_STOPPING,
    "agent",
    HookMode.WATERFALL,
    payload_type=TurnStoppingPayload,
    result_type=(StopTurn, ContinueTurn),
    default=_stop_turn,
    scope=HookScope.AGENT,
)


__all__ = [
    "AGENT_TURN_STOPPING",
    "AGENT_TURN_STOPPING_SPEC",
    "LLM_STREAM",
    "LLM_STREAM_SPEC",
    "TOOLS_EXECUTE_SPEC",
    "TOOLS_POST_EXECUTE_SPEC",
    "TOOLS_PRE_EXECUTE_SPEC",
    "TOOLS_RESULT_SPEC",
    "ContinueTurn",
    "HookDispatcher",
    "HookFailurePolicy",
    "HookMode",
    "HookScope",
    "HookSpec",
    "LLMStreamPayload",
    "StopTurn",
    "ToolAllow",
    "ToolArguments",
    "ToolCallIdentity",
    "ToolDeny",
    "ToolExecutePayload",
    "ToolExecutionResult",
    "ToolPostExecutePayload",
    "ToolPreExecutePayload",
    "ToolResultPayload",
    "TurnStoppingPayload",
]
