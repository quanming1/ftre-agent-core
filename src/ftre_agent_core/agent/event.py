"""
Agent 事件定义。

事件统一结构：{"type": EventType, "data": <TypedDict>}

事件现已从裸 dict 迁移为 @dataclass 子类，通过 to_dict() 保持序列化兼容。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypedDict
import uuid


class EventType(str, Enum):
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ASSISTANT_MESSAGE = "assistant_message"
    ASSISTANT_MESSAGE_COMPLETE = "assistant_message_complete"
    REASONING = "reasoning"
    REASONING_COMPLETE = "reasoning_complete"
    ERROR = "error"
    RETRY = "retry"
    DONE = "done"
    TOOL_CALL_STREAMING = "tool_call_streaming"
    USAGE_UPDATE = "usage_update"
    USER_MESSAGE = "user_message"


class DoneReason(str, Enum):
    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations"
    ERROR = "error"
    CANCELLED = "cancelled"


# ─── TypedDict（保留，外部可能 import）──────────────────────────

class ToolCallData(TypedDict):
    id: str
    name: str
    arguments: dict[str, Any]


class ToolResultData(TypedDict, total=False):
    id: str
    name: str
    result: str
    error: str | None
    status: str
    error_code: str | None
    metadata: dict[str, Any]


class AssistantMessageData(TypedDict):
    content: str


class AssistantMessageCompleteData(TypedDict):
    content: str


class DoneData(TypedDict, total=False):
    success: bool
    reason: DoneReason
    usage: dict


class UsageUpdateData(TypedDict):
    usage: dict


class ErrorData(TypedDict):
    message: str
    code: str


class RetryData(TypedDict):
    code: str
    message: str
    attempt: int
    max_attempts: int


class ToolCallStreamingData(TypedDict):
    tool_calls: list[dict]


# ─── 向后兼容别名 ───────────────────────────────────────────────

AgentEventDict = dict


# ═══════════════════════════════════════════════════════════════════
# @dataclass 事件类
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AgentEvent:
    """事件基类。

    子类在 __post_init__ 中通过 object.__setattr__ 设置 type 字段，
    这是因为 dataclass 不允许不带默认值的字段出现在有默认值字段之后。
    """
    type: EventType = field(init=False)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16], init=False)

    def to_dict(self) -> dict:
        """序列化为 {"type": "...", "data": {...}}，与旧格式 100% 兼容。"""
        return {"type": self.type, "event_id": self.event_id, "data": self._data_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> AgentEvent:
        """从 {"type": "...", "data": {...}} 反序列化。"""
        t = d["type"]
        data = d.get("data", {})
        event = _from_type(t, data)
        event_id = d.get("event_id")
        if not event_id and isinstance(data, dict):
            event_id = data.get("event_id")
        if isinstance(event_id, str) and event_id:
            object.__setattr__(event, "event_id", event_id)
        return event

    def _data_dict(self) -> dict:
        """子类覆盖：返回 data 段内容。"""
        raise NotImplementedError("Subclass must implement _data_dict()")


# ─── 具体事件子类 ────────────────────────────────────────────────

@dataclass
class ToolCallEvent(AgentEvent):
    tool_id: str
    tool_name: str
    arguments: dict[str, Any]

    def __post_init__(self):
        object.__setattr__(self, 'type', EventType.TOOL_CALL)

    def _data_dict(self) -> dict:
        return {"id": self.tool_id, "name": self.tool_name, "arguments": self.arguments}


@dataclass
class ToolResultEvent(AgentEvent):
    tool_id: str
    tool_name: str
    result: str
    error: str | None = None
    status: str = "completed"
    error_code: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        object.__setattr__(self, 'type', EventType.TOOL_RESULT)

    def _data_dict(self) -> dict:
        d: dict[str, Any] = {
            "id": self.tool_id,
            "name": self.tool_name,
            "result": self.result,
            "error": self.error,
            "status": self.status,
            "error_code": self.error_code,
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class AssistantMessageEvent(AgentEvent):
    content: str

    def __post_init__(self):
        object.__setattr__(self, 'type', EventType.ASSISTANT_MESSAGE)

    def _data_dict(self) -> dict:
        return {"content": self.content}


@dataclass
class AssistantMessageCompleteEvent(AgentEvent):
    content: str

    def __post_init__(self):
        object.__setattr__(self, 'type', EventType.ASSISTANT_MESSAGE_COMPLETE)

    def _data_dict(self) -> dict:
        return {"content": self.content}


@dataclass
class ReasoningEvent(AgentEvent):
    content: str

    def __post_init__(self):
        object.__setattr__(self, 'type', EventType.REASONING)

    def _data_dict(self) -> dict:
        return {"content": self.content}


@dataclass
class ReasoningCompleteEvent(AgentEvent):
    content: str

    def __post_init__(self):
        object.__setattr__(self, 'type', EventType.REASONING_COMPLETE)

    def _data_dict(self) -> dict:
        return {"content": self.content}


@dataclass
class DoneEvent(AgentEvent):
    success: bool
    reason: DoneReason
    usage: dict | None = None

    def __post_init__(self):
        object.__setattr__(self, 'type', EventType.DONE)

    def _data_dict(self) -> dict:
        d: dict[str, Any] = {"success": self.success, "reason": self.reason}
        if self.usage:
            d["usage"] = self.usage
        return d


@dataclass
class ErrorEvent(AgentEvent):
    message: str
    code: str = "unknown"

    def __post_init__(self):
        object.__setattr__(self, 'type', EventType.ERROR)

    def _data_dict(self) -> dict:
        return {"message": self.message, "code": self.code}


@dataclass
class RetryEvent(AgentEvent):
    code: str
    message: str
    attempt: int
    max_attempts: int

    def __post_init__(self):
        object.__setattr__(self, 'type', EventType.RETRY)

    def _data_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "attempt": self.attempt, "max_attempts": self.max_attempts}


@dataclass
class ToolCallStreamingEvent(AgentEvent):
    tool_calls: list[dict]

    def __post_init__(self):
        object.__setattr__(self, 'type', EventType.TOOL_CALL_STREAMING)

    def _data_dict(self) -> dict:
        return {"tool_calls": self.tool_calls}


@dataclass
class UsageUpdateEvent(AgentEvent):
    usage: dict

    def __post_init__(self):
        object.__setattr__(self, 'type', EventType.USAGE_UPDATE)

    def _data_dict(self) -> dict:
        return {"usage": self.usage}


@dataclass
class UserMessageEvent(AgentEvent):
    """工具注入的 user message：LLM 可见，前端隐藏。

    Fields:
        content: str（文字）或 list[dict]（多模态 image_url）
        metadata: dict，默认 {"hide": True}
    """
    content: str | list[dict]
    metadata: dict[str, Any] = field(default_factory=lambda: {"hide": True})

    def __post_init__(self):
        object.__setattr__(self, 'type', EventType.USER_MESSAGE)

    def to_openai_message(self) -> dict:
        """转为 OpenAI 格式 user message，可直接追加到 memory。

        content 中的 image_file part 会被转换为 image_url（读文件转 base64），
        以兼容 OpenAI 多模态格式。其他 part 类型原样保留。
        """
        content = self.content
        if isinstance(content, list):
            content = [_convert_image_file_part(p) for p in content]
        return {"role": "user", "content": content}

    def _data_dict(self) -> dict:
        return {"content": self.content, "metadata": self.metadata}


def _convert_image_file_part(part: dict) -> dict:
    """将 image_file part 转换为 image_url（读文件转 base64 data URL）。

    agent-core 不能依赖 ftre 的 image_store，因此用标准库 base64 直接读取。
    文件不存在或读取失败时降级为文本提示，不抛异常。
    """
    if not isinstance(part, dict):
        return part
    if part.get("type") != "image_file":
        return part

    path = part.get("path", "")
    mime = part.get("mime_type", "image/png")
    if not path:
        return {"type": "text", "text": "[图片加载失败: 无文件路径]"}

    try:
        import base64 as _b64
        with open(path, "rb") as f:
            raw = f.read()
        b64 = _b64.b64encode(raw).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        }
    except Exception as e:
        return {"type": "text", "text": f"[图片加载失败: {path} ({e})]"}


# ─── _from_type 分发工厂 ─────────────────────────────────────────

def _from_type(t: str, data: dict) -> AgentEvent:
    """根据 type 字符串分派到对应 dataclass 子类。"""
    if t == EventType.TOOL_CALL:
        return ToolCallEvent(
            tool_id=data["id"],
            tool_name=data["name"],
            arguments=data["arguments"],
        )
    elif t == EventType.TOOL_RESULT:
        return ToolResultEvent(
            tool_id=data.get("id", ""),
            tool_name=data.get("name", ""),
            result=data.get("result", ""),
            error=data.get("error"),
            status=data.get("status", "completed"),
            error_code=data.get("error_code"),
            metadata=data.get("metadata"),
        )
    elif t == EventType.ASSISTANT_MESSAGE:
        return AssistantMessageEvent(content=data.get("content", ""))
    elif t == EventType.ASSISTANT_MESSAGE_COMPLETE:
        return AssistantMessageCompleteEvent(content=data.get("content", ""))
    elif t == EventType.REASONING:
        return ReasoningEvent(content=data.get("content", ""))
    elif t == EventType.REASONING_COMPLETE:
        return ReasoningCompleteEvent(content=data.get("content", ""))
    elif t == EventType.DONE:
        return DoneEvent(
            success=data.get("success", False),
            reason=data.get("reason", DoneReason.COMPLETED),
            usage=data.get("usage"),
        )
    elif t == EventType.ERROR:
        return ErrorEvent(
            message=data.get("message", ""),
            code=data.get("code", "unknown"),
        )
    elif t == EventType.RETRY:
        return RetryEvent(
            code=data.get("code", ""),
            message=data.get("message", ""),
            attempt=data.get("attempt", 0),
            max_attempts=data.get("max_attempts", 0),
        )
    elif t == EventType.TOOL_CALL_STREAMING:
        return ToolCallStreamingEvent(tool_calls=data.get("tool_calls", []))
    elif t == EventType.USAGE_UPDATE:
        return UsageUpdateEvent(usage=data.get("usage", {}))
    elif t == EventType.USER_MESSAGE:
        return UserMessageEvent(
            content=data.get("content", ""),
            metadata=data.get("metadata", {"hide": True}),
        )
    else:
        raise ValueError(f"Unknown event type: {t!r}")


# ─── 事件构造函数（签名不变，返回 dataclass 实例）───────────────

def tool_call_streaming_event(chunks: list) -> AgentEvent:
    """从 ToolCallDeltaChunk 列表或 dict 构造流式工具调用事件。"""
    result = []
    for c in chunks:
        if isinstance(c, dict):
            entry = {k: v for k, v in c.items() if v is not None}
        else:
            entry = {k: v for k, v in {
                "index": c.index,
                "id": c.id,
                "name": c.name,
                "arguments_delta": c.arguments_delta,
            }.items() if v is not None}
        result.append(entry)
    return ToolCallStreamingEvent(tool_calls=result)


def tool_call_event(id: str, name: str, arguments: dict[str, Any]) -> AgentEvent:
    return ToolCallEvent(tool_id=id, tool_name=name, arguments=arguments)


def tool_result_event(
    id: str,
    name: str,
    result: str,
    error: str | None = None,
    *,
    status: str = "completed",
    error_code: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentEvent:
    return ToolResultEvent(
        tool_id=id,
        tool_name=name,
        result=result,
        error=error,
        status=status,
        error_code=error_code,
        metadata=metadata,
    )


def assistant_message_event(content: str) -> AgentEvent:
    return AssistantMessageEvent(content=content)


def reasoning_event(content: str) -> AgentEvent:
    return ReasoningEvent(content=content)


def reasoning_complete_event(content: str) -> AgentEvent:
    """一轮 LLM reasoning 的完整文本，用于持久化和多轮回放。"""
    return ReasoningCompleteEvent(content=content)


def assistant_message_complete_event(content: str) -> AgentEvent:
    return AssistantMessageCompleteEvent(content=content)


def done_event(success: bool, reason: DoneReason, usage: dict | None = None) -> AgentEvent:
    return DoneEvent(success=success, reason=reason, usage=usage)


def usage_update_event(usage: dict) -> AgentEvent:
    return UsageUpdateEvent(usage=usage)


def user_message_event(
    content: str | list[dict], metadata: dict[str, Any] | None = None
) -> UserMessageEvent:
    """构造 UserMessageEvent。metadata.hide=True 表示前端不渲染。"""
    return UserMessageEvent(content=content, metadata=metadata or {"hide": True})


def error_event(message: str, code: str = "unknown") -> AgentEvent:
    return ErrorEvent(message=message, code=code)


def retry_event(code: str, message: str, attempt: int, max_attempts: int) -> AgentEvent:
    return RetryEvent(code=code, message=message, attempt=attempt, max_attempts=max_attempts)
