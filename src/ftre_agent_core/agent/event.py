"""
Agent 事件定义。

事件统一结构：{"type": EventType, "event_id": str, "timestamp": float, "turn_id": str, "data": <TypedDict>}

事件现已从裸 dict 迁移为 @dataclass 子类，通过 to_dict() 保持序列化兼容。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypedDict
import time
import uuid


class EventType(str, Enum):
    TOOL_RESULT = "tool_result"
    ASSISTANT_MESSAGE = "assistant_message"
    ASSISTANT_MESSAGE_COMPLETE = "assistant_message_complete"
    STEP = "step"
    RETRY = "retry"
    USER_MESSAGE = "user_message"


class StepPhase(str, Enum):
    """Step 事件的阶段标识。"""
    PIPELINE_START = "pipeline_start"
    PIPELINE_END = "pipeline_end"
    COMPACT_START = "compact_start"
    COMPACT_END = "compact_end"
    COMMAND_MATCHED = "command_matched"
    TURN_START = "turn_start"
    TURN_END = "turn_end"


class DoneReason(str, Enum):
    """StepEvent.turn_end 时的结束原因。"""
    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations"
    ERROR = "error"
    CANCELLED = "cancelled"


# ─── TypedDict（保留，外部可能 import）──────────────────────────

class ToolResultData(TypedDict, total=False):
    id: str
    name: str
    result: str
    error: str | None
    status: str
    error_code: str | None
    metadata: dict[str, Any]


class AssistantMessageData(TypedDict):
    content: list[dict]


class AssistantMessageCompleteData(TypedDict, total=False):
    content: list[dict]
    metadata: dict


class RetryData(TypedDict):
    code: str
    message: str
    attempt: int
    max_attempts: int



# ═══════════════════════════════════════════════════════════════════
# @dataclass 事件类
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AgentEvent:
    """事件基类。

    子类在 __post_init__ 中通过 object.__setattr__ 设置 type 字段，
    这是因为 dataclass 不允许不带默认值的字段出现在有默认值字段之后。

    顶层字段：
        event_id  — 事件唯一标识（自动生成）
        timestamp — 事件创建时间戳（Unix 秒，自动生成）
        turn_id   — 所属 Turn 的标识（空串表示不在 turn 内；由 runner 统一盖戳）
    """
    type: EventType = field(init=False)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16], init=False)
    timestamp: float = field(default_factory=time.time, init=False)
    turn_id: str = field(default="", init=False)

    def to_dict(self) -> dict:
        """序列化为 {"type": "...", "event_id": "...", "timestamp": ..., "turn_id": "...", "data": {...}}。"""
        d: dict[str, Any] = {
            "type": self.type,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "data": self._data_dict(),
        }
        if self.turn_id:
            d["turn_id"] = self.turn_id
        return d

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
        ts = d.get("timestamp")
        if isinstance(ts, (int, float)):
            object.__setattr__(event, "timestamp", float(ts))
        tid = d.get("turn_id")
        if isinstance(tid, str) and tid:
            object.__setattr__(event, "turn_id", tid)
        return event

    def _data_dict(self) -> dict:
        """子类覆盖：返回 data 段内容。"""
        raise NotImplementedError("Subclass must implement _data_dict()")


# ─── 具体事件子类 ────────────────────────────────────────────────

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
    """流式累积事件 —— 每次携带到当前为止的完整 content[]。"""
    content: list[dict]

    def __post_init__(self):
        object.__setattr__(self, 'type', EventType.ASSISTANT_MESSAGE)

    def _data_dict(self) -> dict:
        return {"content": self.content}


@dataclass
class AssistantMessageCompleteEvent(AgentEvent):
    """一轮 LLM 输出的完整消息。

    content 是内容块数组，混合 text / thinking / toolCall，
    对齐 OpenAI Chat Completions API 的 message content 格式。

    metadata 携带 usage、kind、stopReason 等元信息，
    取代旧的 usage_update / reasoning_complete / tool_call 独立事件。
    """
    content: list[dict]
    metadata: dict

    def __post_init__(self):
        object.__setattr__(self, 'type', EventType.ASSISTANT_MESSAGE_COMPLETE)

    def _data_dict(self) -> dict:
        return {"content": self.content, "metadata": self.metadata}



@dataclass
class StepEvent(AgentEvent):
    """Turn 生命周期事件。

    phase=turn_start: Turn 开始
      - start_trigger: 触发来源（"user" / 未来扩展）
    phase=turn_end:   Turn 结束
      - reason="completed"     → 正常完成
      - reason="max_iterations" → 达到迭代上限
      - reason="cancelled"      → 用户取消
      - reason="error"          → LLM 错误（携带 error_message / error_code）
      - token_usage: 本轮累积的 token 用量统计
    """
    phase: StepPhase
    success: bool = True
    reason: str = ""
    iterations: int = 0
    error_message: str = ""
    error_code: str = ""
    start_trigger: str = ""
    token_usage: dict[str, Any] | None = None
    turn_type: str = ""
    duration_ms: int | None = None
    compact_triggered: bool = False
    command_name: str | None = None

    def __post_init__(self):
        object.__setattr__(self, 'type', EventType.STEP)

    @property
    def is_turn_end(self) -> bool:
        return self.phase == StepPhase.TURN_END

    @property
    def is_error(self) -> bool:
        return self.reason == DoneReason.ERROR

    def _data_dict(self) -> dict:
        d: dict[str, Any] = {
            "phase": self.phase,
            "success": self.success,
            "reason": self.reason,
            "iterations": self.iterations,
        }
        if self.error_message:
            d["error_message"] = self.error_message
        if self.error_code:
            d["error_code"] = self.error_code
        if self.start_trigger:
            d["start_trigger"] = self.start_trigger
        if self.token_usage:
            d["token_usage"] = self.token_usage
        if self.turn_type:
            d["turn_type"] = self.turn_type
        if self.duration_ms is not None:
            d["duration_ms"] = self.duration_ms
        if self.compact_triggered:
            d["compact_triggered"] = self.compact_triggered
        if self.command_name:
            d["command_name"] = self.command_name
        return d


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
    if t == EventType.TOOL_RESULT:
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
        return AssistantMessageEvent(content=data.get("content", []))
    elif t == EventType.ASSISTANT_MESSAGE_COMPLETE:
        return AssistantMessageCompleteEvent(
            content=data.get("content", []),
            metadata=data.get("metadata", {}),
        )
    elif t == EventType.STEP:
        return StepEvent(
            phase=StepPhase(data.get("phase", "turn_end")),
            success=data.get("success", True),
            reason=data.get("reason", ""),
            iterations=data.get("iterations", 0),
            error_message=data.get("error_message", ""),
            error_code=data.get("error_code", ""),
            start_trigger=data.get("start_trigger", ""),
            token_usage=data.get("token_usage"),
        )
    elif t == EventType.RETRY:
        return RetryEvent(
            code=data.get("code", ""),
            message=data.get("message", ""),
            attempt=data.get("attempt", 0),
            max_attempts=data.get("max_attempts", 0),
        )
    elif t == EventType.USER_MESSAGE:
        return UserMessageEvent(
            content=data.get("content", ""),
            metadata=data.get("metadata", {"hide": True}),
        )
    else:
        raise ValueError(f"Unknown event type: {t!r}")


# ─── 事件构造函数（签名不变，返回 dataclass 实例）───────────────

def assistant_message_event(content: list[dict]) -> AgentEvent:
    return AssistantMessageEvent(content=content)


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


def assistant_message_complete_event(
    content: list[dict],
    metadata: dict | None = None,
) -> AgentEvent:
    """构造一轮 LLM 输出的完整消息事件。

    Args:
        content: 内容块数组，如 [{type:"text",text:"..."}, {type:"toolCall",...}]
        metadata: 元信息，如 {kind, usage, stopReason, provider, model, responseId}
    """
    return AssistantMessageCompleteEvent(
        content=content,
        metadata=metadata or {},
    )


def step_event(
    phase: StepPhase,
    *,
    success: bool = True,
    reason: str = "",
    iterations: int = 0,
    error_message: str = "",
    error_code: str = "",
    start_trigger: str = "",
    token_usage: dict[str, Any] | None = None,
) -> AgentEvent:
    """构造 StepEvent（统一 Turn 生命周期事件）。"""
    return StepEvent(
        phase=phase,
        success=success,
        reason=reason,
        iterations=iterations,
        error_message=error_message,
        error_code=error_code,
        start_trigger=start_trigger,
        token_usage=token_usage,
    )


def user_message_event(
    content: str | list[dict], metadata: dict[str, Any] | None = None
) -> UserMessageEvent:
    """构造 UserMessageEvent。metadata.hide=True 表示前端不渲染。"""
    return UserMessageEvent(content=content, metadata=metadata or {"hide": True})


def retry_event(code: str, message: str, attempt: int, max_attempts: int) -> AgentEvent:
    return RetryEvent(code=code, message=message, attempt=attempt, max_attempts=max_attempts)
