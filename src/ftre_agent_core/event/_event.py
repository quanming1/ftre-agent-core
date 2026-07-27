"""
Agent 事件定义。

两部分共存:
  A) ftre 原有事件（react_runner 在用，保留过渡）：
     EventType(6 值) + AgentEvent 基类 + 6 事件子类，to_dict 嵌套结构。
  B) AgentScope 对齐事件（层次 C 新协议，供后续迁移）：
     EventType 扩充 28 值 + EventBase 基类 + ~28 事件类，model_dump 扁平。

两套独立基类：AgentEvent（ftre 旧，event_id/timestamp/turn_id + to_dict）、
EventBase（AgentScope 风格，id/created_at/metadata + model_dump）。
命名不冲突：旧事件 ToolResultEvent 等，新事件 ReplyStartEvent 等。
AgentScope 的 union TypeAlias 改名 AgentStreamEvent，避免与 ftre AgentEvent 撞名。
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, TypedDict, List, TypeAlias, Union
import time
import uuid

from pydantic import BaseModel, ConfigDict, Field

from ..message import (
    DataBlock,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultState,
)


def _gen_id() -> str:
    """16 位 hex id（贴合 ftre event_id 风格 + AgentScope _generate_id 用途）。"""
    return uuid.uuid4().hex[:16]


# ══════════════════════════════════════════════════════════════════
# A) ftre 原有枚举
# ══════════════════════════════════════════════════════════════════

class EventType(StrEnum):
    # ── ftre 原有（react_runner 在用）──
    TOOL_RESULT = "tool_result"
    ASSISTANT_MESSAGE = "assistant_message"
    ASSISTANT_MESSAGE_COMPLETE = "assistant_message_complete"
    STEP = "step"
    RETRY = "retry"
    USER_MESSAGE = "user_message"

    # ── AgentScope 对齐（层次 C 新协议）──
    REPLY_START = "REPLY_START"
    REPLY_END = "REPLY_END"
    MODEL_CALL_START = "MODEL_CALL_START"
    MODEL_CALL_END = "MODEL_CALL_END"
    TEXT_BLOCK_START = "TEXT_BLOCK_START"
    TEXT_BLOCK_DELTA = "TEXT_BLOCK_DELTA"
    TEXT_BLOCK_END = "TEXT_BLOCK_END"
    DATA_BLOCK_START = "DATA_BLOCK_START"
    DATA_BLOCK_DELTA = "DATA_BLOCK_DELTA"
    DATA_BLOCK_END = "DATA_BLOCK_END"
    THINKING_BLOCK_START = "THINKING_BLOCK_START"
    THINKING_BLOCK_DELTA = "THINKING_BLOCK_DELTA"
    THINKING_BLOCK_END = "THINKING_BLOCK_END"
    HINT_BLOCK = "HINT_BLOCK"
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_DELTA = "TOOL_CALL_DELTA"
    TOOL_CALL_END = "TOOL_CALL_END"
    TOOL_RESULT_START = "TOOL_RESULT_START"
    TOOL_RESULT_TEXT_DELTA = "TOOL_RESULT_TEXT_DELTA"
    TOOL_RESULT_DATA_DELTA = "TOOL_RESULT_DATA_DELTA"
    TOOL_RESULT_END = "TOOL_RESULT_END"
    EXCEED_MAX_ITERS = "EXCEED_MAX_ITERS"
    REQUIRE_USER_CONFIRM = "REQUIRE_USER_CONFIRM"
    REQUIRE_EXTERNAL_EXECUTION = "REQUIRE_EXTERNAL_EXECUTION"
    USER_CONFIRM_RESULT = "USER_CONFIRM_RESULT"
    USER_INTERRUPT = "USER_INTERRUPT"
    EXTERNAL_EXECUTION_RESULT = "EXTERNAL_EXECUTION_RESULT"
    CUSTOM = "CUSTOM"


class StepPhase(StrEnum):
    """Step 事件的阶段标识。"""
    PIPELINE_START = "pipeline_start"
    PIPELINE_END = "pipeline_end"
    COMPACT_START = "compact_start"
    COMPACT_END = "compact_end"
    COMMAND_MATCHED = "command_matched"
    TURN_START = "turn_start"
    TURN_END = "turn_end"


class DoneReason(StrEnum):
    """StepEvent.turn_end 时的结束原因。"""
    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations"
    ERROR = "error"
    CANCELLED = "cancelled"


# AgentScope 对齐：ReplyEndEvent 的结束原因，定义在 types.py（避免 event↔message 循环 import）
from ..types import ReplyFinishedReason


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


# ══════════════════════════════════════════════════════════════════
# A) ftre 原有事件基类 + 子类（react_runner 在用，保留）
# ══════════════════════════════════════════════════════════════════

class AgentEvent(BaseModel):
    """ftre 旧事件基类（to_dict 嵌套结构，react_runner 在用）。"""
    model_config = ConfigDict(use_enum_values=True)

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    timestamp: float = Field(default_factory=time.time)
    turn_id: str = ""

    def to_dict(self) -> dict:
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
        raise NotImplementedError("Subclass must implement _data_dict()")


class ToolResultEvent(AgentEvent):
    type: Literal["tool_result"] = "tool_result"
    tool_id: str
    tool_name: str
    result: str
    error: str | None = None
    status: str = "completed"
    error_code: str | None = None
    metadata: dict[str, Any] | None = None

    def _data_dict(self) -> dict:
        d: dict[str, Any] = {
            "id": self.tool_id, "name": self.tool_name, "result": self.result,
            "error": self.error, "status": self.status, "error_code": self.error_code,
        }
        if self.metadata:
            d["metadata"] = self.metadata
        return d


class AssistantMessageEvent(AgentEvent):
    type: Literal["assistant_message"] = "assistant_message"
    content: list[dict]

    def _data_dict(self) -> dict:
        return {"content": self.content}


class AssistantMessageCompleteEvent(AgentEvent):
    type: Literal["assistant_message_complete"] = "assistant_message_complete"
    content: list[dict]
    metadata: dict

    def _data_dict(self) -> dict:
        return {"content": self.content, "metadata": self.metadata}


class StepEvent(AgentEvent):
    type: Literal["step"] = "step"
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

    @property
    def is_turn_end(self) -> bool:
        return self.phase == StepPhase.TURN_END

    @property
    def is_error(self) -> bool:
        return self.reason == DoneReason.ERROR

    def _data_dict(self) -> dict:
        d: dict[str, Any] = {
            "phase": self.phase, "success": self.success,
            "reason": self.reason, "iterations": self.iterations,
        }
        if self.error_message: d["error_message"] = self.error_message
        if self.error_code: d["error_code"] = self.error_code
        if self.start_trigger: d["start_trigger"] = self.start_trigger
        if self.token_usage: d["token_usage"] = self.token_usage
        if self.turn_type: d["turn_type"] = self.turn_type
        if self.duration_ms is not None: d["duration_ms"] = self.duration_ms
        if self.compact_triggered: d["compact_triggered"] = self.compact_triggered
        if self.command_name: d["command_name"] = self.command_name
        return d


class RetryEvent(AgentEvent):
    type: Literal["retry"] = "retry"
    code: str
    message: str
    attempt: int
    max_attempts: int

    def _data_dict(self) -> dict:
        return {"code": self.code, "message": self.message,
                "attempt": self.attempt, "max_attempts": self.max_attempts}


class UserMessageEvent(AgentEvent):
    type: Literal["user_message"] = "user_message"
    content: str | list[dict]
    metadata: dict[str, Any] = Field(default_factory=lambda: {"hide": True})

    def to_openai_message(self) -> dict:
        content = self.content
        if isinstance(content, list):
            content = [_convert_image_file_part(p) for p in content]
        return {"role": "user", "content": content}

    def _data_dict(self) -> dict:
        return {"content": self.content, "metadata": self.metadata}


def _convert_image_file_part(part: dict) -> dict:
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
        with open(path, "rb") as f:  # noqa: PTH123
            raw = f.read()
        b64 = _b64.b64encode(raw).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
    except Exception as e:
        return {"type": "text", "text": f"[图片加载失败: {path} ({e})]"}


def _from_type(t: str, data: dict) -> AgentEvent:
    if t == EventType.TOOL_RESULT:
        return ToolResultEvent(tool_id=data.get("id",""), tool_name=data.get("name",""),
            result=data.get("result",""), error=data.get("error"),
            status=data.get("status","completed"), error_code=data.get("error_code"),
            metadata=data.get("metadata"))
    elif t == EventType.ASSISTANT_MESSAGE:
        return AssistantMessageEvent(content=data.get("content", []))
    elif t == EventType.ASSISTANT_MESSAGE_COMPLETE:
        return AssistantMessageCompleteEvent(content=data.get("content", []),
            metadata=data.get("metadata", {}))
    elif t == EventType.STEP:
        return StepEvent(phase=StepPhase(data.get("phase","turn_end")),
            success=data.get("success",True), reason=data.get("reason",""),
            iterations=data.get("iterations",0), error_message=data.get("error_message",""),
            error_code=data.get("error_code",""), start_trigger=data.get("start_trigger",""),
            token_usage=data.get("token_usage"))
    elif t == EventType.RETRY:
        return RetryEvent(code=data.get("code",""), message=data.get("message",""),
            attempt=data.get("attempt",0), max_attempts=data.get("max_attempts",0))
    elif t == EventType.USER_MESSAGE:
        return UserMessageEvent(content=data.get("content",""),
            metadata=data.get("metadata", {"hide": True}))
    else:
        raise ValueError(f"Unknown event type: {t!r}")


def assistant_message_event(content: list[dict]) -> AgentEvent:
    return AssistantMessageEvent(content=content)

def tool_result_event(id, name, result, error=None, *, status="completed",
                      error_code=None, metadata=None) -> AgentEvent:
    return ToolResultEvent(tool_id=id, tool_name=name, result=result, error=error,
        status=status, error_code=error_code, metadata=metadata)

def assistant_message_complete_event(content, metadata=None) -> AgentEvent:
    return AssistantMessageCompleteEvent(content=content, metadata=metadata or {})

def step_event(phase, *, success=True, reason="", iterations=0, error_message="",
               error_code="", start_trigger="", token_usage=None) -> AgentEvent:
    return StepEvent(phase=phase, success=success, reason=reason, iterations=iterations,
        error_message=error_message, error_code=error_code,
        start_trigger=start_trigger, token_usage=token_usage)

def user_message_event(content, metadata=None) -> UserMessageEvent:
    return UserMessageEvent(content=content, metadata=metadata or {"hide": True})

def retry_event(code, message, attempt, max_attempts) -> AgentEvent:
    return RetryEvent(code=code, message=message, attempt=attempt, max_attempts=max_attempts)


# ══════════════════════════════════════════════════════════════════
# B) AgentScope 对齐事件（层次 C 新协议）
# 字段对齐 AgentScope：id/created_at/metadata + reply_id/block_id/tool_call_id
# 序列化用 pydantic model_dump（扁平），不加 to_dict
# ══════════════════════════════════════════════════════════════════

class EventBase(BaseModel):
    """AgentScope 风格事件基类（model_dump 扁平序列化）。"""
    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(default_factory=_gen_id)
    created_at: str = Field(default_factory=lambda: __import__("datetime").datetime.now().isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── 生命周期 ──
class ReplyStartEvent(EventBase):
    type: Literal["REPLY_START"] = "REPLY_START"
    session_id: str
    reply_id: str
    name: str
    role: Literal["user", "assistant", "system"] = "assistant"


class ReplyEndEvent(EventBase):
    type: Literal["REPLY_END"] = "REPLY_END"
    session_id: str
    reply_id: str
    finished_reason: ReplyFinishedReason = ReplyFinishedReason.COMPLETED
    error: dict[str, Any] | None = None  # AgentScope ErrorInfo 占位


class ExceedMaxItersEvent(EventBase):
    type: Literal["EXCEED_MAX_ITERS"] = "EXCEED_MAX_ITERS"
    reply_id: str
    name: str


# ── 模型调用 ──
class ModelCallStartEvent(EventBase):
    type: Literal["MODEL_CALL_START"] = "MODEL_CALL_START"
    reply_id: str
    model_name: str


class ModelCallEndEvent(EventBase):
    type: Literal["MODEL_CALL_END"] = "MODEL_CALL_END"
    reply_id: str
    input_tokens: int
    output_tokens: int
    finished_reason: str = "completed"  # AgentScope FinishedReason 占位


# ── 文本块流式 ──
class TextBlockStartEvent(EventBase):
    type: Literal["TEXT_BLOCK_START"] = "TEXT_BLOCK_START"
    reply_id: str
    block_id: str

class TextBlockDeltaEvent(EventBase):
    type: Literal["TEXT_BLOCK_DELTA"] = "TEXT_BLOCK_DELTA"
    reply_id: str
    block_id: str
    delta: str

class TextBlockEndEvent(EventBase):
    type: Literal["TEXT_BLOCK_END"] = "TEXT_BLOCK_END"
    reply_id: str
    block_id: str


# ── 数据块流式 ──
class DataBlockStartEvent(EventBase):
    type: Literal["DATA_BLOCK_START"] = "DATA_BLOCK_START"
    reply_id: str
    block_id: str
    media_type: str

class DataBlockDeltaEvent(EventBase):
    type: Literal["DATA_BLOCK_DELTA"] = "DATA_BLOCK_DELTA"
    reply_id: str
    block_id: str
    data: str
    media_type: str

class DataBlockEndEvent(EventBase):
    type: Literal["DATA_BLOCK_END"] = "DATA_BLOCK_END"
    reply_id: str
    block_id: str


# ── 思考块流式 ──
class ThinkingBlockStartEvent(EventBase):
    type: Literal["THINKING_BLOCK_START"] = "THINKING_BLOCK_START"
    reply_id: str
    block_id: str

class ThinkingBlockDeltaEvent(EventBase):
    type: Literal["THINKING_BLOCK_DELTA"] = "THINKING_BLOCK_DELTA"
    reply_id: str
    block_id: str
    delta: str

class ThinkingBlockEndEvent(EventBase):
    type: Literal["THINKING_BLOCK_END"] = "THINKING_BLOCK_END"
    reply_id: str
    block_id: str


# ── 提示块（一次性）──
class HintBlockEvent(EventBase):
    type: Literal["HINT_BLOCK"] = "HINT_BLOCK"
    reply_id: str
    block_id: str
    source: str | None = None
    hint: str | List[TextBlock | DataBlock]


# ── 工具调用流式 ──
class ToolCallStartEvent(EventBase):
    type: Literal["TOOL_CALL_START"] = "TOOL_CALL_START"
    reply_id: str
    tool_call_id: str
    tool_call_name: str

class ToolCallDeltaEvent(EventBase):
    type: Literal["TOOL_CALL_DELTA"] = "TOOL_CALL_DELTA"
    reply_id: str
    tool_call_id: str
    delta: str

class ToolCallEndEvent(EventBase):
    type: Literal["TOOL_CALL_END"] = "TOOL_CALL_END"
    reply_id: str
    tool_call_id: str


# ── 工具结果流式 ──
class ToolResultStartEvent(EventBase):
    type: Literal["TOOL_RESULT_START"] = "TOOL_RESULT_START"
    reply_id: str
    tool_call_id: str
    tool_call_name: str

class ToolResultTextDeltaEvent(EventBase):
    type: Literal["TOOL_RESULT_TEXT_DELTA"] = "TOOL_RESULT_TEXT_DELTA"
    reply_id: str
    tool_call_id: str
    delta: str

class ToolResultDataDeltaEvent(EventBase):
    type: Literal["TOOL_RESULT_DATA_DELTA"] = "TOOL_RESULT_DATA_DELTA"
    reply_id: str
    tool_call_id: str
    block_id: str = Field(default_factory=_gen_id)
    media_type: str
    data: str | None = None
    url: str | None = None

class ToolResultEndEvent(EventBase):
    type: Literal["TOOL_RESULT_END"] = "TOOL_RESULT_END"
    reply_id: str
    tool_call_id: str
    state: ToolResultState
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── 人工介入 ──
class RequireUserConfirmEvent(EventBase):
    type: Literal["REQUIRE_USER_CONFIRM"] = "REQUIRE_USER_CONFIRM"
    reply_id: str
    tool_calls: List[ToolCallBlock]

class RequireExternalExecutionEvent(EventBase):
    type: Literal["REQUIRE_EXTERNAL_EXECUTION"] = "REQUIRE_EXTERNAL_EXECUTION"
    reply_id: str
    tool_calls: List[ToolCallBlock]


class ConfirmResult(BaseModel):
    """工具调用确认结果（辅助模型）。PermissionRule 占位 Any。"""
    confirmed: bool
    tool_call: ToolCallBlock
    rules: list[Any] | None = None  # AgentScope PermissionRule 占位


class UserConfirmResultEvent(EventBase):
    type: Literal["USER_CONFIRM_RESULT"] = "USER_CONFIRM_RESULT"
    reply_id: str
    confirm_results: List[ConfirmResult]

class UserInterruptEvent(EventBase):
    type: Literal["USER_INTERRUPT"] = "USER_INTERRUPT"
    reply_id: str

class ExternalExecutionResultEvent(EventBase):
    type: Literal["EXTERNAL_EXECUTION_RESULT"] = "EXTERNAL_EXECUTION_RESULT"
    reply_id: str
    execution_results: List[ToolResultBlock]


# ── 自定义扩展 ──
class CustomEvent(EventBase):
    type: Literal["CUSTOM"] = "CUSTOM"
    name: str
    value: dict = Field(default_factory=dict)


# ── 新事件联合类型（改名避免与 ftre AgentEvent 基类冲突）──
AgentStreamEvent: TypeAlias = Union[
    ReplyStartEvent, ReplyEndEvent, ExceedMaxItersEvent,
    ModelCallStartEvent, ModelCallEndEvent,
    TextBlockStartEvent, TextBlockDeltaEvent, TextBlockEndEvent,
    DataBlockStartEvent, DataBlockDeltaEvent, DataBlockEndEvent,
    ThinkingBlockStartEvent, ThinkingBlockDeltaEvent, ThinkingBlockEndEvent,
    HintBlockEvent,
    ToolCallStartEvent, ToolCallDeltaEvent, ToolCallEndEvent,
    ToolResultStartEvent, ToolResultTextDeltaEvent,
    ToolResultDataDeltaEvent, ToolResultEndEvent,
    RequireUserConfirmEvent, RequireExternalExecutionEvent,
    UserConfirmResultEvent, UserInterruptEvent,
    ExternalExecutionResultEvent, CustomEvent,
]
