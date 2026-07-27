# -*- coding: utf-8 -*-
"""Agent 事件定义（破坏性重构后，纯 AgentScope 对齐协议）。

旧 ftre 事件（AgentEvent 基类 + 6 子类 + 工厂 + DoneReason/StepPhase +
TypedDict）已删除，不再兼容。所有事件继承 EventBase（pydantic，model_dump
扁平序列化）。RetryEvent 为 ftre 特有，改继承 EventBase。
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, List, TypeAlias, Union
import uuid

from pydantic import BaseModel, ConfigDict, Field

from ..types import ReplyFinishedReason
from ..message import (
    DataBlock,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    ToolResultState,
)


def _gen_id() -> str:
    return uuid.uuid4().hex[:16]


def _now_iso() -> str:
    return datetime.now().isoformat()


class EventType(StrEnum):
    """事件类型枚举（AgentScope 对齐）。"""
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
    RETRY = "retry"
    CUSTOM = "CUSTOM"


# ══════════════════════════════════════════════════════════════════
# EventBase
# ══════════════════════════════════════════════════════════════════

class EventBase(BaseModel):
    """事件基类（pydantic，model_dump 扁平序列化）。"""
    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(default_factory=_gen_id)
    created_at: str = Field(default_factory=_now_iso)
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
    error: dict[str, Any] | None = None


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
    finished_reason: str = "completed"


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


# ── RetryEvent（ftre 特有，继承 EventBase）──
class RetryEvent(EventBase):
    """重试事件。ftre 特有（AgentScope 无），对齐 EventBase 风格。"""
    type: Literal["retry"] = "retry"
    reply_id: str
    code: str
    message: str
    attempt: int
    max_attempts: int


# ── 自定义扩展 ──
class CustomEvent(EventBase):
    type: Literal["CUSTOM"] = "CUSTOM"
    name: str
    value: dict = Field(default_factory=dict)


# ── 联合类型 ──
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
    RetryEvent, CustomEvent,
]
