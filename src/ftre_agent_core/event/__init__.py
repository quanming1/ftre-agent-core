# -*- coding: utf-8 -*-
"""Agent 事件包（纯 AgentScope 对齐协议）。

旧 ftre 事件已删除，不再兼容。所有事件继承 EventBase（pydantic，model_dump 扁平）。
"""
from ._event import (
    EventType,
    EventBase,
    ReplyStartEvent,
    ReplyEndEvent,
    ExceedMaxItersEvent,
    ModelCallStartEvent,
    ModelCallEndEvent,
    TextBlockStartEvent,
    TextBlockDeltaEvent,
    TextBlockEndEvent,
    DataBlockStartEvent,
    DataBlockDeltaEvent,
    DataBlockEndEvent,
    ThinkingBlockStartEvent,
    ThinkingBlockDeltaEvent,
    ThinkingBlockEndEvent,
    HintBlockEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolResultStartEvent,
    ToolResultTextDeltaEvent,
    ToolResultDataDeltaEvent,
    ToolResultEndEvent,
    RequireUserConfirmEvent,
    RetryEvent,
    CustomEvent,
    UserMessageEvent,
    AgentStreamEvent,
)
from ..types import ReplyFinishedReason

__all__ = [
    "EventType",
    "EventBase",
    "ReplyFinishedReason",
    "ReplyStartEvent", "ReplyEndEvent", "ExceedMaxItersEvent",
    "ModelCallStartEvent", "ModelCallEndEvent",
    "TextBlockStartEvent", "TextBlockDeltaEvent", "TextBlockEndEvent",
    "DataBlockStartEvent", "DataBlockDeltaEvent", "DataBlockEndEvent",
    "ThinkingBlockStartEvent", "ThinkingBlockDeltaEvent", "ThinkingBlockEndEvent",
    "HintBlockEvent",
    "ToolCallStartEvent", "ToolCallDeltaEvent", "ToolCallEndEvent",
    "ToolResultStartEvent", "ToolResultTextDeltaEvent",
    "ToolResultDataDeltaEvent", "ToolResultEndEvent",
    "RequireUserConfirmEvent",
    "RetryEvent", "CustomEvent", "UserMessageEvent", "AgentStreamEvent",
]
