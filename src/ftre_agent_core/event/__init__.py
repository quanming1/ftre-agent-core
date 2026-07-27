"""Agent 事件包。

两部分共存:
  A) ftre 原有事件（react_runner 在用）：EventType + AgentEvent + 6 事件 + 工厂函数
  B) AgentScope 对齐事件（层次 C 新协议）：EventBase + ~28 事件 + AgentStreamEvent

新事件用 pydantic model_dump 扁平序列化；旧事件用 to_dict 嵌套。两套独立基类，
命名不冲突。AgentScope 的 union 改名 AgentStreamEvent 避免与 ftre AgentEvent 撞名。
"""
from ._event import (
    # ── 枚举 ──
    EventType,
    StepPhase,
    DoneReason,
    ReplyFinishedReason,
    # ── TypedDict（旧）──
    ToolResultData,
    AssistantMessageData,
    AssistantMessageCompleteData,
    RetryData,
    # ── A) ftre 旧事件基类 + 子类 ──
    AgentEvent,
    ToolResultEvent,
    AssistantMessageEvent,
    AssistantMessageCompleteEvent,
    StepEvent,
    RetryEvent,
    UserMessageEvent,
    # ── A) 旧工厂函数 ──
    assistant_message_event,
    tool_result_event,
    assistant_message_complete_event,
    step_event,
    user_message_event,
    retry_event,
    # ── A) 内部辅助（保兼容）──
    _from_type,
    _convert_image_file_part,
    # ── B) AgentScope 对齐事件基类 ──
    EventBase,
    # ── B) 生命周期 ──
    ReplyStartEvent,
    ReplyEndEvent,
    ExceedMaxItersEvent,
    # ── B) 模型调用 ──
    ModelCallStartEvent,
    ModelCallEndEvent,
    # ── B) 文本块流式 ──
    TextBlockStartEvent,
    TextBlockDeltaEvent,
    TextBlockEndEvent,
    # ── B) 数据块流式 ──
    DataBlockStartEvent,
    DataBlockDeltaEvent,
    DataBlockEndEvent,
    # ── B) 思考块流式 ──
    ThinkingBlockStartEvent,
    ThinkingBlockDeltaEvent,
    ThinkingBlockEndEvent,
    # ── B) 提示块（一次性）──
    HintBlockEvent,
    # ── B) 工具调用流式 ──
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    # ── B) 工具结果流式 ──
    ToolResultStartEvent,
    ToolResultTextDeltaEvent,
    ToolResultDataDeltaEvent,
    ToolResultEndEvent,
    # ── B) 人工介入 ──
    RequireUserConfirmEvent,
    RequireExternalExecutionEvent,
    UserConfirmResultEvent,
    UserInterruptEvent,
    ExternalExecutionResultEvent,
    # ── B) 辅助模型 ──
    ConfirmResult,
    # ── B) 自定义扩展 ──
    CustomEvent,
    # ── B) 新事件联合类型 ──
    AgentStreamEvent,
)

__all__ = [
    # 枚举
    "EventType", "StepPhase", "DoneReason", "ReplyFinishedReason",
    # TypedDict
    "ToolResultData", "AssistantMessageData", "AssistantMessageCompleteData", "RetryData",
    # A) ftre 旧事件
    "AgentEvent", "ToolResultEvent", "AssistantMessageEvent",
    "AssistantMessageCompleteEvent", "StepEvent", "RetryEvent", "UserMessageEvent",
    # A) 旧工厂函数
    "assistant_message_event", "tool_result_event", "assistant_message_complete_event",
    "step_event", "user_message_event", "retry_event",
    # B) AgentScope 对齐
    "EventBase",
    "ReplyStartEvent", "ReplyEndEvent", "ExceedMaxItersEvent",
    "ModelCallStartEvent", "ModelCallEndEvent",
    "TextBlockStartEvent", "TextBlockDeltaEvent", "TextBlockEndEvent",
    "DataBlockStartEvent", "DataBlockDeltaEvent", "DataBlockEndEvent",
    "ThinkingBlockStartEvent", "ThinkingBlockDeltaEvent", "ThinkingBlockEndEvent",
    "HintBlockEvent",
    "ToolCallStartEvent", "ToolCallDeltaEvent", "ToolCallEndEvent",
    "ToolResultStartEvent", "ToolResultTextDeltaEvent",
    "ToolResultDataDeltaEvent", "ToolResultEndEvent",
    "RequireUserConfirmEvent", "RequireExternalExecutionEvent",
    "UserConfirmResultEvent", "UserInterruptEvent", "ExternalExecutionResultEvent",
    "ConfirmResult", "CustomEvent", "AgentStreamEvent",
]
