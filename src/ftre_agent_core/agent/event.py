"""
Agent 事件定义。

事件统一结构：{"type": EventType, "data": <TypedDict>}
"""
from enum import Enum
from typing import Any, TypedDict


class EventType(str, Enum):
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    MESSAGE = "message"
    MESSAGE_COMPLETE = "message_complete"
    REASONING = "reasoning"
    REASONING_COMPLETE = "reasoning_complete"
    ERROR = "error"
    RETRY = "retry"
    DONE = "done"
    TOOL_CALL_STREAMING = "tool_call_streaming"
    USAGE_UPDATE = "usage_update"


class DoneReason(str, Enum):
    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations"
    ERROR = "error"
    CANCELLED = "cancelled"


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


class MessageData(TypedDict):
    content: str


class MessageCompleteData(TypedDict):
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


# 统一事件类型：dict with "type" + "data"
AgentEvent = dict


# ─── 事件构造函数 ────────────────────────────────────────────────

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
    return {
        "type": EventType.TOOL_CALL_STREAMING,
        "data": {"tool_calls": result}
    }


def tool_call_event(id: str, name: str, arguments: dict[str, Any]) -> AgentEvent:
    return {"type": EventType.TOOL_CALL, "data": {"id": id, "name": name, "arguments": arguments}}


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
    data: ToolResultData = {
        "id": id,
        "name": name,
        "result": result,
        "error": error,
        "status": status,
        "error_code": error_code,
    }
    if metadata:
        data["metadata"] = metadata
    return {"type": EventType.TOOL_RESULT, "data": data}


def message_event(content: str) -> AgentEvent:
    return {"type": EventType.MESSAGE, "data": {"content": content}}


def reasoning_event(content: str) -> AgentEvent:
    return {"type": EventType.REASONING, "data": {"content": content}}


def reasoning_complete_event(content: str) -> AgentEvent:
    """一轮 LLM reasoning 的完整文本，用于持久化和多轮回放。"""
    return {"type": EventType.REASONING_COMPLETE, "data": {"content": content}}


def message_complete_event(content: str) -> AgentEvent:
    return {"type": EventType.MESSAGE_COMPLETE, "data": {"content": content}}


def done_event(success: bool, reason: DoneReason, usage: dict | None = None) -> AgentEvent:
    data: DoneData = {"success": success, "reason": reason}
    if usage:
        data["usage"] = usage
    return {"type": EventType.DONE, "data": data}


def usage_update_event(usage: dict) -> AgentEvent:
    return {"type": EventType.USAGE_UPDATE, "data": {"usage": usage}}


def error_event(message: str, code: str = "unknown") -> AgentEvent:
    return {"type": EventType.ERROR, "data": {"message": message, "code": code}}


def retry_event(code: str, message: str, attempt: int, max_attempts: int) -> AgentEvent:
    return {"type": EventType.RETRY, "data": {"code": code, "message": message, "attempt": attempt, "max_attempts": max_attempts}}
