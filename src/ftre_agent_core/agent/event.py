"""

Agent 事件定义

事件结构：{"type": EventType, "data": <TypedDict>}

"""

from enum import Enum

from typing import Any, TypedDict

class EventType(str, Enum):

    TOOL_CALL = "tool_call"

    TOOL_RESULT = "tool_result"

    TOOL_CANCEL_REQUESTED = "tool_cancel_requested"

    TOOL_CANCELLED = "tool_cancelled"

    TOOL_TIMED_OUT = "tool_timed_out"

    MESSAGE = "message"

    MESSAGE_COMPLETE = "message_complete"

    MAX_ITERATIONS = "max_iterations"

    INTERRUPT = "interrupt"

    ERROR = "error"

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

class ToolResultData(TypedDict):
    id: str
    name: str
    result: str
    error: str | None
    status: str
    error_code: str | None

class ToolLifecycleData(TypedDict, total=False):
    id: str
    name: str
    arguments: dict[str, Any]
    reason: str
    status: str
    error_code: str | None
    result_status: str | None


class MessageData(TypedDict):

    content: str

class MessageCompleteData(TypedDict):

    content: str

class MaxIterationsData(TypedDict):

    iterations: int

class DoneData(TypedDict, total=False):

    success: bool

    reason: DoneReason

    usage: dict

class InterruptData(TypedDict):

    tool_call_id: str

    tool_name: str

    arguments: dict[str, Any]

    checkpoint_id: str

class UsageUpdateData(TypedDict):

    usage: dict

class ErrorData(TypedDict):

    message: str

    code: str

class ToolCallStreamingData(TypedDict):
    tool_calls: list[dict]

class ToolCallStreamingEvent(TypedDict):
    type: EventType
    data: ToolCallStreamingData

class ToolCallEvent(TypedDict):

    type: EventType

    data: ToolCallData

class ToolResultEvent(TypedDict):

    type: EventType

    data: ToolResultData

class ToolLifecycleEvent(TypedDict):

    type: EventType

    data: ToolLifecycleData

class MessageEvent(TypedDict):

    type: EventType

    data: MessageData

class MessageCompleteEvent(TypedDict):

    type: EventType

    data: MessageCompleteData

class MaxIterationsEvent(TypedDict):

    type: EventType

    data: MaxIterationsData

class DoneEvent(TypedDict):

    type: EventType

    data: DoneData

class InterruptEvent(TypedDict):

    type: EventType

    data: InterruptData

class UsageUpdateEvent(TypedDict):

    type: EventType

    data: UsageUpdateData

class ErrorEvent(TypedDict):

    type: EventType

    data: ErrorData

AgentEvent = (

    ToolCallEvent

    | ToolResultEvent

    | ToolLifecycleEvent

    | ToolCallStreamingEvent

    | MessageEvent

    | MessageCompleteEvent

    | MaxIterationsEvent

    | DoneEvent

    | InterruptEvent

    | UsageUpdateEvent

    | ErrorEvent

)

def tool_call_streaming_event(chunks: list) -> ToolCallStreamingEvent:
    """从 ToolCallDeltaChunk 列表构造流式事件"""
    return {
        "type": EventType.TOOL_CALL_STREAMING,
        "data": {
            "tool_calls": [
                {k: v for k, v in {
                    "index": c.index,
                    "id": c.id,
                    "name": c.name,
                    "arguments_delta": c.arguments_delta,
                }.items() if v is not None}
                for c in chunks
            ]
        }
    }

def tool_call_event(id: str, name: str, arguments: dict[str, Any]) -> ToolCallEvent:

    return {"type": EventType.TOOL_CALL, "data": {"id": id, "name": name, "arguments": arguments}}

def tool_result_event(

    id: str,

    name: str,

    result: str,

    error: str | None = None,

    *,

    status: str = "completed",

    error_code: str | None = None,

) -> ToolResultEvent:

    return {

        "type": EventType.TOOL_RESULT,

        "data": {

            "id": id,

            "name": name,

            "result": result,

            "error": error,

            "status": status,

            "error_code": error_code,

        },

    }

def tool_cancel_requested_event(

    id: str,

    name: str,

    reason: str = "user_cancelled",

    *,

    error_code: str | None = None,

    result_status: str | None = "cancelled",

) -> ToolLifecycleEvent:

    return {"type": EventType.TOOL_CANCEL_REQUESTED, "data": {"id": id, "name": name, "reason": reason, "status": "cancelling", "error_code": error_code, "result_status": result_status}}

def tool_cancelled_event(

    id: str,

    name: str,

    reason: str = "user_cancelled",

    *,

    error_code: str | None = "cancelled",

    result_status: str | None = "cancelled",

) -> ToolLifecycleEvent:

    return {"type": EventType.TOOL_CANCELLED, "data": {"id": id, "name": name, "reason": reason, "status": "cancelled", "error_code": error_code, "result_status": result_status}}

def tool_timed_out_event(

    id: str,

    name: str,

    reason: str = "timed_out",

    *,

    error_code: str | None = "timed_out",

    result_status: str | None = "timed_out",

) -> ToolLifecycleEvent:

    return {"type": EventType.TOOL_TIMED_OUT, "data": {"id": id, "name": name, "reason": reason, "status": "timed_out", "error_code": error_code, "result_status": result_status}}

def message_event(content: str) -> MessageEvent:

    return {"type": EventType.MESSAGE, "data": {"content": content}}

def message_complete_event(content: str) -> MessageCompleteEvent:

    return {"type": EventType.MESSAGE_COMPLETE, "data": {"content": content}}

def max_iterations_event(iterations: int) -> MaxIterationsEvent:

    return {"type": EventType.MAX_ITERATIONS, "data": {"iterations": iterations}}

def done_event(success: bool, reason: DoneReason, usage: dict | None = None) -> DoneEvent:

    data: DoneData = {"success": success, "reason": reason}

    if usage:

        data["usage"] = usage

    return {"type": EventType.DONE, "data": data}

def interrupt_event(tool_call_id: str, tool_name: str, arguments: dict[str, Any], checkpoint_id: str) -> InterruptEvent:

    return {

        "type": EventType.INTERRUPT,

        "data": {

            "tool_call_id": tool_call_id,

            "tool_name": tool_name,

            "arguments": arguments,

            "checkpoint_id": checkpoint_id,

        },

    }

def usage_update_event(usage: dict) -> UsageUpdateEvent:

    return {"type": EventType.USAGE_UPDATE, "data": {"usage": usage}}

def error_event(message: str, code: str = "unknown") -> ErrorEvent:

    return {"type": EventType.ERROR, "data": {"message": message, "code": code}}

