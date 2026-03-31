"""
LLM 相关类型定义
"""
import litellm
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMError:
    """LLM 调用错误"""
    message: str
    code: str

    @staticmethod
    def classify(e: Exception) -> "LLMError":
        """根据异常类型分类错误（适配 LiteLLM）"""
        if isinstance(e, litellm.RateLimitError):
            return LLMError(message=f"请求频率超限: {e}", code="rate_limit")
        if isinstance(e, litellm.Timeout):
            return LLMError(message=f"请求超时: {e}", code="timeout")
        if isinstance(e, litellm.APIConnectionError):
            return LLMError(message=f"网络连接失败: {e}", code="network")
        if isinstance(e, litellm.ContentPolicyViolationError):
            return LLMError(message=f"内容审核未通过: {e}", code="content_filter")
        if isinstance(e, litellm.APIError):
            return LLMError(message=f"API 错误: {e}", code="api_error")
        return LLMError(message=f"未知错误: {e}", code="unknown")


@dataclass
class LLMResponse:
    """LLM 完整响应（用于 tool_calls 场景）"""
    content: str | None = None
    tool_calls: list[Any] = field(default_factory=list)
    usage: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass
class ToolCallDeltaChunk:
    """单个 tool_call 的增量信息"""
    index: int
    id: str | None = None
    name: str | None = None
    arguments_delta: str | None = None


@dataclass
class StreamDelta:
    """流式输出的 delta 片段"""
    content: str | None = None
    tool_calls: list[ToolCallDeltaChunk] | None = None
    usage: Any = None


class ToolCallWrapper:
    """统一的 tool_call 对象"""
    def __init__(self, data: dict):
        self.id = data["id"]
        self.type = data["type"]
        self.function = _FunctionWrapper(data["function"])


class _FunctionWrapper:
    """模拟 function 对象（tc.function.name / tc.function.arguments）"""
    def __init__(self, data: dict):
        self.name = data["name"]
        self.arguments = data["arguments"]
