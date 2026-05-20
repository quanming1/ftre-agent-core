"""
LLM 调用 - Completions API 适配器 + 类型定义 + LLMHandler
"""
import json
import logging
import litellm
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Generator

from .utils import LLMLogger

logger = logging.getLogger(__name__)


# ============================================================
# 类型定义
# ============================================================

@dataclass
class LLMError:
    """LLM 调用错误"""
    message: str
    code: str

    @staticmethod
    def classify(e: Exception) -> "LLMError":
        if isinstance(e, litellm.RateLimitError):
            return LLMError(message=f"请求频率超限: {e}", code="rate_limit")
        if isinstance(e, litellm.Timeout):
            return LLMError(message=f"请求超时: {e}", code="timeout")
        if isinstance(e, litellm.APIConnectionError):
            return LLMError(message=f"网络连接失败: {e}", code="network")
        if isinstance(e, litellm.ContentPolicyViolationError):
            return LLMError(message=f"内容审核未通过: {e}", code="content_filter")
        if isinstance(e, litellm.AuthenticationError):
            return LLMError(message=f"认证失败: {e}", code="auth_error")
        if isinstance(e, litellm.BadRequestError):
            return LLMError(message=f"请求无效: {e}", code="bad_request")
        if isinstance(e, litellm.APIError):
            return LLMError(message=f"API 错误: {e}", code="api_error")
        return LLMError(message=f"未知错误: {e}", code="unknown")


@dataclass
class LLMResponse:
    """LLM 完整响应（tool_calls 场景）"""
    content: str | None = None
    tool_calls: list[Any] = field(default_factory=list)
    usage: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass
class ToolCallDeltaChunk:
    """单个 tool_call 的增量"""
    index: int
    id: str | None = None
    name: str | None = None
    arguments_delta: str | None = None


@dataclass
class StreamDelta:
    """流式 delta 片段"""
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCallDeltaChunk] | None = None
    usage: Any = None


class ToolCallWrapper:
    """统一的 tool_call 对象"""
    def __init__(self, data: dict):
        self.id = data["id"]
        self.type = data["type"]
        self.function = _FunctionWrapper(data["function"])


class _FunctionWrapper:
    def __init__(self, data: dict):
        self.name = data["name"]
        self.arguments = data["arguments"]


# ============================================================
# StreamAdapter 基类
# ============================================================

class StreamAdapter(ABC):
    """流式适配器基类"""

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str | None = None,
        cancelled_check: Callable[[], bool] | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self._cancelled_check = cancelled_check or (lambda: False)
        self._active_response = None

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled_check()

    def close_stream(self) -> None:
        """硬关活跃的流式连接"""
        resp = self._active_response
        if resp is None:
            return
        try:
            inner = getattr(resp, "completion_stream", None)
            if inner and hasattr(inner, "close"):
                inner.close()
            http_resp = getattr(inner, "response", None)
            if http_resp and hasattr(http_resp, "close"):
                http_resp.close()
        except Exception:
            pass

    @abstractmethod
    def stream(self, messages: list[dict], tools: list[dict] | None = None) -> Generator[StreamDelta | LLMResponse, None, None]:
        pass


# ============================================================
# ToolCallAccumulator
# ============================================================

class _SplitReason(Enum):
    ID_CHANGED = auto()
    ARGS_COMPLETE = auto()
    NAME_CHANGED = auto()


class ToolCallAccumulator:
    """流式 tool_call delta 累积器"""

    def __init__(self):
        self._buffer: dict[int, dict] = {}
        self._index_remap: dict[int, int] = {}

    def feed(self, tc) -> ToolCallDeltaChunk:
        raw_idx = tc.index
        idx = self._index_remap.get(raw_idx, raw_idx)
        existing = self._buffer.get(idx)

        split_reason = self._detect_split(existing, tc)
        if split_reason is not None:
            idx = self._allocate_new_slot(raw_idx, tc.id, split_reason)
            existing = None

        if not existing:
            self._buffer[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
        self._merge_delta(idx, tc)

        return ToolCallDeltaChunk(
            index=idx,
            id=tc.id or None,
            name=tc.function.name if tc.function and tc.function.name else None,
            arguments_delta=tc.function.arguments if tc.function and tc.function.arguments else None,
        )

    @property
    def has_data(self) -> bool:
        return len(self._buffer) > 0

    def build(self) -> list[ToolCallWrapper]:
        return [ToolCallWrapper(tc_data) for _, tc_data in sorted(self._buffer.items())]

    def _detect_split(self, existing: dict | None, tc) -> _SplitReason | None:
        if not existing:
            return None
        if tc.id and existing["id"] and tc.id != existing["id"]:
            return _SplitReason.ID_CHANGED
        if tc.function and tc.function.arguments:
            cur_args = existing["function"]["arguments"]
            if cur_args:
                try:
                    json.loads(cur_args)
                    return _SplitReason.ARGS_COMPLETE
                except json.JSONDecodeError:
                    pass
        if tc.function and tc.function.name:
            cur_name = existing["function"]["name"]
            if cur_name and cur_name != tc.function.name:
                return _SplitReason.NAME_CHANGED
        return None

    def _allocate_new_slot(self, raw_idx: int, tc_id: str | None, reason: _SplitReason) -> int:
        new_idx = max(self._buffer.keys()) + 1 if self._buffer else 0
        self._index_remap[raw_idx] = new_idx
        logger.warning(f"tool_call index 复用拆分: raw_idx={raw_idx} -> idx={new_idx}, reason={reason.name}")
        return new_idx

    def _merge_delta(self, idx: int, tc) -> None:
        entry = self._buffer[idx]
        if tc.id:
            entry["id"] = tc.id
        if tc.function:
            if tc.function.name:
                entry["function"]["name"] = tc.function.name
            if tc.function.arguments:
                entry["function"]["arguments"] += tc.function.arguments


# ============================================================
# CompletionAdapter
# ============================================================

class CompletionAdapter(StreamAdapter):
    """Completions API 适配器（litellm.completion）"""

    def stream(self, messages: list[dict], tools: list[dict] | None = None) -> Generator[StreamDelta | LLMResponse, None, None]:
        llm_log = LLMLogger(self.model)
        llm_log.log_input(messages, tools)

        response = litellm.completion(
            model=self.model,
            messages=messages,
            tools=tools if tools else None,
            api_key=self.api_key,
            api_base=self.api_base,
            stream=True,
            stream_options={"include_usage": True},
        )
        self._active_response = response

        accumulator = ToolCallAccumulator()
        content_buffer: list[str] = []
        usage = None

        try:
            for chunk in response:
                if self.is_cancelled:
                    break

                llm_log.log_chunk(chunk)

                if hasattr(chunk, "usage") and chunk.usage:
                    usage = chunk.usage
                if not hasattr(chunk, "choices") or not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    tc_deltas = [accumulator.feed(tc) for tc in delta.tool_calls]
                    yield StreamDelta(tool_calls=tc_deltas)

                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    yield StreamDelta(reasoning=reasoning)

                if hasattr(delta, "content") and delta.content:
                    content_buffer.append(delta.content)
                    yield StreamDelta(content=delta.content)

            if accumulator.has_data:
                yield LLMResponse(
                    content="".join(content_buffer) if content_buffer else None,
                    tool_calls=accumulator.build(),
                    usage=usage,
                )
            else:
                if usage:
                    yield StreamDelta(usage=usage)
        finally:
            self._active_response = None
            llm_log.flush()


# ============================================================
# LLMHandler（对外入口）
# ============================================================

class LLMHandler:
    """LLM 调用封装，根据 api_type 选择适配器。"""

    def __init__(self, model: str, api_key: str, api_base: str | None = None, api_type: str = "completions"):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self._cancelled = False
        self._adapter = self._create_adapter(api_type)

    def _create_adapter(self, api_type: str) -> StreamAdapter:
        from .responses import ResponsesAdapter
        if api_type == "responses":
            return ResponsesAdapter(model=self.model, api_key=self.api_key, api_base=self.api_base, cancelled_check=lambda: self._cancelled)
        return CompletionAdapter(model=self.model, api_key=self.api_key, api_base=self.api_base, cancelled_check=lambda: self._cancelled)

    def cancel(self) -> None:
        """取消：设标志位 + 硬关连接"""
        self._cancelled = True
        self._adapter.close_stream()

    def stream(self, messages: list[dict], tools: list[dict] | None = None) -> Generator[StreamDelta | LLMResponse, None, None]:
        self._cancelled = False
        yield from self._adapter.stream(messages, tools)
