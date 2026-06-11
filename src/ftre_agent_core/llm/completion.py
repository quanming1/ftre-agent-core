"""
OpenAI Chat Completions 流式客户端。

事件模型参考 opencode 的 LLMEvent tagged union：

  Provider 流式产出：
    StepStart       一轮 provider 调用开始
    TextDelta       assistant 文本增量
    ReasoningDelta  reasoning 文本增量
    ToolInputDelta  工具参数 JSON 原始片段
    ToolCall        流结束后组装完成的工具调用
    StepFinish      一轮 provider 调用结束，包含 finish_reason 和 usage

  Core 后续注入：
    ToolResult      工具执行成功结果
    ToolError       工具执行失败结果

  错误：
    ProviderError   provider 异常分类后的事件

所有事件都是带 type 字段的 dataclass，调用方可以用 isinstance()
或者 event.type 做分支判断。
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

import openai

from .utils import LLMLogger

logger = logging.getLogger(__name__)


# LLM 错误分类
@dataclass
class LLMError(Exception):
    """runner 重试策略使用的统一 LLM 错误。"""

    message: str
    code: str

    # 异常类型 → 错误码映射
    _TYPE_MAP = {
        openai.RateLimitError: "rate_limit",
        openai.APITimeoutError: "timeout",
        openai.APIConnectionError: "network",
        openai.AuthenticationError: "auth_error",
        openai.PermissionDeniedError: "content_filter",
        openai.BadRequestError: "bad_request",
        openai.InternalServerError: "internal_server_error",
        openai.APIError: "api_error",
    }

    # exc.code 优先级覆盖（OpenAI SDK 的 APIError 可能带更精确 code）
    _CODE_OVERRIDES = {
        "invalid_request_error": "bad_request",
        "bad_request": "bad_request",
        "rate_limit_exceeded": "rate_limit",
        "context_length_exceeded": "bad_request",
        "invalid_api_key": "auth_error",
        "authentication_error": "auth_error",
        "permission_denied": "content_filter",
    }

    # 不可重试的错误码
    UNRETRYABLE_CODES = {"auth_error", "bad_request", "content_filter"}

    @staticmethod
    def classify(exc: Exception) -> "LLMError":
        # 优先用 SDK 的 exc.code（更精确）
        if isinstance(exc, openai.APIError) and hasattr(exc, "code") and exc.code:
            code = LLMError._CODE_OVERRIDES.get(exc.code)
            if code:
                return LLMError(message=str(exc), code=code)

        # 按异常类型映射
        for exc_type, code in LLMError._TYPE_MAP.items():
            if isinstance(exc, exc_type):
                return LLMError(message=str(exc), code=code)

        # httpx 兜底
        try:
            import httpx
            if isinstance(exc, httpx.RemoteProtocolError):
                return LLMError(message=str(exc), code="network")
            if isinstance(exc, httpx.ReadTimeout):
                return LLMError(message=str(exc), code="timeout")
        except ImportError:
            pass

        return LLMError(message=str(exc), code="unknown")


# LLM 事件类型
@dataclass
class StepStart:
    """一轮 provider 调用开始。"""
    type: str = field(default="step-start", init=False)


@dataclass
class TextDelta:
    """assistant 文本增量。"""
    type: str = field(default="text-delta", init=False)
    text: str = ""


@dataclass
class ReasoningDelta:
    """reasoning 文本增量。"""
    type: str = field(default="reasoning-delta", init=False)
    text: str = ""


@dataclass
class ToolInputDelta:
    """单个工具调用的参数 JSON 原始片段。"""
    type: str = field(default="tool-input-delta", init=False)
    id: str = ""
    name: str = ""
    text: str = ""


@dataclass
class ToolCall:
    """已经组装完整、可以执行的工具调用。"""
    type: str = field(default="tool-call", init=False)
    id: str = ""
    name: str = ""
    # None 表示 JSON parse 失败，调用方必须按工具错误处理。
    input: dict | None = field(default_factory=dict)


@dataclass
class StepFinish:
    """一轮 provider 调用结束。"""
    type: str = field(default="step-finish", init=False)
    finish_reason: str = "stop"
    usage: dict | None = None


@dataclass
class ToolResult:
    """工具执行成功结果；由 core 注入，不来自 provider。"""
    type: str = field(default="tool-result", init=False)
    id: str = ""
    name: str = ""
    result: str = ""


@dataclass
class ToolError:
    """工具执行失败结果；由 core 注入，不来自 provider。"""
    type: str = field(default="tool-error", init=False)
    id: str = ""
    name: str = ""
    message: str = ""


@dataclass
class ProviderError:
    """provider 异常分类后的事件。"""
    type: str = field(default="provider-error", init=False)
    message: str = ""
    code: str = "unknown"
    retryable: bool = False


# 统一事件类型别名。
LLMEvent = (
    StepStart | TextDelta | ReasoningDelta | ToolInputDelta
    | ToolCall | StepFinish | ToolResult | ToolError | ProviderError
)


# 内部辅助函数
def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def normalize_usage(usage: Any) -> dict | None:
    """把 SDK usage 对象规范化成普通 dict。"""
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    if hasattr(usage, "__dict__"):
        return {k: v for k, v in vars(usage).items() if v is not None and not k.startswith("_")}
    return None


class _ToolCallAccumulator:
    """按 OpenAI streaming 的 index 累积 tool_call delta。"""

    def __init__(self):
        # index -> {id, name, arguments}
        self._items: dict[int, dict] = {}

    @property
    def has_data(self) -> bool:
        return bool(self._items)

    def feed(self, tc_delta: Any) -> ToolInputDelta | None:
        """喂入一个 delta；如果包含新的参数片段，则返回 ToolInputDelta。"""
        index = _get_attr(tc_delta, "index")
        if index is None:
            index = len(self._items)

        entry = self._items.setdefault(
            index,
            {"id": "", "name": "", "arguments": ""},
        )

        function = _get_attr(tc_delta, "function")
        call_id = _get_attr(tc_delta, "id") or ""
        name = _get_attr(function, "name") or ""
        args_fragment = _get_attr(function, "arguments") or ""

        if call_id:
            entry["id"] = call_id
        if name:
            entry["name"] = name
        if args_fragment:
            entry["arguments"] += args_fragment

        if args_fragment:
            return ToolInputDelta(
                id=entry["id"],
                name=entry["name"],
                text=args_fragment,
            )
        return None

    def finalize(self) -> list[ToolCall]:
        """流结束后生成完整 ToolCall 列表。"""
        import json
        calls: list[ToolCall] = []
        for _, entry in sorted(self._items.items()):
            if not entry["id"] or not entry["name"]:
                continue
            try:
                parsed = json.loads(entry["arguments"]) if entry["arguments"] else {}
            except json.JSONDecodeError:
                logger.warning(
                    "[accumulator] 工具 %s 的 JSON 参数解析失败: %r",
                    entry["name"], entry["arguments"][:200],
                )
                parsed = None
            calls.append(ToolCall(id=entry["id"], name=entry["name"], input=parsed))
        return calls


class LLMHandler:
    """OpenAI Chat Completions 异步流式封装。

    正常情况下产出顺序：

        StepStart
        ReasoningDelta* / TextDelta* / ToolInputDelta*
        ToolCall*
        StepFinish

    调用方负责执行 ToolCall，并把工具结果写入自身的事件流和 memory。
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str | None = None,
        api_type: str = "completions",
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        if api_type != "completions":
            logger.warning(
                "当前只支持 OpenAI chat completions；忽略 api_type=%s", api_type
            )

        self.model = model
        self._active_stream = None
        self._active_loop: asyncio.AbstractEventLoop | None = None
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
            max_retries=max_retries,
        )

    def cancel(self) -> None:
        """取消当前请求，并尽快关闭正在读取的 HTTP stream。"""
        stream = self._active_stream
        if stream is None:
            return
        self._active_stream = None
        loop = self._active_loop
        if loop is None or loop.is_closed():
            return

        async def close_stream() -> None:
            close_result = stream.close()
            if inspect.isawaitable(close_result):
                await close_result

        loop.call_soon_threadsafe(lambda: asyncio.create_task(close_stream()))

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncGenerator[LLMEvent, None]:
        """执行一次 provider turn，并产出 LLMEvent。"""
        llm_log = LLMLogger(self.model)
        llm_log.log_input(messages, tools)

        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            params["tools"] = tools

        response = None
        try:
            self._active_loop = asyncio.get_running_loop()
            response = await self._client.chat.completions.create(**params)
            self._active_stream = response

            accumulator = _ToolCallAccumulator()
            usage: dict | None = None
            finish_reason: str = "stop"

            yield StepStart()

            async for chunk in response:
                llm_log.log_chunk(chunk)

                # OpenAI 会在最后额外返回一个 usage-only chunk。
                chunk_usage = normalize_usage(_get_attr(chunk, "usage"))
                if chunk_usage:
                    usage = chunk_usage

                choices = _get_attr(chunk, "choices", []) or []
                if not choices:
                    continue

                choice = choices[0]
                delta = _get_attr(choice, "delta")

                # 部分推理模型会把 reasoning 放在 reasoning_content 字段。
                reasoning = _get_attr(delta, "reasoning_content")
                if reasoning:
                    yield ReasoningDelta(text=reasoning)

                # 普通 assistant 文本。
                content = _get_attr(delta, "content")
                if content:
                    yield TextDelta(text=content)

                # 工具调用参数的 JSON 流式片段。
                tc_deltas = _get_attr(delta, "tool_calls") or []
                for tc_delta in tc_deltas:
                    event = accumulator.feed(tc_delta)
                    if event is not None:
                        yield event

                fr = _get_attr(choice, "finish_reason")
                if fr:
                    finish_reason = fr

            # OpenAI Chat 没有 per-tool 结束事件，流结束后统一 finalize。
            for tc in accumulator.finalize():
                yield tc

            logger.info(
                "[completion] stream ended: finish_reason=%s has_tool_calls=%s",
                finish_reason,
                accumulator.has_data,
            )

            yield StepFinish(finish_reason=finish_reason, usage=usage)

        except Exception as exc:
            err = LLMError.classify(exc)
            yield ProviderError(
                message=err.message,
                code=err.code,
                retryable=err.code not in {"auth_error", "bad_request", "content_filter"},
            )
        finally:
            self._active_stream = None
            self._active_loop = None
            if response is not None:
                close_result = response.close()
                if inspect.isawaitable(close_result):
                    await close_result
            llm_log.flush()
