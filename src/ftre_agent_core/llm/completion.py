"""
OpenAI Chat Completions 流式客户端。

事件模型参考 opencode 的 LLMEvent tagged union：

  Provider 流式产出：
    TextDelta       assistant 文本增量
    ReasoningDelta  reasoning 文本增量
    ToolInputDelta  工具参数 JSON 原始片段
    ToolCall        流结束后组装完成的工具调用
    StepFinish      一轮 provider 调用结束，包含 finish_reason 和 usage

provider 异常会直接 raise LLMError，由调用方决定是否重试。
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

def _normalize_chat_messages(messages: list[dict]) -> list[dict]:
    """复制并规范化 Chat Completions 消息，不污染调用方的 memory/history。

    遵循 OpenAI-compatible 工具调用语义：
    - assistant 内容为空（包括 reasoning-only）时统一设 content=None
    - 不使用占位符字符串：模型会将其作为可见文本 echo 回来，
      导致 runner 误判为正常完成而停止
    """
    def normalize_assistant_content(message: dict) -> None:
        content = message.get("content")
        is_empty = content is None or content == [] or (
            isinstance(content, str) and not content.strip()
        )
        if is_empty:
            message["content"] = None

    def has_assistant_payload(message: dict) -> bool:
        # OpenAI-compatible providers require visible content or tool_calls.
        # reasoning_content alone is provider-specific metadata, not a valid
        # assistant payload.
        return bool(message.get("content"))

    # OpenAI-compatible provider 要求 assistant.tool_calls 后紧跟、且只紧跟
    # 每个 call 一条对应 tool result。持久化快照在中断或并发工具时可能不完整，
    # 这里不重排用户历史，只移除无效的协议片段以避免整个请求被 400 拒绝。
    normalized: list[dict] = []
    dropped_orphan_results = 0
    stripped_incomplete_calls = 0
    index = 0
    while index < len(messages):
        current = dict(messages[index])
        role = current.get("role")

        if role == "tool":
            # 只有紧跟对应 tool_calls 的分支才允许 result 通过。
            dropped_orphan_results += 1
            index += 1
            continue

        if role != "assistant":
            normalized.append(current)
            index += 1
            continue

        normalize_assistant_content(current)
        tool_calls = current.get("tool_calls") or []
        if not tool_calls:
            if has_assistant_payload(current):
                normalized.append(current)
            index += 1
            continue

        call_ids = [call.get("id") for call in tool_calls if isinstance(call, dict)]
        valid_call_ids = (
            len(call_ids) == len(tool_calls)
            and all(isinstance(call_id, str) and call_id for call_id in call_ids)
            and len(set(call_ids)) == len(call_ids)
        )
        following: list[dict] = []
        cursor = index + 1
        while cursor < len(messages) and messages[cursor].get("role") == "tool":
            following.append(dict(messages[cursor]))
            cursor += 1

        result_ids = [result.get("tool_call_id") for result in following]
        complete_pair = (
            valid_call_ids
            and len(following) == len(call_ids)
            and set(result_ids) == set(call_ids)
            and all(isinstance(result_id, str) and result_id for result_id in result_ids)
        )
        if complete_pair:
            normalized.append(current)
            normalized.extend(following)
            index = cursor
            continue

        # 没有结果、结果不全，或被其他消息隔开：保留可读 assistant 文本，
        # 解除其对 tool result 的协议约束；后续 tool result 会作为 orphan 丢弃。
        current.pop("tool_calls", None)
        stripped_incomplete_calls += 1
        if has_assistant_payload(current):
            normalized.append(current)
        index += 1

    if stripped_incomplete_calls or dropped_orphan_results:
        logger.warning(
            "[completion] normalized invalid tool protocol: stripped_calls=%d orphan_results=%d",
            stripped_incomplete_calls,
            dropped_orphan_results,
        )
    return normalized


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

        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None) if response is not None else None
        if status_code == 400:
            return LLMError(message=str(exc), code="bad_request")
        if status_code in (401, 402):
            return LLMError(message=str(exc), code="auth_error" if status_code == 401 else "bad_request")
        if status_code == 403:
            return LLMError(message=str(exc), code="content_filter")

        message_lower = str(exc).lower()
        if "invalidparameter" in message_lower or "invalid parameter" in message_lower:
            return LLMError(message=str(exc), code="bad_request")

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
    finish_reason: str = "unknown"
    usage: dict | None = None
    response_metadata: dict = field(default_factory=dict)


# 统一事件类型别名。
LLMEvent = (
    TextDelta | ReasoningDelta | ToolInputDelta | ToolCall | StepFinish
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
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str = "",
    ):
        self.model = model
        self.api_type = api_type
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort or ""
        self._active_stream = None
        self._active_loop: asyncio.AbstractEventLoop | None = None
        self._cancelled: bool = False
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
            max_retries=max_retries,
        )

    def cancel(self) -> None:
        """取消当前请求：设置 flag 让 stream 循环主动退出。

        在主循环化之后，cancel() 在主循环的同步上下文里被调用，
        而 stream() 也在同一个循环里 await。所以不需要
        call_soon_threadsafe 跨循环投递——只需设置 _cancelled flag，
        stream 循环会在下一个 chunk 到达时检查 flag 并主动 break。
        """
        if self._active_stream is None:
            return
        self._cancelled = True
        self._active_stream = None

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncGenerator[LLMEvent, None]:
        """执行一次 provider turn，并产出 LLMEvent。"""
        if self.api_type == "responses":
            async for event in self._stream_responses(messages, tools):
                yield event
            return
        async for event in self._stream_completions(messages, tools):
            yield event

    async def _stream_completions(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncGenerator[LLMEvent, None]:
        """OpenAI Chat Completions 流式路径。"""
        request_messages = _normalize_chat_messages(messages)
        llm_log = LLMLogger(self.model)
        llm_log.log_input(request_messages, tools)

        params: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "tool_choice": "auto",
        }
        if self.max_tokens is not None:
            params["max_tokens"] = max(1, int(self.max_tokens))
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
            # thinking 参数仅 DeepSeek 模型支持
            if "deepseek" in self.model.lower():
                params["extra_body"] = {"thinking": {"type": "enabled"}}
        if tools:
            params["tools"] = tools

        response = None
        try:
            self._active_loop = asyncio.get_running_loop()
            response = await self._client.chat.completions.create(**params)
            self._active_stream = response

            accumulator = _ToolCallAccumulator()
            usage: dict | None = None
            finish_reason: str = "unknown"
            response_metadata: dict[str, Any] = {}

            async for chunk in response:
                if self._cancelled:
                    logger.info("[completion] stream cancelled by cancel()")
                    break

                llm_log.log_chunk(chunk)

                for key in ("id", "model", "created", "system_fingerprint"):
                    value = _get_attr(chunk, key)
                    if value is not None:
                        response_metadata[key] = value

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

            # 仅在异常 finish_reason 时打印（正常 stop / tool_calls 不打印）
            if finish_reason not in ("stop", "tool_calls", "length"):
                logger.warning(
                    "[completion] stream ended: finish_reason=%s has_tool_calls=%s",
                    finish_reason,
                    accumulator.has_data,
                )

            yield StepFinish(
                finish_reason=finish_reason,
                usage=usage,
                response_metadata=response_metadata,
            )

        except Exception as exc:
            raise LLMError.classify(exc) from exc
        finally:
            self._active_stream = None
            self._active_loop = None
            self._cancelled = False
            if response is not None:
                close_result = response.close()
                if inspect.isawaitable(close_result):
                    await close_result
            llm_log.flush()

    # ── Responses API 路径 ──────────────────────────────────────

    @staticmethod
    def _convert_messages_to_responses_input(
        messages: list[dict],
    ) -> tuple[str | None, list[dict]]:
        """Chat Completions messages → Responses API (instructions, input)。

        转换规则：
          system       → instructions 参数（多条拼接）
          user         → {role: user, content}
          assistant    → {role: assistant, content}（无 tool_calls 时）
          assistant+tc → 先 assistant content（如有），再若干 function_call 条目
          tool         → {type: function_call_output, call_id, output}
        """
        import json as _json

        instructions: str | None = None
        input_items: list[dict] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = "\n".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") in ("text", "input_text")
                    )
                else:
                    text = ""
                if text:
                    instructions = text if not instructions else f"{instructions}\n\n{text}"

            elif role == "user":
                input_items.append({"role": "user", "content": content})

            elif role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    if content:
                        input_items.append({"role": "assistant", "content": content})
                    for tc in tool_calls:
                        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                        input_items.append({
                            "type": "function_call",
                            "call_id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", ""),
                        })
                else:
                    input_items.append({"role": "assistant", "content": content})

            elif role == "tool":
                call_id = msg.get("tool_call_id", "")
                if isinstance(content, str):
                    output = content
                else:
                    output = _json.dumps(content, ensure_ascii=False)
                input_items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                })

        return instructions, input_items

    @staticmethod
    def _convert_tools_to_responses(tools: list[dict]) -> list[dict]:
        """Chat Completions tools → Responses API tools。

        Chat Completions: {type: function, function: {name, description, parameters}}
        Responses API:    {type: function, name, description, parameters}
        """
        result: list[dict] = []
        for tool in tools:
            if tool.get("type") == "function":
                fn = tool.get("function", {})
                result.append({
                    "type": "function",
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                })
        return result

    async def _stream_responses(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncGenerator[LLMEvent, None]:
        """OpenAI Responses API 流式路径（支持 reasoning + tools 同时使用）。

        产出与 _stream_completions 相同的 LLMEvent，调用方无需区分。
        """
        import json as _json

        # Responses API 也从同一份 Chat 消息历史转换而来，先复用同一个工具调用
        # 协议边界，避免把孤立 function_call_output 传给 provider。
        request_messages = _normalize_chat_messages(messages)
        llm_log = LLMLogger(self.model)
        llm_log.log_input(request_messages, tools)

        instructions, input_items = self._convert_messages_to_responses_input(request_messages)
        resp_tools = self._convert_tools_to_responses(tools) if tools else []

        params: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "stream": True,
            "store": False,
            "tool_choice": "auto",
        }
        if instructions is not None:
            params["instructions"] = instructions
        if self.max_tokens is not None:
            params["max_output_tokens"] = max(1, int(self.max_tokens))
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.reasoning_effort and self.reasoning_effort != "none":
            params["reasoning"] = {"effort": self.reasoning_effort}
        if resp_tools:
            params["tools"] = resp_tools

        response = None
        try:
            self._active_loop = asyncio.get_running_loop()
            response = await self._client.responses.create(**params)
            self._active_stream = response

            usage: dict | None = None
            finish_reason: str = "unknown"
            response_metadata: dict[str, Any] = {}
            collected_tool_calls: list[ToolCall] = []
            has_function_call = False

            # item_id → {call_id, name}，从 OutputItemAdded 获取
            fc_info: dict[str, dict] = {}

            async for event in response:
                if self._cancelled:
                    logger.info("[completion] responses stream cancelled by cancel()")
                    break

                llm_log.log_chunk(event)

                event_type = type(event).__name__

                # ── 文本增量 ──
                if event_type == "ResponseTextDeltaEvent":
                    delta = _get_attr(event, "delta")
                    if delta:
                        yield TextDelta(text=delta)

                # ── 推理增量（如果网关支持流式返回推理文本）──
                elif event_type == "ResponseReasoningDeltaEvent":
                    delta = _get_attr(event, "delta")
                    if delta:
                        yield ReasoningDelta(text=delta)

                # ── 工具调用参数增量 ──
                elif event_type == "ResponseFunctionCallArgumentsDeltaEvent":
                    item_id = _get_attr(event, "item_id", "")
                    delta = _get_attr(event, "delta", "")
                    info = fc_info.get(item_id, {})
                    if delta:
                        yield ToolInputDelta(
                            id=info.get("call_id", ""),
                            name=info.get("name", ""),
                            text=delta,
                        )

                # ── 输出项添加：记录 function_call 的 call_id 和 name ──
                elif event_type == "ResponseOutputItemAddedEvent":
                    item = _get_attr(event, "item", {})
                    if _get_attr(item, "type") == "function_call":
                        item_id = _get_attr(item, "id", "")
                        fc_info[item_id] = {
                            "call_id": _get_attr(item, "call_id", ""),
                            "name": _get_attr(item, "name", ""),
                        }

                # ── 输出项完成：收集完整的 function_call ──
                elif event_type == "ResponseOutputItemDoneEvent":
                    item = _get_attr(event, "item", {})
                    if _get_attr(item, "type") == "function_call":
                        has_function_call = True
                        call_id = _get_attr(item, "call_id", "")
                        name = _get_attr(item, "name", "")
                        arguments = _get_attr(item, "arguments", "")
                        try:
                            parsed = _json.loads(arguments) if arguments else {}
                        except _json.JSONDecodeError:
                            logger.warning(
                                "[completion] responses 工具 %s 的 JSON 参数解析失败: %r",
                                name, arguments[:200],
                            )
                            parsed = None
                        collected_tool_calls.append(
                            ToolCall(id=call_id, name=name, input=parsed)
                        )

                # ── 响应完成 ──
                elif event_type == "ResponseCompletedEvent":
                    resp = _get_attr(event, "response", {})
                    usage = normalize_usage(_get_attr(resp, "usage"))
                    for key in ("id", "model", "created_at"):
                        value = _get_attr(resp, key)
                        if value is not None:
                            response_metadata[key] = value
                    status = _get_attr(resp, "status", "")
                    incomplete = _get_attr(resp, "incomplete_details", None)
                    if incomplete:
                        finish_reason = "length"
                    elif has_function_call:
                        finish_reason = "tool_calls"
                    elif status == "completed":
                        finish_reason = "stop"

            # yield 收集到的 tool calls
            for tc in collected_tool_calls:
                yield tc

            if finish_reason not in ("stop", "tool_calls", "length"):
                logger.warning(
                    "[completion] responses stream ended: finish_reason=%s has_tool_calls=%s",
                    finish_reason,
                    bool(collected_tool_calls),
                )

            yield StepFinish(
                finish_reason=finish_reason,
                usage=usage,
                response_metadata=response_metadata,
            )

        except Exception as exc:
            raise LLMError.classify(exc) from exc
        finally:
            self._active_stream = None
            self._active_loop = None
            self._cancelled = False
            if response is not None:
                close_result = response.close()
                if inspect.isawaitable(close_result):
                    await close_result
            llm_log.flush()
