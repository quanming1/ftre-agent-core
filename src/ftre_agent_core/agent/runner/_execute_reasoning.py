"""LLM 调用执行器：流式消费 + 重试 + 产出事件 + 返回 TurnResult。"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, AsyncGenerator

from ...event import (
    AgentStreamEvent,
    ModelCallStartEvent, ModelCallEndEvent,
    TextBlockStartEvent, TextBlockDeltaEvent, TextBlockEndEvent,
    ThinkingBlockStartEvent, ThinkingBlockDeltaEvent, ThinkingBlockEndEvent,
    ToolCallStartEvent, ToolCallDeltaEvent, ToolCallEndEvent,
    HintBlockEvent,
    RetryEvent,
)
from ...llm import LLMHandler, LLMError, TextDelta, ReasoningDelta, ToolInputDelta, ToolCall, StepFinish
from ...tracing import RunType, RunStatus as TraceRunStatus
from ._actions import Reasoning, TurnResult

if TYPE_CHECKING:
    from ...hooks import FtreCoreHookManager
    from ._state import RunState

logger = logging.getLogger(__name__)


class ReasoningExecutor:
    """执行 Reasoning 动作：调用 LLM，流式产出事件，返回 TurnResult。"""

    def __init__(
        self,
        agent,
        state: "RunState",
        llm: LLMHandler,
        hook_manager: "FtreCoreHookManager",
    ):
        self.agent = agent
        self.state = state
        self.llm = llm
        self.hook_manager = hook_manager
        self.result: TurnResult | None = None

    async def stream(self, action: Reasoning) -> AsyncGenerator[AgentStreamEvent, None]:
        """执行 LLM 调用，yield 事件，结束后设置 self.result。"""
        reply_id = self.state.reply_id
        model_name = self.agent.model

        # hint 写入 memory（在 LLM 调用之前）
        if action.hint:
            self.agent.memory.add_raw({"role": "user", "content": action.hint})
            yield HintBlockEvent(
                reply_id=reply_id,
                block_id=uuid.uuid4().hex[:16],
                source="system",
                hint=action.hint,
                metadata={"hide": True, "internal": True, "reason": "finalization_retry"},
            )

        messages = self.agent.memory.get_messages()
        tools = None if action.force_no_tools else self.agent.tool_registry.to_openai_tools() or None

        max_attempts = 1 + self.agent.max_retries
        turn_start_ts = time.perf_counter()
        first_token_logged = False

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        finish_reason = "unknown"
        usage: dict | None = None
        response_metadata: dict = {}

        for attempt in range(max_attempts):
            llm_span = None
            if self.state.trace_span:
                llm_span = self.state.trace_span.child(
                    "llm", RunType.LLM,
                    inputs={"messages": messages, "tools": tools},
                    metadata={
                        "model": model_name,
                        "api_type": self.agent.api_type,
                        "iteration": self.state.iteration,
                        "attempt": attempt + 1,
                    },
                )

            try:
                yield ModelCallStartEvent(reply_id=reply_id, model_name=model_name)

                text_block_id: str | None = None
                thinking_block_id: str | None = None
                tool_call_started: set[str] = set()

                async for event in self.llm.stream(messages, tools):
                    if not first_token_logged:
                        first_token_logged = True
                        elapsed_ms = (time.perf_counter() - turn_start_ts) * 1000
                        logger.info(f"[react] 第 {self.state.iteration} 轮 TTFT {elapsed_ms:.0f}ms")
                        if llm_span and not llm_span.ended:
                            llm_span.add_event("ttft", {"ms": round(elapsed_ms)})

                    if isinstance(event, TextDelta):
                        text_parts.append(event.text)
                        if text_block_id is None:
                            text_block_id = uuid.uuid4().hex[:16]
                            yield TextBlockStartEvent(reply_id=reply_id, block_id=text_block_id)
                        yield TextBlockDeltaEvent(reply_id=reply_id, block_id=text_block_id, delta=event.text)

                    elif isinstance(event, ReasoningDelta):
                        reasoning_parts.append(event.text)
                        if thinking_block_id is None:
                            thinking_block_id = uuid.uuid4().hex[:16]
                            yield ThinkingBlockStartEvent(reply_id=reply_id, block_id=thinking_block_id)
                        yield ThinkingBlockDeltaEvent(reply_id=reply_id, block_id=thinking_block_id, delta=event.text)

                    elif isinstance(event, ToolInputDelta):
                        if event.id not in tool_call_started:
                            tool_call_started.add(event.id)
                            yield ToolCallStartEvent(reply_id=reply_id, tool_call_id=event.id, tool_call_name=event.name or "")
                        yield ToolCallDeltaEvent(reply_id=reply_id, tool_call_id=event.id, delta=event.text)

                    elif isinstance(event, ToolCall):
                        if event.id not in tool_call_started:
                            tool_call_started.add(event.id)
                            yield ToolCallStartEvent(reply_id=reply_id, tool_call_id=event.id, tool_call_name=event.name)
                        yield ToolCallEndEvent(reply_id=reply_id, tool_call_id=event.id)
                        tool_calls.append(event)

                    elif isinstance(event, StepFinish):
                        finish_reason = event.finish_reason
                        response_metadata = event.response_metadata
                        if event.usage:
                            usage = event.usage
                            self.state.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                            self.state.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                            self.state.token_usage["cached_tokens"] += usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                            self.state.token_usage["llm_calls"] += 1

                        if text_block_id is not None:
                            yield TextBlockEndEvent(reply_id=reply_id, block_id=text_block_id)
                            text_block_id = None
                        if thinking_block_id is not None:
                            yield ThinkingBlockEndEvent(reply_id=reply_id, block_id=thinking_block_id)
                            thinking_block_id = None

                        input_tokens = usage.get("prompt_tokens", 0) if usage else 0
                        output_tokens = usage.get("completion_tokens", 0) if usage else 0
                        yield ModelCallEndEvent(
                            reply_id=reply_id,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            finished_reason=finish_reason,
                        )

                # 关闭 LLM span
                full_text = "".join(text_parts)
                full_reasoning = "".join(reasoning_parts)
                if llm_span and not llm_span.ended:
                    llm_span.end(outputs={
                        "text": full_text,
                        "reasoning": full_reasoning,
                        "finish_reason": finish_reason,
                        "has_tool_calls": bool(tool_calls),
                        "usage": usage,
                        "response_metadata": response_metadata,
                    })

                # 写入 memory
                self.agent.memory.add_assistant(full_text, reasoning=full_reasoning or None)

                # 成功完成
                self.result = TurnResult(
                    text=full_text,
                    reasoning=full_reasoning,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                    usage=usage,
                )
                return

            except asyncio.CancelledError:
                if llm_span and not llm_span.ended:
                    llm_span.end(status=TraceRunStatus.CANCELLED)
                # 收尾 open blocks
                if text_block_id is not None:
                    yield TextBlockEndEvent(reply_id=reply_id, block_id=text_block_id)
                    text_block_id = None
                if thinking_block_id is not None:
                    yield ThinkingBlockEndEvent(reply_id=reply_id, block_id=thinking_block_id)
                    thinking_block_id = None
                # 写入半截文本到 memory
                _full_text = "".join(text_parts)
                _full_reasoning = "".join(reasoning_parts)
                if _full_text.strip():
                    self.agent.memory.add_assistant(_full_text, reasoning=_full_reasoning or None)
                raise

            except Exception as exc:
                if llm_span and not llm_span.ended:
                    llm_span.end(error=exc)

                # 收尾 open blocks
                if text_block_id is not None:
                    yield TextBlockEndEvent(reply_id=reply_id, block_id=text_block_id)
                    text_block_id = None
                if thinking_block_id is not None:
                    yield ThinkingBlockEndEvent(reply_id=reply_id, block_id=thinking_block_id)
                    thinking_block_id = None

                # 写入半截文本到 memory
                _full_text = "".join(text_parts)
                _full_reasoning = "".join(reasoning_parts)
                if _full_text.strip():
                    self.agent.memory.add_assistant(_full_text, reasoning=_full_reasoning or None)

                err = exc if isinstance(exc, LLMError) else LLMError.classify(exc)
                is_last = attempt >= max_attempts - 1

                logger.warning(
                    "LLM 调用失败 [%s] %s (第 %d/%d 次尝试)",
                    err.code, err.message[:200], attempt + 1, max_attempts,
                )

                if err.code in LLMError.UNRETRYABLE_CODES or is_last:
                    self.result = TurnResult(
                        text="",
                        reasoning="",
                        tool_calls=[],
                        finish_reason="error",
                        error=err,
                    )
                    return

                # 可重试 → 发出 RetryEvent，等待后继续
                yield RetryEvent(
                    reply_id=reply_id,
                    code=err.code,
                    message=err.message,
                    attempt=attempt + 1,
                    max_attempts=max_attempts - 1,
                )
                await asyncio.sleep(self.agent.retry_delay)
                # 重新读取 messages
                messages = self.agent.memory.get_messages()
                # 重置收集器
                text_parts = []
                reasoning_parts = []
                tool_calls = []
                finish_reason = "unknown"
                usage = None
                response_metadata = {}
