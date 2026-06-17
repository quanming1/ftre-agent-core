"""
异步 ReAct 执行引擎。

整体结构参考 opencode 的 runner/llm.ts：

  _loop()        外层循环，一次迭代对应一次 provider 调用
  _run_turn()    单轮调用，负责重试和向外透传事件
  _stream_turn() 消费 LLM 事件流，遇到工具调用后并发执行工具

关键不变量：
  1. 消费 provider 流时不等待工具执行完成。
  2. 文本和 reasoning 事件实时向外 yield。
  3. 带 tool_calls 的 assistant 消息和对应 tool 结果必须成组写入 memory。
  4. 每轮开始前修复历史里缺失 tool result 的悬空 tool_call。

工具的并发调度、取消和结果归并下沉到 ToolHandler；本模块只负责控制流、
memory 写入和事件 yield。
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator, TYPE_CHECKING

from ftre_agent_core.llm import (
    LLMHandler, LLMError,
    TextDelta, ReasoningDelta, ToolInputDelta,
    ToolCall, StepFinish,
)
from ftre_agent_core.tool import CancellationToken, ToolCancelledError
from .tool_handler import ToolHandler
from ..event import (
    DoneReason,
    AgentEvent,
    assistant_message_event,
    reasoning_event,
    reasoning_complete_event,
    assistant_message_complete_event,
    done_event,
    usage_update_event,
    error_event,
    retry_event,
    tool_call_streaming_event,
    tool_call_event,
    tool_result_event,
    user_message_event,
)

if TYPE_CHECKING:
    from ..react import ReActAgent

logger = logging.getLogger(__name__)

LENGTH_CONTINUATION_PROMPT = (
    "输出长度达到上限。请从刚才中断的位置继续，不要重述已经说过的内容，也不要道歉。"
)

FINALIZATION_RETRY_PROMPT = (
    "请根据上面的对话，直接给出回复用户的最终内容。"
)

EMPTY_FINAL_RESPONSE_MESSAGE = (
    "模型多次重试后仍未生成可见的最终文本回复。"
)

MAX_EMPTY_RESPONSE_RETRIES = 2

ACTIVE_CONTINUATION_PROMPT = (
    "请继续完成用户的请求。需要时使用工具，只有在任务真正完成后才停止。"
)


# 运行状态
class RunStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class CancelledError(Exception):
    pass


@dataclass
class RunState:
    status: RunStatus = RunStatus.IDLE
    iteration: int = 0
    error: str | None = None
    cancel_token: CancellationToken = field(default_factory=CancellationToken)
    runtime_context: dict = field(default_factory=dict)
    empty_content_retries: int = 0
    force_no_tools_once: bool = False
    finalization_retrying: bool = False

    @property
    def is_cancelled(self) -> bool:
        return self.status == RunStatus.CANCELLED

    @property
    def is_done(self) -> bool:
        return self.status in (RunStatus.COMPLETED, RunStatus.ERROR, RunStatus.CANCELLED)

    def start(self) -> None:
        self.status = RunStatus.RUNNING
        self.iteration = 0
        self.error = None
        self.cancel_token = CancellationToken()
        self.empty_content_retries = 0
        self.force_no_tools_once = False
        self.finalization_retrying = False

    def cancel(self) -> None:
        if self.status != RunStatus.RUNNING:
            return
        self.status = RunStatus.CANCELLED
        self.cancel_token.cancel("user_cancelled")

    def check_cancel(self) -> None:
        try:
            self.cancel_token.raise_if_cancelled()
        except ToolCancelledError as exc:
            raise CancelledError(str(exc)) from exc


# ReActRunner 主执行器
class ReActRunner:

    def __init__(self, agent: "ReActAgent"):
        self.agent = agent
        self.state = RunState()
        self.llm = LLMHandler(agent.model, agent.api_key, agent.api_base, agent.api_type)
        self.tool_handler = ToolHandler(agent.tools)

    async def run(
        self, message, runtime_context: dict | None = None
    ) -> AsyncGenerator[AgentEvent, None]:
        self.state.start()
        self.state.runtime_context = runtime_context or {}

        if isinstance(message, str):
            self.agent.memory.add_user(message)
        else:
            for msg in message:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    continue
                self.agent.memory.add_raw(msg)

        async for event in self._loop():
            yield event

    def cancel(self) -> None:
        self.state.cancel()
        self.llm.cancel()

    # 外层循环：一次循环代表一次 LLM turn。
    async def _loop(self) -> AsyncGenerator[AgentEvent, None]:
        try:
            while self.agent.max_iterations is None or self.state.iteration < self.agent.max_iterations:
                # 每轮开始前兜底修复历史中的悬空 tool_call。
                self._repair_dangling_tool_calls()
                self.state.check_cancel()
                self.state.iteration += 1

                async for event in self._run_turn():
                    yield event

                # 纯文本 turn 会在 _run_turn 内标记完成；有工具调用时保持 RUNNING。
                if self.state.is_done:
                    return
                # 仍然 RUNNING 说明本轮执行了工具，继续下一轮让模型读取工具结果。

            if not self.state.is_done:
                yield done_event(success=False, reason=DoneReason.MAX_ITERATIONS)
                self.state.status = RunStatus.COMPLETED

        except CancelledError:
            yield done_event(success=False, reason=DoneReason.CANCELLED)

    # 单轮 LLM 调用，带重试。重试使用 for 循环，避免递归放大重试次数。
    async def _run_turn(self) -> AsyncGenerator[AgentEvent, None]:
        """执行一次 provider turn，并对可重试错误自动重试。

        成功时有两种结果：
          - 没有工具调用，状态变成 COMPLETED。
          - 有工具调用，状态保持 RUNNING，外层循环继续下一轮。
        """
        messages = self.agent.memory.get_messages()
        tools = None if self.state.force_no_tools_once else self.agent.tools.to_openai_tools() or None
        self.state.force_no_tools_once = False
        max_attempts = 1 + self.agent.max_retries

        for attempt in range(max_attempts):
            try:
                async for event in self._stream_turn(messages, tools):
                    yield event
                return

            except CancelledError:
                raise

            except Exception as exc:
                if self.state.is_cancelled:
                    raise CancelledError() from exc
                # _stream_turn 抛出的 provider 异常已经是 LLMError；其他异常在此分类。
                err = exc if isinstance(exc, LLMError) else LLMError.classify(exc)
                is_last = attempt >= max_attempts - 1
                logger.warning(
                    "LLM 调用失败 [%s] %s (第 %d/%d 次尝试)",
                    err.code, err.message[:200], attempt + 1, max_attempts,
                )
                if err.code in LLMError.UNRETRYABLE_CODES or is_last:
                    yield error_event(message=err.message, code=err.code)
                    yield done_event(success=False, reason=DoneReason.ERROR)
                    self.state.status = RunStatus.ERROR
                    self.state.error = f"[{err.code}] {err.message}"
                    return

            # 可重试错误：发出 retry 事件，等待一小段时间后进入下一次 attempt。
            yield retry_event(
                code=err.code,
                message=err.message,
                attempt=attempt + 1,
                max_attempts=max_attempts - 1,
            )
            await asyncio.sleep(self.agent.retry_delay)
            if self.state.cancel_token.is_cancelled():
                raise CancelledError()
            # 重试前重新读取 messages，避免使用已经过期的上下文。
            messages = self.agent.memory.get_messages()

    # 单轮流式执行：消费 LLM 事件、并发执行工具、写入 memory。
    async def _stream_turn(
        self,
        messages: list[dict],
        tools: list[dict] | None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """流式消费 LLM 输出，并在收齐工具调用后执行工具。

        文本和 reasoning 会实时 yield。工具结果事件会在所有工具完成且
        memory 写入完整之后再 yield，防止调用方取消时留下不完整历史。

        如果 provider 报错，self.llm.stream 会抛出 LLMError，交给 _run_turn 判断是否重试。
        """
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason: str = "unknown"

        # tool_call_id 到工具任务的映射；保持确定性的写入顺序。
        tool_tasks: dict[str, asyncio.Task] = {}
        tool_calls: list[ToolCall] = []

        # 阶段 1：消费 LLM 流。这里如果发生异常（含 provider LLMError），
        # 必须取消已启动的工具任务后再向外传播。
        try:
            async for event in self.llm.stream(messages, tools):
                self.state.check_cancel()

                if isinstance(event, TextDelta):
                    text_parts.append(event.text)
                    yield assistant_message_event(content=event.text)

                elif isinstance(event, ReasoningDelta):
                    reasoning_parts.append(event.text)
                    yield reasoning_event(content=event.text)

                elif isinstance(event, ToolInputDelta):
                    # 给 UI 实时展示工具参数 JSON 片段。
                    yield tool_call_streaming_event([{
                        "id": event.id,
                        "name": event.name,
                        "arguments_delta": event.text,
                    }])

                elif isinstance(event, ToolCall):
                    tool_calls.append(event)
                    # 立即并发执行，不在这里 await。
                    tool_tasks[event.id] = self.tool_handler.spawn(event, self.state)

                elif isinstance(event, StepFinish):
                    finish_reason = event.finish_reason
                    if event.usage:
                        yield usage_update_event(event.usage)

        except BaseException:
            await self.tool_handler.drain(tool_tasks)
            raise

        # 阶段 2：输出本轮完整文本/reasoning 事件。
        full_text = "".join(text_parts)
        full_reasoning = "".join(reasoning_parts)
        if full_reasoning:
            yield reasoning_complete_event(content=full_reasoning)
        if full_text:
            yield assistant_message_complete_event(content=full_text)
        if finish_reason == "unknown":
            logger.warning(
                "[react_runner] provider 未返回明确 finish_reason，保持 ReAct Loop 继续; "
                "迭代=%s has_text=%s has_reasoning=%s has_tool_calls=%s",
                self.state.iteration,
                bool(full_text.strip()),
                bool(full_reasoning.strip()),
                bool(tool_calls),
            )

        # 阶段 3：没有工具调用的 turn。
        if not tool_calls:
            if full_text.strip():
                self.state.empty_content_retries = 0
                self.state.finalization_retrying = False
                self.agent.memory.add_assistant(full_text, reasoning=full_reasoning or None)
                if finish_reason == "length":
                    yield self._append_internal_user_message(
                        LENGTH_CONTINUATION_PROMPT,
                        reason="length_recovery",
                    )
                    return

                pending_user_messages = await self._drain_pending_user_messages()
                if pending_user_messages:
                    logger.info(
                        "[react_runner] 疑似最终回复后取出 %s 条待注入用户消息",
                        len(pending_user_messages),
                    )
                    for ev in pending_user_messages:
                        self.agent.memory.add_raw(ev.to_openai_message())
                        yield ev
                    return

                if await self._is_continuation_active():
                    logger.info(
                        "[react_runner] 疑似最终回复后检测到活跃 continuation 请求; "
                        "迭代=%s finish_reason=%s",
                        self.state.iteration,
                        finish_reason,
                    )
                    yield self._append_internal_user_message(
                        self._active_continuation_prompt(),
                        reason="active_continuation",
                    )
                    return

                if finish_reason == "unknown":
                    return

                yield done_event(success=True, reason=DoneReason.COMPLETED)
                self.state.status = RunStatus.COMPLETED
                return

            if finish_reason == "length":
                yield self._append_internal_user_message(
                    LENGTH_CONTINUATION_PROMPT,
                    reason="length_recovery",
                )
                return

            pending_user_messages = await self._drain_pending_user_messages()
            if pending_user_messages:
                logger.info(
                    "[react_runner] 空响应后取出 %s 条待注入用户消息",
                    len(pending_user_messages),
                )
                for ev in pending_user_messages:
                    self.agent.memory.add_raw(ev.to_openai_message())
                    yield ev
                return
            if await self._is_continuation_active():
                logger.info(
                    "[react_runner] 空响应后检测到活跃 continuation 请求; 迭代=%s",
                    self.state.iteration,
                )
                yield self._append_internal_user_message(
                    self._active_continuation_prompt(),
                    reason="active_continuation",
                )
                return
            if finish_reason == "unknown":
                return

            if self.state.finalization_retrying:
                yield error_event(message=EMPTY_FINAL_RESPONSE_MESSAGE, code="empty_response")
                yield done_event(success=False, reason=DoneReason.ERROR)
                self.state.status = RunStatus.ERROR
                self.state.error = EMPTY_FINAL_RESPONSE_MESSAGE
                return

            self.state.empty_content_retries += 1
            if self.state.empty_content_retries < MAX_EMPTY_RESPONSE_RETRIES:
                logger.warning(
                    "[react_runner] 空响应重试; 迭代=%s 第 %s/%s 次 finish_reason=%s 有推理=%s",
                    self.state.iteration,
                    self.state.empty_content_retries,
                    MAX_EMPTY_RESPONSE_RETRIES,
                    finish_reason,
                    bool(full_reasoning),
                )
                return

            logger.warning(
                "[react_runner] 空响应重试 %s 次后仍失败，请求最终回复; "
                "迭代=%s finish_reason=%s 有推理=%s",
                self.state.empty_content_retries,
                self.state.iteration,
                finish_reason,
                bool(full_reasoning),
            )
            self.state.finalization_retrying = True
            self.state.force_no_tools_once = True
            yield self._append_internal_user_message(
                FINALIZATION_RETRY_PROMPT,
                reason="empty_finalization_retry",
            )
            return

        # 阶段 4：等待全部工具完成、按顺序归并结果（含取消处理）。
        self.state.empty_content_retries = 0
        self.state.finalization_retrying = False
        results, cancelled = await self.tool_handler.gather_results(
            tool_calls, tool_tasks, self.state
        )

        # 阶段 5：成组写入 memory。这里不能出现 await/yield，否则调用方取消
        # generator 时可能留下只有 assistant tool_calls、没有 tool result 的非法历史。
        self.agent.memory.add_raw(
            self.tool_handler.build_assistant_message(
                tool_calls=tool_calls,
                content=full_text or None,
                reasoning=full_reasoning or None,
            )
        )
        events: list[AgentEvent] = []
        # 工具返回的 UserMessageEvent 需在所有 tool_result 之后追加，
        # 否则会打断 assistant(tool_calls) → tool(...) → tool(...) 顺序，
        # 导致 provider 报 "tool call result does not follow tool call"。
        pending_user_messages: list[AgentEvent] = []

        for tc, result in zip(tool_calls, results):
            # 写入 tool_result 到 memory
            self.agent.memory.add_tool_result(
                tc.id, result.result or f"[{tc.name}] 已完成"
            )
            # yield tool_call + tool_result
            events.append(tool_call_event(id=tc.id, name=tc.name, arguments=tc.input or {}))
            events.append(tool_result_event(
                id=result.call_id, name=result.name, result=result.result or f"[{tc.name}] 已完成",
                error=result.error, status=result.status,
            ))
            # 收集 UserMessageEvent（延后追加）
            if result.event is not None:
                pending_user_messages.append(result.event)

        # 统一追加 UserMessageEvent：确保在 tool_result 之后，消息顺序合法
        for ev in pending_user_messages:
            self.agent.memory.add_raw(ev.to_openai_message())
            events.append(ev)

        for event in events:
            yield event

        if cancelled:
            raise CancelledError()

        # 工具已执行，本轮不标记完成，外层循环会进入下一轮。

    def _append_internal_user_message(self, content: str, *, reason: str) -> AgentEvent:
        ev = user_message_event(
            content,
            metadata={"hide": True, "internal": True, "reason": reason},
        )
        self.agent.memory.add_raw(ev.to_openai_message())
        return ev

    def _active_continuation_prompt(self) -> str:
        ctx = self.state.runtime_context or {}
        return str(
            ctx.get("continuation_message")
            or ctx.get("goal_continue_message")
            or ACTIVE_CONTINUATION_PROMPT
        )

    async def _is_continuation_active(self) -> bool:
        ctx = self.state.runtime_context or {}
        for key in (
            "continuation_active",
            "continue_active",
            "goal_active",
            "sustained_goal_active",
            "goal_active_predicate",
        ):
            if key not in ctx:
                continue
            value = ctx.get(key)
            if callable(value):
                value = value()
            if inspect.isawaitable(value):
                value = await value
            if bool(value):
                return True
        return False

    async def _drain_pending_user_messages(self) -> list[AgentEvent]:
        ctx = self.state.runtime_context or {}
        raw_messages: list = []

        drain = ctx.get("drain_pending_user_messages")
        if callable(drain):
            drained = drain()
            if inspect.isawaitable(drained):
                drained = await drained
            raw_messages.extend(self._coerce_pending_message_list(drained))

        for key in ("pending_user_messages", "injected_user_messages"):
            if key not in ctx:
                continue
            value = ctx.get(key)
            if callable(value):
                value = value()
                if inspect.isawaitable(value):
                    value = await value
                raw_messages.extend(self._coerce_pending_message_list(value))
                continue
            if isinstance(value, list):
                raw_messages.extend(value)
                value.clear()
            else:
                raw_messages.extend(self._coerce_pending_message_list(value))

        events: list[AgentEvent] = []
        for item in raw_messages:
            ev = self._coerce_pending_user_message(item)
            if ev is not None:
                events.append(ev)
        return events

    @staticmethod
    def _coerce_pending_message_list(value) -> list:
        if value is None:
            return []
        if isinstance(value, list):
            return list(value)
        return [value]

    @staticmethod
    def _coerce_pending_user_message(item) -> AgentEvent | None:
        if hasattr(item, "to_openai_message") and getattr(item, "type", None):
            return item
        if isinstance(item, dict):
            if item.get("role") == "user":
                return user_message_event(
                    item.get("content", ""),
                    metadata=item.get("metadata") or {"hide": True, "internal": True},
                )
            if "content" in item:
                return user_message_event(
                    item.get("content", ""),
                    metadata=item.get("metadata") or {"hide": True, "internal": True},
                )
            return None
        if isinstance(item, str):
            return user_message_event(
                item,
                metadata={"hide": True, "internal": True, "reason": "pending_user_message"},
            )
        return None

    # 辅助函数
    def _repair_dangling_tool_calls(self) -> None:
        """为缺少结果的历史 tool_call 补写错误结果。

        这个兜底逻辑在每轮开始前执行，确保发给 provider 的消息历史始终合法。
        """
        messages = self.agent.memory.messages
        existing: set[str] = {
            msg["tool_call_id"]
            for msg in messages
            if msg.get("role") == "tool" and msg.get("tool_call_id")
        }
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls") or []:
                call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if call_id and call_id not in existing:
                    logger.warning(
                        "[repair] dangling tool_call id=%s，补写 error result", call_id
                    )
                    self.agent.memory.add_tool_result(
                        call_id,
                        "[INTERRUPTED] Tool execution was interrupted. Please retry.",
                    )
                    existing.add(call_id)
