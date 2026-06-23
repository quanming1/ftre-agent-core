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
from ftre_agent_core.tracing import RunStatus as TraceRunStatus, RunType, TraceSpan
from .tool_handler import ToolHandler
from ..event import (
    DoneReason,
    AgentEvent,
    EventType,
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

# ── 内部提示词常量 ────────────────────────────────────────────────────────────

# 当 finish_reason == "length"（输出被截断）时，注入这条消息让模型继续输出。
LENGTH_CONTINUATION_PROMPT = (
    "输出长度达到上限。请从刚才中断的位置继续，不要重述已经说过的内容，也不要道歉。"
)

# 当模型多次返回空内容后，注入这条消息强制要求模型给出最终回复。
FINALIZATION_RETRY_PROMPT = (
    "请根据上面的对话，直接给出回复用户的最终内容。"
)

# 空响应重试耗尽后的错误提示。
EMPTY_FINAL_RESPONSE_MESSAGE = (
    "模型多次重试后仍未生成可见的最终文本回复。"
)

# 空响应最多重试次数（不含强制最终化那一次）。
MAX_EMPTY_RESPONSE_RETRIES = 2

# 主动续跑提示词：当 runtime_context 标记了 continuation_active 时注入。
ACTIVE_CONTINUATION_PROMPT = (
    "请继续完成用户的请求。需要时使用工具，只有在任务真正完成后才停止。"
)


# ── 运行状态 ──────────────────────────────────────────────────────────────────

class RunStatus(str, Enum):
    """单次 run() 调用的生命周期状态。"""
    IDLE = "idle"          # 尚未启动
    RUNNING = "running"    # 正在执行 ReAct 循环
    COMPLETED = "completed"  # 正常结束（含 max_iterations 到顶）
    ERROR = "error"        # 因不可重试错误终止
    CANCELLED = "cancelled"  # 被用户取消


class CancelledError(Exception):
    """内部取消异常，与 asyncio.CancelledError 区分。"""
    pass


@dataclass
class RunState:
    """一次 run() 执行期间的可变状态。

    所有字段在 start() 时重置，在 _loop / _run_turn / _stream_turn 中读写。
    """
    status: RunStatus = RunStatus.IDLE
    iteration: int = 0                              # 当前迭代轮次（从 1 开始）
    error: str | None = None                        # 最终错误描述（ERROR 状态时填充）
    cancel_token: CancellationToken = field(default_factory=CancellationToken)
    runtime_context: dict = field(default_factory=dict)  # 调用方传入的上下文
    empty_content_retries: int = 0                  # 连续空响应计数
    force_no_tools_once: bool = False               # 下一轮强制不带 tools（用于空响应最终化）
    finalization_retrying: bool = False             # 是否已进入"强制最终化"阶段
    trace_span: TraceSpan | None = None             # 本次执行的根 trace span

    @property
    def is_cancelled(self) -> bool:
        return self.status == RunStatus.CANCELLED

    @property
    def is_done(self) -> bool:
        """是否处于终态（无论成功/失败/取消）。"""
        return self.status in (RunStatus.COMPLETED, RunStatus.ERROR, RunStatus.CANCELLED)

    def start(self) -> None:
        """重置全部字段，开始新一轮执行。"""
        self.status = RunStatus.RUNNING
        self.iteration = 0
        self.error = None
        self.cancel_token = CancellationToken()
        self.empty_content_retries = 0
        self.force_no_tools_once = False
        self.finalization_retrying = False
        self.trace_span = None

    def cancel(self) -> None:
        """标记为取消，同时触发 cancel_token 通知正在执行的工具。"""
        if self.status != RunStatus.RUNNING:
            return
        self.status = RunStatus.CANCELLED
        self.cancel_token.cancel("user_cancelled")

    def check_cancel(self) -> None:
        """检查是否被取消，是则抛出 CancelledError。"""
        try:
            self.cancel_token.raise_if_cancelled()
        except ToolCancelledError as exc:
            raise CancelledError(str(exc)) from exc


# ── ReActRunner 主执行器 ──────────────────────────────────────────────────────

class ReActRunner:
    """ReAct Agent 的核心执行引擎。

    职责：
      - 驱动 Reason → Act → Observe 循环
      - 管理 LLM 调用（含重试）
      - 并发执行工具并归并结果
      - 维护 memory（对话历史）的合法性
      - 向外 yield AgentEvent 供 UI/调用方消费
    """

    def __init__(self, agent: "ReActAgent"):
        self.agent = agent
        self.state = RunState()
        # LLMHandler 封装了 provider 调用细节（OpenAI SDK、流式解析、reasoning 提取等）。
        # max_tokens 来自 agent 配置（config.json 的 max_output），None 表示不传该参数。
        self.llm = LLMHandler(
            agent.model,
            agent.api_key,
            agent.api_base,
            agent.api_type,
            max_tokens=agent.max_tokens,
        )
        # ToolHandler 负责工具的并发调度、取消传播和结果归并。
        self.tool_handler = ToolHandler(agent.tools)

    # ── 入口：run() ────────────────────────────────────────────────────────

    async def run(
        self, message, runtime_context: dict | None = None
    ) -> AsyncGenerator[AgentEvent, None]:
        """启动一次完整的 ReAct 执行。

        Args:
            message: 用户消息。可以是字符串（单条）或消息列表（多条）。
            runtime_context: 调用方传入的上下文，可包含：
                - trace_metadata / trace_tags：tracing 元数据
                - trace_name：span 名称
                - continuation_active / pending_user_messages 等控制字段

        Yields:
            AgentEvent：包括文本增量、推理增量、工具调用/结果、done 等。
        """
        self.state.start()
        self.state.runtime_context = runtime_context or {}

        # ── 准备 tracing 元数据 ──
        # 调用方可以通过 runtime_context 传入自定义 trace 元数据，
        # 这里兜底处理：非 dict 自动包一层，然后强制带上 model / api_type。
        trace_metadata = self.state.runtime_context.get("trace_metadata") or {}
        if not isinstance(trace_metadata, dict):
            trace_metadata = {"value": trace_metadata}
        trace_metadata = {
            "model": self.agent.model,
            "api_type": self.agent.api_type,
            **trace_metadata,  # 调用方自定义字段可覆盖上面的默认值
        }
        # tags 允许传单个字符串，统一规整成 list。
        trace_tags = self.state.runtime_context.get("trace_tags") or []
        if isinstance(trace_tags, str):
            trace_tags = [trace_tags]

        # 开启根 span（AGENT 节点）；tracer 未配置 exporter 时为空操作。
        self.state.trace_span = self.agent.tracer.start_run(
            str(self.state.runtime_context.get("trace_name") or "react_agent"),
            RunType.AGENT,
            inputs={"message": message},
            metadata=trace_metadata,
            tags=list(trace_tags),
        )

        # 这两个变量在 finally 中用于写入根 span 的输出。
        done_reason = None
        done_success = None

        # ── 将用户消息写入 memory ──
        if isinstance(message, str):
            self.agent.memory.add_user(message)
        else:
            # 列表形式：跳过 system 消息（system_prompt 由 memory 单独管理），其余原样写入。
            for msg in message:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    continue
                self.agent.memory.add_raw(msg)

        # ── 主循环 + 异常/收尾处理 ──
        try:
            async for event in self._loop():
                # 捕获 DONE 事件的成败与原因，留待 finally 写入根 span。
                if event.type == EventType.DONE:
                    done_reason = event.reason.value
                    done_success = event.success
                yield event

        except BaseException as exc:
            # 异常路径：区分「取消」与「真实错误」，分别落不同的 trace 状态。
            if self.state.trace_span and not self.state.trace_span.ended:
                if isinstance(exc, (asyncio.CancelledError, GeneratorExit)):
                    self.state.trace_span.end(
                        status=TraceRunStatus.CANCELLED,
                        outputs={"iterations": self.state.iteration},
                    )
                else:
                    self.state.trace_span.end(
                        error=exc,
                        outputs={"iterations": self.state.iteration},
                    )
            raise

        finally:
            # 正常路径收尾：依据 RunState 推断最终 trace 状态并关闭根 span。
            # （异常路径已在 except 中提前 end，这里因 ended 判断而跳过。）
            if self.state.trace_span and not self.state.trace_span.ended:
                trace_status = (
                    TraceRunStatus.CANCELLED
                    if self.state.status == RunStatus.CANCELLED
                    else TraceRunStatus.ERROR
                    if self.state.status == RunStatus.ERROR
                    else TraceRunStatus.COMPLETED
                )
                self.state.trace_span.end(
                    status=trace_status,
                    outputs={
                        "success": done_success,
                        "done_reason": done_reason,
                        "iterations": self.state.iteration,
                    },
                    error=self.state.error if self.state.status == RunStatus.ERROR else None,
                )

    def cancel(self) -> None:
        """外部调用：取消当前执行。同时通知 LLM stream 和工具任务。"""
        self.state.cancel()
        self.llm.cancel()

    # ── 外层循环：_loop() ──────────────────────────────────────────────────

    async def _loop(self) -> AsyncGenerator[AgentEvent, None]:
        """ReAct 主循环：Reason → Act → Observe。

        每次循环代表一次完整的 LLM turn：
          1. _run_turn() 调用 LLM，消费流式输出
          2. 如果 LLM 返回了纯文本（无工具调用），循环结束
          3. 如果 LLM 返回了工具调用，执行工具后继续下一轮
        """
        try:
            while self.agent.max_iterations is None or self.state.iteration < self.agent.max_iterations:
                self.state.check_cancel()
                self.state.iteration += 1

                async for event in self._run_turn():
                    yield event

                # 纯文本 turn 会在 _run_turn 内标记 COMPLETED；有工具调用时保持 RUNNING。
                if self.state.is_done:
                    return
                # 仍然 RUNNING 说明本轮执行了工具，继续下一轮让模型读取工具结果。

            # 达到 max_iterations 上限，标记为完成（非错误）。
            if not self.state.is_done:
                yield done_event(success=False, reason=DoneReason.MAX_ITERATIONS)
                self.state.status = RunStatus.COMPLETED

        except CancelledError:
            yield done_event(success=False, reason=DoneReason.CANCELLED)
            self.state.status = RunStatus.CANCELLED

    # ── 单轮调用（带重试）：_run_turn() ────────────────────────────────────

    async def _run_turn(self) -> AsyncGenerator[AgentEvent, None]:
        """执行一次 provider turn，并对可重试错误自动重试。

        成功时有两种结果：
          - 没有工具调用 → 状态变成 COMPLETED，外层循环结束
          - 有工具调用 → 状态保持 RUNNING，外层循环继续下一轮

        重试使用 for 循环（非递归），避免递归放大重试次数。
        重试策略：
          - 不可重试错误（如 400 bad_request）→ 立即终止
          - 可重试错误（如 429 rate_limit）→ 等待 retry_delay 后重试
          - 达到 max_attempts → 终止
        """
        messages = self.agent.memory.get_messages()
        # force_no_tools_once 用于空响应最终化：强制模型不带工具，只输出文本。
        tools = None if self.state.force_no_tools_once else self.agent.tools.to_openai_tools() or None
        self.state.force_no_tools_once = False
        max_attempts = 1 + self.agent.max_retries  # 首次 + 重试次数

        for attempt in range(max_attempts):
            # 每次尝试创建一个 LLM 子 span，挂在根 AGENT span 下。
            llm_span = self.state.trace_span.child(
                "llm",
                RunType.LLM,
                inputs={"messages": messages, "tools": tools},
                metadata={
                    "model": self.agent.model,
                    "api_type": self.agent.api_type,
                    "iteration": self.state.iteration,
                    "attempt": attempt + 1,
                },
            ) if self.state.trace_span else None

            try:
                async for event in self._stream_turn(messages, tools, llm_span=llm_span):
                    yield event
                return  # 成功完成，退出重试循环

            except CancelledError:
                # 取消：标记 LLM span 为 CANCELLED 后继续向上传播。
                if llm_span and not llm_span.ended:
                    llm_span.end(status=TraceRunStatus.CANCELLED)
                raise

            except Exception as exc:
                # 出错：先记录到 LLM span（正常完成路径已在 _stream_turn 内 end）。
                if llm_span and not llm_span.ended:
                    llm_span.end(error=exc)
                if self.state.is_cancelled:
                    raise CancelledError() from exc

                # _stream_turn 抛出的 provider 异常已经是 LLMError；其他异常在此分类。
                err = exc if isinstance(exc, LLMError) else LLMError.classify(exc)
                is_last = attempt >= max_attempts - 1
                logger.warning(
                    "LLM 调用失败 [%s] %s (第 %d/%d 次尝试)",
                    err.code, err.message[:200], attempt + 1, max_attempts,
                )

                # 不可重试或已是最后一次尝试 → 终止。
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

    # ── 流式执行：_stream_turn() ───────────────────────────────────────────

    async def _stream_turn(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        llm_span: TraceSpan | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        """流式消费 LLM 输出，并在收齐工具调用后执行工具。

        执行分为 5 个阶段：
          1. 消费 LLM 流（文本/reasoning/工具调用增量实时 yield）
          2. 输出本轮完整文本/reasoning 事件
          3. 如果没有工具调用 → 处理纯文本 turn（含空响应/截断/续跑等分支）
          4. 等待全部工具完成，归并结果
          5. 成组写入 memory（assistant tool_calls + tool results），然后 yield 事件

        关键设计：阶段 5 中不能出现 await/yield，否则调用方取消 generator 时
        可能留下只有 assistant tool_calls、没有 tool result 的非法历史。
        """
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason: str = "unknown"
        usage: dict | None = None
        response_metadata: dict = {}

        # tool_call_id → 工具任务的映射；保持确定性的写入顺序。
        tool_tasks: dict[str, asyncio.Task] = {}
        tool_calls: list[ToolCall] = []

        # ── 阶段 1：消费 LLM 流 ──
        # 这里如果发生异常（含 provider LLMError），
        # 必须先取消已启动的工具任务再向外传播。
        try:
            async for event in self.llm.stream(messages, tools):
                self.state.check_cancel()

                if isinstance(event, TextDelta):
                    # 文本增量：实时 yield 给 UI 展示。
                    text_parts.append(event.text)
                    yield assistant_message_event(content=event.text)

                elif isinstance(event, ReasoningDelta):
                    # 推理增量（thinking model 的 chain-of-thought）：实时 yield。
                    reasoning_parts.append(event.text)
                    yield reasoning_event(content=event.text)

                elif isinstance(event, ToolInputDelta):
                    # 工具参数 JSON 片段：给 UI 实时展示工具调用正在输入。
                    yield tool_call_streaming_event([{
                        "id": event.id,
                        "name": event.name,
                        "arguments_delta": event.text,
                    }])

                elif isinstance(event, ToolCall):
                    # 完整的工具调用：立即 spawn 异步任务，不在这里 await。
                    # 这样多个工具调用可以并发执行。
                    tool_calls.append(event)
                    tool_tasks[event.id] = self.tool_handler.spawn(
                        event,
                        self.state,
                        parent_span=self.state.trace_span,
                    )

                elif isinstance(event, StepFinish):
                    # LLM 流结束：拿到 finish_reason 和 usage。
                    finish_reason = event.finish_reason
                    response_metadata = event.response_metadata
                    if event.usage:
                        usage = event.usage
                        yield usage_update_event(event.usage)

        except BaseException:
            # 异常时必须清理已启动的工具任务，防止泄露。
            await self.tool_handler.drain(tool_tasks)
            raise

        # ── 阶段 2：输出完整文本/reasoning 事件 ──
        full_text = "".join(text_parts)
        full_reasoning = "".join(reasoning_parts)
        if full_reasoning:
            yield reasoning_complete_event(content=full_reasoning)
        if full_text:
            yield assistant_message_complete_event(content=full_text)

        # 关闭 LLM span（记录本轮完整输出）。
        if llm_span and not llm_span.ended:
            llm_span.end(outputs={
                "text": full_text,
                "reasoning": full_reasoning,
                "finish_reason": finish_reason,
                "has_tool_calls": bool(tool_calls),
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": call.input}
                    for call in tool_calls
                ],
                "usage": usage,
                "response_metadata": response_metadata,
            })

        # provider 未返回明确 finish_reason 时的告警。
        if finish_reason == "unknown":
            logger.warning(
                "[react_runner] provider 未返回明确 finish_reason，保持 ReAct Loop 继续; "
                "迭代=%s has_text=%s has_reasoning=%s has_tool_calls=%s",
                self.state.iteration,
                bool(full_text.strip()),
                bool(full_reasoning.strip()),
                bool(tool_calls),
            )

        # ── 阶段 3：没有工具调用的 turn ──
        if not tool_calls:
            if full_text.strip():
                # 正常有文本输出：重置空响应计数器，写入 memory。
                self.state.empty_content_retries = 0
                self.state.finalization_retrying = False
                self.agent.memory.add_assistant(full_text, reasoning=full_reasoning or None)

                # 分支 3a：输出被截断（达到 max_tokens）→ 注入续写提示。
                if finish_reason == "length":
                    yield self._append_internal_user_message(
                        LENGTH_CONTINUATION_PROMPT,
                        reason="length_recovery",
                    )
                    return

                # 分支 3b：检查是否有调用方注入的 pending user messages。
                # 这些消息在模型给出最终回复后才追加到 memory，避免打断 ReAct 流程。
                pending_user_messages = await self._drain_pending_user_messages()
                if pending_user_messages:
                    logger.info(
                        "[react_runner] 疑似最终回复后取出 %s 条待注入用户消息",
                        len(pending_user_messages),
                    )
                    for ev in pending_user_messages:
                        self.agent.memory.add_raw(ev.to_openai_message())
                        yield ev
                    # 有注入消息 → 不结束，让模型处理新消息。
                    return

                # 分支 3d：检查是否需要主动续跑。
                if await self._is_continuation_active():
                    logger.info(
                        "[react_runner] 主动续跑模式激活，注入续跑提示; 迭代=%s",
                        self.state.iteration,
                    )
                    yield self._append_internal_user_message(
                        self._active_continuation_prompt(),
                        reason="active_continuation",
                    )
                    return

                # 分支 3e：正常结束。
                yield done_event(success=True, reason=DoneReason.COMPLETED)
                self.state.status = RunStatus.COMPLETED
                return

            # ── 以下处理空文本（full_text 为空）的情况 ──

            # 只有 reasoning 没有正文：模型还在思考，继续下一轮。
            if full_reasoning.strip():
                self.agent.memory.add_assistant("", reasoning=full_reasoning)
                return

            # finish_reason=length 但文本为空：可能是 max_tokens 设太小，
            # provider 返回了空 content。直接结束，避免无限循环。
            if finish_reason == "length":
                return
            # finish_reason=unknown：无法判断，直接结束。
            if finish_reason == "unknown":
                return

            # 已经在"强制最终化"阶段还是空 → 彻底失败。
            if self.state.finalization_retrying:
                yield error_event(message=EMPTY_FINAL_RESPONSE_MESSAGE, code="empty_response")
                yield done_event(success=False, reason=DoneReason.ERROR)
                self.state.status = RunStatus.ERROR
                self.state.error = EMPTY_FINAL_RESPONSE_MESSAGE
                return

            # 空响应重试：计数 +1，未达上限则继续下一轮（让模型重新生成）。
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

            # 重试耗尽 → 进入"强制最终化"：下一轮强制不带工具，只输出文本。
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

        # ── 阶段 4：等待全部工具完成 ──
        # gather_results 会并发等待所有工具任务，处理取消，返回按调用顺序排列的结果。
        self.state.empty_content_retries = 0
        self.state.finalization_retrying = False
        results, cancelled = await self.tool_handler.gather_results(
            tool_calls, tool_tasks, self.state
        )

        # ── 阶段 5：成组写入 memory ──
        # ⚠️ 这里不能出现 await/yield！否则调用方取消 generator 时
        # 可能留下只有 assistant tool_calls、没有 tool result 的非法历史。
        #
        # 写入顺序必须是：
        #   assistant(tool_calls) → tool(result_1) → tool(result_2) → ...
        # 这样 provider 才能正确匹配 tool_call 和 tool_result。
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
            # 收集 UserMessageEvent（延后追加）
            if result.event is not None:
                pending_user_messages.append(result.event)

        # 构建向外 yield 的事件列表（tool_call → tool_result → pending_user_messages）
        for tc in tool_calls:
            events.append(tool_call_event(id=tc.id, name=tc.name, arguments=tc.input or {}))

        for tc, result in zip(tool_calls, results):
            events.append(tool_result_event(
                id=result.call_id, name=result.name, result=result.result or f"[{tc.name}] 已完成",
                error=result.error, status=result.status,
            ))

        # 统一追加 UserMessageEvent：确保在 tool_result 之后，消息顺序合法
        for ev in pending_user_messages:
            self.agent.memory.add_raw(ev.to_openai_message())
            events.append(ev)

        for event in events:
            yield event

        if cancelled:
            raise CancelledError()

        # 工具已执行，本轮不标记完成，外层循环会进入下一轮让模型读取工具结果。

    # ── 辅助方法 ────────────────────────────────────────────────────────────

    def _append_internal_user_message(self, content: str, *, reason: str) -> AgentEvent:
        """向 memory 追加一条内部用户消息（对用户不可见），并返回对应事件。

        用于：
          - length 截断后的续写提示
          - 空响应最终化提示
          - 主动续跑提示
        """
        ev = user_message_event(
            content,
            metadata={"hide": True, "internal": True, "reason": reason},
        )
        self.agent.memory.add_raw(ev.to_openai_message())
        return ev

    def _active_continuation_prompt(self) -> str:
        """获取主动续跑提示词。优先用 runtime_context 中自定义的，兜底用默认值。"""
        ctx = self.state.runtime_context or {}
        return str(
            ctx.get("continuation_message")
            or ctx.get("goal_continue_message")
            or ACTIVE_CONTINUATION_PROMPT
        )

    async def _is_continuation_active(self) -> bool:
        """检查是否需要主动续跑。

        检查 runtime_context 中的多个 key（兼容不同调用方的命名）：
          - continuation_active / continue_active
          - goal_active / sustained_goal_active
          - goal_active_predicate
        值可以是 bool、callable（同步/异步）。
        """
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
        """从 runtime_context 中取出调用方注入的 pending user messages。

        支持两种来源：
          1. drain_pending_user_messages：callable，返回消息列表
          2. pending_user_messages / injected_user_messages：list（取出后清空）

        这些消息通常用于在模型给出最终回复后，追加新的用户指令而不打断 ReAct 流程。
        """
        ctx = self.state.runtime_context or {}
        raw_messages: list = []

        # 来源 1：callable drain 函数
        drain = ctx.get("drain_pending_user_messages")
        if callable(drain):
            drained = drain()
            if inspect.isawaitable(drained):
                drained = await drained
            raw_messages.extend(self._coerce_pending_message_list(drained))

        # 来源 2：list / 单值
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
                value.clear()  # 取出后清空，避免重复消费
            else:
                raw_messages.extend(self._coerce_pending_message_list(value))

        # 将原始消息统一转换为 AgentEvent
        events: list[AgentEvent] = []
        for item in raw_messages:
            ev = self._coerce_pending_user_message(item)
            if ev is not None:
                events.append(ev)
        return events

    @staticmethod
    def _coerce_pending_message_list(value) -> list:
        """将任意值规整为 list。None → []，非 list → [value]。"""
        if value is None:
            return []
        if isinstance(value, list):
            return list(value)
        return [value]

    @staticmethod
    def _coerce_pending_user_message(item) -> AgentEvent | None:
        """将单条原始消息转换为 UserMessageEvent。

        支持的输入格式：
          - 已有 to_openai_message 方法的 AgentEvent（直接返回）
          - dict: {"role": "user", "content": ...} 或 {"content": ...}
          - str: 直接作为 content
        """
        # 已是 AgentEvent
        if hasattr(item, "to_openai_message") and getattr(item, "type", None):
            return item
        # dict 格式
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
        # 字符串格式
        if isinstance(item, str):
            return user_message_event(
                item,
                metadata={"hide": True, "internal": True, "reason": "pending_user_message"},
            )
        return None
