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
import logging
import time
import uuid
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
from ...event import (
    ReplyFinishedReason,
    AgentStreamEvent,
    ReplyStartEvent, ReplyEndEvent,
    ModelCallStartEvent, ModelCallEndEvent,
    TextBlockStartEvent, TextBlockDeltaEvent, TextBlockEndEvent,
    ThinkingBlockStartEvent, ThinkingBlockDeltaEvent, ThinkingBlockEndEvent,
    HintBlockEvent,
    ToolCallStartEvent, ToolCallDeltaEvent, ToolCallEndEvent,
    ToolResultStartEvent, ToolResultTextDeltaEvent, ToolResultEndEvent,
    RetryEvent,
)
from ...message import Msg, AssistantMsg, ToolResultState
from ...hooks import (
    ON_TURN_START,
    ON_STOP,
    ON_TURN_END,
    TurnStartInput,
    TurnStartOutput,
    StopInput,
    TurnEndInput,
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
    error_code: str | None = None                   # 错误代码（ERROR 状态时填充）
    done_reason: ReplyFinishedReason | None = None     # Turn 结束原因（COMPLETED/EXCEED_MAX_ITERS/ERROR/INTERRUPTED）
    cancel_token: CancellationToken = field(default_factory=CancellationToken)
    runtime_context: dict = field(default_factory=dict)  # 调用方传入的上下文
    empty_content_retries: int = 0                  # 连续空响应计数
    force_no_tools_once: bool = False               # 下一轮强制不带 tools（用于空响应最终化）
    finalization_retrying: bool = False             # 是否已进入"强制最终化"阶段
    trace_span: TraceSpan | None = None             # 本次执行的根 trace span
    reply_msg: Msg | None = None                    # 当前 reply 重建的 Msg（由 append_event 累积）
    reply_id: str = ""                              # 当前 reply 的 id
    turn_id: str = ""                               # 本次 Turn 的唯一标识
    _turn_start_ts: float = 0.0                     # 当前轮开始时间（perf_counter）
    _first_token_logged: bool = False               # 当前轮是否已记录首个字符
    _ttft_ms: float | None = None                   # 当前 LLM call 的 TTFT（毫秒）
    token_usage: dict = field(default_factory=lambda: {  # 本 Turn 累积的 token 用量
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "llm_calls": 0,
    })

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
        self.error_code = None
        self.done_reason = None
        self.cancel_token = CancellationToken()
        self.empty_content_retries = 0
        self.force_no_tools_once = False
        self.finalization_retrying = False
        self.trace_span = None
        self.reply_msg = None
        self.reply_id = ""
        self.turn_id = self.runtime_context.get("turn_id") or f"turn_{uuid.uuid4().hex[:12]}"
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "llm_calls": 0,
        }

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
      - 向外 yield AgentStreamEvent 供 UI/调用方消费
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
            reasoning_effort=agent.reasoning_effort,
        )
        # ToolHandler 负责工具的并发调度、取消传播和结果归并。
        self.tool_handler = ToolHandler(agent.tool_registry, agent.hook_manager)

    # ── 入口：run() ────────────────────────────────────────────────────────

    async def run(
        self, message, runtime_context: dict | None = None
    ) -> AsyncGenerator[AgentStreamEvent, None]:
        """启动一次完整的 ReAct 执行。

        Args:
            message: 用户消息。可以是字符串（单条）或消息列表（多条）。
            runtime_context: 调用方传入的上下文，可包含：
                - trace_metadata / trace_tags：tracing 元数据
                - trace_name：span 名称

        Yields:
            AgentStreamEvent：包括文本增量、推理增量、工具调用/结果、step 等。
        """
        self.state.runtime_context = runtime_context or {}
        self.state.start()

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

        # ── 将用户消息写入 memory ──
        if isinstance(message, str):
            self.agent.memory.add_user(message)
        else:
            # 列表形式：原样写入 memory。
            # system 消息（如插件通过 BEFORE_AGENT_RUN hook 注入的 MCP/Skill 提示）
            # 也会被写入 _messages，最终在 get_messages() 中出现在 memory.system_prompt 之后。
            for msg in message:
                self.agent.memory.add_raw(msg)

        # ── 主循环 + 异常/收尾处理 ──
        try:
            async for event in self._loop():
                # 给每个事件盖 turn_id 戳
                object.__setattr__(event, "turn_id", self.state.turn_id)
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
                        "success": self.state.status == RunStatus.COMPLETED,
                        "done_reason": self.state.done_reason,
                        "iterations": self.state.iteration,
                    },
                    error=self.state.error if self.state.status == RunStatus.ERROR else None,
                )

    def cancel(self) -> None:
        """外部调用：取消当前执行。同时通知 LLM stream 和工具任务。"""
        self.state.cancel()
        self.llm.cancel()

    # ── 外层循环：_loop() ──────────────────────────────────────────────────

    async def _loop(self) -> AsyncGenerator[AgentStreamEvent, None]:
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
                self.state._turn_start_ts = time.perf_counter()
                self.state._first_token_logged = False
                self.state._ttft_ms = None

                # ── on_turn_start hook ──
                ts_output = await self.agent.hook_manager.trigger(
                    ON_TURN_START,
                    lambda: TurnStartInput(
                        session_id=self.state.runtime_context.get("session_id", ""),
                        turn_id=self.state.turn_id,
                        iteration=self.state.iteration,
                        messages=self.agent.memory.get_messages(),
                        runtime_context=self.state.runtime_context,
                    ),
                )
                if ts_output is not None and isinstance(ts_output, TurnStartOutput):
                    for msg in ts_output.inject_messages:
                        self.agent.memory.add_raw(msg)

                async for event in self._run_turn():
                    yield event

                # 纯文本 turn 会在 _run_turn 内标记 COMPLETED；有工具调用时保持 RUNNING。
                if self.state.is_done:
                    # ── on_turn_end hook（只读观察）──
                    await self.agent.hook_manager.trigger(
                        ON_TURN_END,
                        lambda: TurnEndInput(
                            session_id=self.state.runtime_context.get("session_id", ""),
                            turn_id=self.state.turn_id,
                            iteration=self.state.iteration,
                            done_reason=self.state.done_reason or ReplyFinishedReason.COMPLETED,
                            runtime_context=self.state.runtime_context,
                        ),
                    )
                    return
                # 仍然 RUNNING 说明本轮执行了工具，继续下一轮让模型读取工具结果。

            # 达到 max_iterations 上限，标记为完成（非错误）。
            if not self.state.is_done:
                self.state.done_reason = ReplyFinishedReason.EXCEED_MAX_ITERS
                self.state.status = RunStatus.COMPLETED

        except CancelledError:
            self.state.done_reason = ReplyFinishedReason.INTERRUPTED
            self.state.status = RunStatus.CANCELLED

    # ── 单轮调用（带重试）：_run_turn() ────────────────────────────────────

    async def _run_turn(self) -> AsyncGenerator[AgentStreamEvent, None]:
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
        tools = None if self.state.force_no_tools_once else self.agent.tool_registry.to_openai_tools() or None
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
                    self.state.done_reason = ReplyFinishedReason.ERROR
                    self.state.status = RunStatus.ERROR
                    self.state.error = f"[{err.code}] {err.message}"
                    self.state.error_code = err.code
                    return

            # 可重试错误：发出 retry 事件，等待一小段时间后进入下一次 attempt。
            yield RetryEvent(
                reply_id=self.state.reply_id,
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
    ) -> AsyncGenerator[AgentStreamEvent, None]:
        """流式消费 LLM 输出，并在收齐工具调用后执行工具。

        执行分为 6 个阶段：
          0. 初始化 reply（reply_id / msg / ReplyStart / ModelCallStart）
          1. 消费 LLM 流（文本/reasoning/工具调用增量实时 yield 三段式事件）
          2. 收尾 open blocks + 关闭 LLM span
          3. 如果没有工具调用 → 处理纯文本 turn（含空响应/截断/续跑等分支）
          4. 等待全部工具完成，归并结果
          5. 成组写入 memory（assistant tool_calls + tool results），yield 事件 + ReplyEnd

        关键设计：阶段 5 中 memory 写入必须先于 yield，否则调用方取消 generator 时
        可能留下只有 assistant tool_calls、没有 tool result 的非法历史。
        """
        # ── 阶段 0：初始化 reply ──
        reply_id = uuid.uuid4().hex[:16]
        self.state.reply_id = reply_id
        session_id = self.state.runtime_context.get("session_id", "")
        model_name = self.agent.model
        msg = AssistantMsg(name=model_name, content=[], id=reply_id)

        rs = ReplyStartEvent(session_id=session_id, reply_id=reply_id, name=model_name)
        yield rs; msg.append_event(rs)
        mcs = ModelCallStartEvent(reply_id=reply_id, model_name=model_name)
        yield mcs; msg.append_event(mcs)

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason: str = "unknown"
        usage: dict | None = None
        response_metadata: dict = {}

        # tool_call_id → 工具任务的映射；保持确定性的写入顺序。
        tool_tasks: dict[str, asyncio.Task] = {}
        tool_calls: list[ToolCall] = []

        # 流式状态变量
        text_block_id: str | None = None
        thinking_block_id: str | None = None
        tool_call_started: set[str] = set()

        # ── 阶段 1：消费 LLM 流 ──
        # 这里如果发生异常（含 provider LLMError），
        # 必须先发 ReplyEnd(INTERRUPTED) 再取消已启动的工具任务，最后向上传播。
        try:
            async for event in self.llm.stream(messages, tools):
                self.state.check_cancel()

                if not self.state._first_token_logged:
                    self.state._first_token_logged = True
                    elapsed_ms = (time.perf_counter() - self.state._turn_start_ts) * 1000
                    logger.info(f"[react] 第 {self.state.iteration} 轮 TTFT {elapsed_ms:.0f}ms")
                    if llm_span and not llm_span.ended:
                        llm_span.add_event("ttft", {"ms": round(elapsed_ms)})

                if isinstance(event, TextDelta):
                    text_parts.append(event.text)
                    if text_block_id is None:
                        text_block_id = uuid.uuid4().hex[:16]
                        tbs = TextBlockStartEvent(reply_id=reply_id, block_id=text_block_id)
                        yield tbs; msg.append_event(tbs)
                    tbd = TextBlockDeltaEvent(reply_id=reply_id, block_id=text_block_id, delta=event.text)
                    yield tbd; msg.append_event(tbd)

                elif isinstance(event, ReasoningDelta):
                    reasoning_parts.append(event.text)
                    if thinking_block_id is None:
                        thinking_block_id = uuid.uuid4().hex[:16]
                        thbs = ThinkingBlockStartEvent(reply_id=reply_id, block_id=thinking_block_id)
                        yield thbs; msg.append_event(thbs)
                    thbd = ThinkingBlockDeltaEvent(reply_id=reply_id, block_id=thinking_block_id, delta=event.text)
                    yield thbd; msg.append_event(thbd)

                elif isinstance(event, ToolInputDelta):
                    if event.id not in tool_call_started:
                        tool_call_started.add(event.id)
                        tcs = ToolCallStartEvent(reply_id=reply_id, tool_call_id=event.id, tool_call_name=event.name or "")
                        yield tcs; msg.append_event(tcs)
                    tcd = ToolCallDeltaEvent(reply_id=reply_id, tool_call_id=event.id, delta=event.text)
                    yield tcd; msg.append_event(tcd)

                elif isinstance(event, ToolCall):
                    if event.id not in tool_call_started:
                        tool_call_started.add(event.id)
                        tcs = ToolCallStartEvent(reply_id=reply_id, tool_call_id=event.id, tool_call_name=event.name)
                        yield tcs; msg.append_event(tcs)
                    tce = ToolCallEndEvent(reply_id=reply_id, tool_call_id=event.id)
                    yield tce; msg.append_event(tce)

                    tool_calls.append(event)
                    tool_tasks[event.id] = self.tool_handler.spawn(
                        event,
                        self.state,
                        parent_span=self.state.trace_span,
                    )

                elif isinstance(event, StepFinish):
                    finish_reason = event.finish_reason
                    response_metadata = event.response_metadata
                    if event.usage:
                        usage = event.usage
                        # 累积到 turn 级统计
                        self.state.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                        self.state.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                        self.state.token_usage["cached_tokens"] += usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                        self.state.token_usage["llm_calls"] += 1
                    # 收尾 open blocks
                    if text_block_id is not None:
                        tbe = TextBlockEndEvent(reply_id=reply_id, block_id=text_block_id)
                        yield tbe; msg.append_event(tbe)
                        text_block_id = None
                    if thinking_block_id is not None:
                        thbe = ThinkingBlockEndEvent(reply_id=reply_id, block_id=thinking_block_id)
                        yield thbe; msg.append_event(thbe)
                        thinking_block_id = None
                    # ModelCallEnd
                    input_tokens = usage.get("prompt_tokens", 0) if usage else 0
                    output_tokens = usage.get("completion_tokens", 0) if usage else 0
                    mce = ModelCallEndEvent(
                        reply_id=reply_id,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        finished_reason=finish_reason,
                    )
                    yield mce; msg.append_event(mce)

        except BaseException:
            # 取消/异常：收尾 open blocks，写入半截文本到 memory（续跑时可见），
            # 发 ReplyEnd(INTERRUPTED)，再清理已启动的工具任务，最后向上传播。
            if text_block_id is not None:
                tbe = TextBlockEndEvent(reply_id=reply_id, block_id=text_block_id)
                yield tbe; msg.append_event(tbe)
                text_block_id = None
            if thinking_block_id is not None:
                thbe = ThinkingBlockEndEvent(reply_id=reply_id, block_id=thinking_block_id)
                yield thbe; msg.append_event(thbe)
                thinking_block_id = None
            _full_text = "".join(text_parts)
            _full_reasoning = "".join(reasoning_parts)
            if _full_text.strip():
                self.agent.memory.add_assistant(_full_text, reasoning=_full_reasoning or None)
            re = ReplyEndEvent(
                session_id=session_id, reply_id=reply_id,
                finished_reason=ReplyFinishedReason.INTERRUPTED,
            )
            yield re; msg.append_event(re)
            self.state.reply_msg = msg
            await self.tool_handler.drain(tool_tasks)
            raise

        # ── 阶段 2：关闭 LLM span ──
        full_text = "".join(text_parts)
        full_reasoning = "".join(reasoning_parts)
        self.state.reply_msg = msg

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
                    ev = self._append_internal_user_message(
                        LENGTH_CONTINUATION_PROMPT,
                        reason="length_recovery",
                    )
                    yield ev; msg.append_event(ev)
                    re = ReplyEndEvent(
                        session_id=session_id, reply_id=reply_id,
                        finished_reason=self.state.done_reason or ReplyFinishedReason.COMPLETED,
                    )
                    yield re; msg.append_event(re)
                    self.state.reply_msg = msg
                    return

                if finish_reason == "unknown":
                    re = ReplyEndEvent(
                        session_id=session_id, reply_id=reply_id,
                        finished_reason=self.state.done_reason or ReplyFinishedReason.COMPLETED,
                    )
                    yield re; msg.append_event(re)
                    self.state.reply_msg = msg
                    return

                # 分支 3e：on_stop hook — 外部可阻止 Agent 停止。
                stop_output = await self.agent.hook_manager.trigger(
                    ON_STOP,
                    lambda: StopInput(
                        session_id=self.state.runtime_context.get("session_id", ""),
                        turn_id=self.state.turn_id,
                        iteration=self.state.iteration,
                        last_assistant_text=full_text,
                        finish_reason=finish_reason,
                        token_usage=dict(self.state.token_usage),
                        runtime_context=self.state.runtime_context,
                    ),
                )
                if stop_output is not None and stop_output.decision == "block":
                    logger.info(
                        "[react_runner] on_stop hook 阻止停止; 迭代=%s reason=%s",
                        self.state.iteration,
                        stop_output.reason[:100],
                    )
                    ev = self._append_internal_user_message(
                        stop_output.reason or "继续工作。",
                        reason="stop_hook_block",
                    )
                    yield ev; msg.append_event(ev)
                    re = ReplyEndEvent(
                        session_id=session_id, reply_id=reply_id,
                        finished_reason=self.state.done_reason or ReplyFinishedReason.COMPLETED,
                    )
                    yield re; msg.append_event(re)
                    self.state.reply_msg = msg
                    return  # 不设 COMPLETED，_loop 继续下一轮

                # 分支 3f：正常结束。
                self.state.done_reason = ReplyFinishedReason.COMPLETED
                self.state.status = RunStatus.COMPLETED
                re = ReplyEndEvent(
                    session_id=session_id, reply_id=reply_id,
                    finished_reason=self.state.done_reason,
                )
                yield re; msg.append_event(re)
                self.state.reply_msg = msg
                return

            # ── 以下处理空文本（full_text 为空）的情况 ──

            # finish_reason=length 但文本为空：可能是 max_tokens 设太小，
            # provider 返回了空 content。直接结束，避免无限循环。
            if finish_reason == "length":
                re = ReplyEndEvent(
                    session_id=session_id, reply_id=reply_id,
                    finished_reason=self.state.done_reason or ReplyFinishedReason.COMPLETED,
                )
                yield re; msg.append_event(re)
                self.state.reply_msg = msg
                return
            # finish_reason=unknown：无法判断，直接结束。
            if finish_reason == "unknown":
                re = ReplyEndEvent(
                    session_id=session_id, reply_id=reply_id,
                    finished_reason=self.state.done_reason or ReplyFinishedReason.COMPLETED,
                )
                yield re; msg.append_event(re)
                self.state.reply_msg = msg
                return

            # 已经在"强制最终化"阶段还是空 → 彻底失败。
            if self.state.finalization_retrying:
                self.state.done_reason = ReplyFinishedReason.ERROR
                self.state.status = RunStatus.ERROR
                self.state.error = EMPTY_FINAL_RESPONSE_MESSAGE
                self.state.error_code = "empty_response"
                re = ReplyEndEvent(
                    session_id=session_id, reply_id=reply_id,
                    finished_reason=self.state.done_reason,
                )
                yield re; msg.append_event(re)
                self.state.reply_msg = msg
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
                re = ReplyEndEvent(
                    session_id=session_id, reply_id=reply_id,
                    finished_reason=self.state.done_reason or ReplyFinishedReason.COMPLETED,
                )
                yield re; msg.append_event(re)
                self.state.reply_msg = msg
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
            ev = self._append_internal_user_message(
                FINALIZATION_RETRY_PROMPT,
                reason="empty_finalization_retry",
            )
            yield ev; msg.append_event(ev)
            re = ReplyEndEvent(
                session_id=session_id, reply_id=reply_id,
                finished_reason=self.state.done_reason or ReplyFinishedReason.COMPLETED,
            )
            yield re; msg.append_event(re)
            self.state.reply_msg = msg
            return

        # ── 阶段 4：等待全部工具完成 ──
        # gather_results 会并发等待所有工具任务，处理取消，返回按调用顺序排列的结果。
        self.state.empty_content_retries = 0
        self.state.finalization_retrying = False
        results, cancelled = await self.tool_handler.gather_results(
            tool_calls, tool_tasks, self.state
        )

        # ── 阶段 5：成组写入 memory + yield 事件 ──
        # ⚠️ memory 写入必须先于 yield，否则调用方取消 generator 时
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

        # 收集工具返回的 pending hint（延后处理，确保在所有 tool_result 之后追加，
        # 否则会打断 assistant(tool_calls) → tool(...) → tool(...) 顺序，
        # 导致 provider 报 "tool call result does not follow tool call"）。
        pending_hints: list[AgentStreamEvent] = []

        for tc, result in zip(tool_calls, results):
            # 写入 tool_result 到 memory
            self.agent.memory.add_tool_result(
                tc.id, result.result or f"[{tc.name}] 已完成"
            )
            # 产 ToolResult 三段式事件
            trs = ToolResultStartEvent(
                reply_id=reply_id, tool_call_id=tc.id, tool_call_name=tc.name,
            )
            yield trs; msg.append_event(trs)
            if result.result:
                trd = ToolResultTextDeltaEvent(
                    reply_id=reply_id, tool_call_id=tc.id, delta=result.result,
                )
                yield trd; msg.append_event(trd)
            state = ToolResultState.SUCCESS if not result.error else ToolResultState.ERROR
            tre = ToolResultEndEvent(
                reply_id=reply_id, tool_call_id=tc.id,
                state=state, metadata=result.metadata or {},
            )
            yield tre; msg.append_event(tre)
            # 收集 pending hint（延后追加）
            if result.event is not None:
                pending_hints.append(result.event)

        # 统一追加 HintBlockEvent：确保在 tool_result 之后，消息顺序合法。
        for ev in pending_hints:
            if isinstance(ev, HintBlockEvent):
                content = ev.hint if isinstance(ev.hint, str) else str(ev.hint)
            else:
                content = str(ev)
            self.agent.memory.add_raw({"role": "user", "content": content})
            hint_ev = HintBlockEvent(
                reply_id=reply_id,
                block_id=uuid.uuid4().hex[:16],
                source="tool",
                hint=content,
            )
            yield hint_ev; msg.append_event(hint_ev)

        if cancelled:
            re = ReplyEndEvent(
                session_id=session_id, reply_id=reply_id,
                finished_reason=ReplyFinishedReason.INTERRUPTED,
            )
            yield re; msg.append_event(re)
            self.state.reply_msg = msg
            raise CancelledError()

        re = ReplyEndEvent(
            session_id=session_id, reply_id=reply_id,
            finished_reason=self.state.done_reason or ReplyFinishedReason.COMPLETED,
        )
        yield re; msg.append_event(re)
        self.state.reply_msg = msg
        # 工具已执行，本轮不标记完成，外层循环会进入下一轮让模型读取工具结果。

    # ── 辅助方法 ────────────────────────────────────────────────────────────

    def _append_internal_user_message(self, content: str, *, reason: str) -> AgentStreamEvent:
        """向 memory 追加一条内部用户消息（对用户不可见），并返回 HintBlockEvent。

        用于：
          - length 截断后的续写提示
          - 空响应最终化提示
          - on_stop hook block 时的 continuation prompt
        """
        self.agent.memory.add_raw({"role": "user", "content": content})
        return HintBlockEvent(
            reply_id=self.state.reply_id or "",
            block_id=uuid.uuid4().hex[:16],
            source="system",
            hint=content,
            metadata={"hide": True, "internal": True, "reason": reason},
        )


