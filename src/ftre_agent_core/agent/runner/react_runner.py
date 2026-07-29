"""ReActRunner — ReAct Agent 的核心执行引擎（状态机重构版）。

整体设计借鉴 AgentScope 的 _next_action() 纯决策函数模式，同时保留
原有的 LLM 重试、空响应恢复、ON_STOP Hook、Tracing、工具并发执行、
成组写入 Memory 等生产能力。

架构分三层：

  decide()               纯决策函数，只读状态，返回动作类型
  ReasoningExecutor      执行 Reasoning 动作：调 LLM + 流式 + 重试
  ActingExecutor         执行 Acting 动作：工具并发 + 成组写入 Memory
  ExitExecutor           执行 Exit 动作：ON_STOP Hook + 产出 ReplyEnd

主循环 _loop() 只做 match 分发：

  while True:
      action = decide(state, prev)
      match action:
          Reasoning → prev = reasoning_executor.stream(action)
          Acting    → acting_executor.stream(action); prev = None
          Exit      → exit_executor.stream(action); return

取消协议：
  外部调用 cancel_nowait() → Task.cancel() → CancelledError 沿调用栈传播
  → _loop 的 except 捕获 → _finalize(INTERRUPTED) → yield ReplyEnd(INTERRUPTED)
  不引入 CancellationToken，纯依赖 asyncio 协作式取消。

运行锁：
  同一 Agent 禁止并发 run()。run() 开始时记录 asyncio.current_task()，
  重复调用直接抛 RuntimeError。finally 中清除。
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, AsyncGenerator

from ...event import AgentStreamEvent, ReplyStartEvent, ReplyEndEvent
from ...tracing import RunStatus as TraceRunStatus, RunType
from ...types import ReplyFinishedReason
from ...llm import LLMHandler
from ._state import Reasoning, Acting, Exit, TurnResult, RunState, RunStatus, CancelledError
from ._execute_acting import ActingExecutor, ExitExecutor
from ._execute_reasoning import ReasoningExecutor
from .tool_handler import ToolHandler

if TYPE_CHECKING:
    from ..react import ReActAgent

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# 纯决策函数
# ═══════════════════════════════════════════════════════════════

# 空响应最多重试次数（不含强制最终化那一次）。
# 超过此次数后进入"强制最终化"阶段：去掉工具，注入提示，让模型只输出文本。
MAX_EMPTY_RESPONSE_RETRIES = 2

# 强制最终化时注入 Memory 的提示词，要求模型直接给出最终回复。
FINALIZATION_RETRY_PROMPT = "请根据上面的对话，直接给出回复用户的最终内容。"

# 强制最终化后仍然返回空响应时的错误提示。
EMPTY_FINAL_RESPONSE_MESSAGE = "模型多次重试后仍未生成可见的最终文本回复。"


def decide(state: RunState, prev: TurnResult | None) -> Reasoning | Acting | Exit:
    """根据当前状态和上一轮 TurnResult 决定下一步动作。

    纯函数：不执行 I/O，不 yield 事件。
    副作用仅限修改 state.empty_retries 和 state.in_finalization。

    判断优先级（从高到低）：
        1. prev.error 非空 → Exit(ERROR)
           LLM 调用重试耗尽或遇到不可重试错误，直接退出。
        2. prev.tool_calls 非空 → Acting(tool_calls)
           模型本轮产生了工具调用，需要执行这些工具。
        3. prev.text 非空且无工具调用 → Exit(COMPLETED)
           模型本轮给出了纯文本回答，正常完成。
        4. 空响应 + in_finalization → Exit(ERROR)
           已在强制最终化阶段还是空响应，彻底失败。
        5. 空响应 + empty_retries < MAX → Reasoning() 重试
           模型返回空内容，重试计数 +1，继续推理。
        6. 空响应 + 重试耗尽 → Reasoning(最终化提示, force_no_tools)
           重试次数用尽，进入强制最终化：去掉工具，注入提示。
        7. iteration >= max_iterations → Exit(EXCEED_MAX_ITERS)
           达到最大迭代次数，防止无限循环。
        8. 默认 → Reasoning()
           首轮（prev=None）或工具执行后（prev=None），继续推理。

    Args:
        state: 当前 RunState（只读 iteration / empty_retries /
               in_finalization / runtime_context）。
        prev: 上一轮推理的 TurnResult，None 表示首轮或刚执行完工具。

    Returns:
        Reasoning: 继续调用大模型进行推理
        Acting:    执行模型产生的工具调用
        Exit:      结束（或暂停）当前回复
    """
    max_iterations = state.runtime_context.get("max_iterations")

    # 1. LLM 错误 → 直接退出
    if prev is not None and prev.error is not None:
        return Exit(
            finished_reason=ReplyFinishedReason.ERROR,
            error=f"[{prev.error.code}] {prev.error.message}",
            error_code=prev.error.code,
        )

    # 2. 有工具调用 → 执行工具
    if prev is not None and prev.tool_calls:
        return Acting(tool_calls=prev.tool_calls)

    # 3. 有非空文本且无工具调用 → 正常完成
    if prev is not None and prev.text.strip():
        return Exit(finished_reason=ReplyFinishedReason.COMPLETED)

    # 4-6. 空响应处理（prev 非空但文本为空）
    if prev is not None and not prev.text.strip():
        has_reasoning = bool(prev.reasoning and prev.reasoning.strip())
        has_tools = bool(prev.tool_calls)
        logger.warning(
            "[react] 空响应: text=%r reasoning=%d chars tools=%d finish_reason=%s "
            "empty_retries=%d/%d in_finalization=%s iteration=%d",
            prev.text[:80] if prev.text else "",
            len(prev.reasoning or ""),
            len(prev.tool_calls or []),
            prev.finish_reason,
            state.empty_retries,
            MAX_EMPTY_RESPONSE_RETRIES,
            state.in_finalization,
            state.iteration,
        )
        if state.in_finalization:
            # 4. 已在最终化阶段还是空 → 彻底失败
            return Exit(
                finished_reason=ReplyFinishedReason.ERROR,
                error=EMPTY_FINAL_RESPONSE_MESSAGE,
                error_code="empty_response",
            )
        if state.empty_retries < MAX_EMPTY_RESPONSE_RETRIES:
            # 5. 重试次数未达上限 → 继续 Reasoning
            state.empty_retries += 1
            return Reasoning()
        # 6. 重试耗尽 → 进入强制最终化
        state.in_finalization = True
        return Reasoning(
            hint=FINALIZATION_RETRY_PROMPT,
            force_no_tools=True,
        )

    # 7. 达到最大迭代次数
    if max_iterations is not None and state.iteration >= max_iterations:
        return Exit(finished_reason=ReplyFinishedReason.EXCEED_MAX_ITERS)

    # 8. 默认 → 继续推理
    return Reasoning()

# ReplyFinishedReason → RunStatus 映射
# EXCEED_MAX_ITERS 映射为 COMPLETED（非错误，只是达到上限）
_REASON_TO_STATUS = {
    ReplyFinishedReason.COMPLETED: RunStatus.COMPLETED,
    ReplyFinishedReason.INTERRUPTED: RunStatus.CANCELLED,
    ReplyFinishedReason.ERROR: RunStatus.ERROR,
    ReplyFinishedReason.EXCEED_MAX_ITERS: RunStatus.COMPLETED,
}

# RunStatus → Tracing RunStatus 映射
_TRACE_STATUS = {
    RunStatus.COMPLETED: TraceRunStatus.COMPLETED,
    RunStatus.CANCELLED: TraceRunStatus.CANCELLED,
    RunStatus.ERROR: TraceRunStatus.ERROR,
}


class ReActRunner:
    """ReAct Agent 的核心执行引擎。

    职责：
      - 驱动 Reason → Act → Observe 循环（通过 _decide + 执行器）
      - 管理运行锁（同一 Agent 禁止并发 run）
      - 取消入口（cancel_nowait → Task.cancel）
      - Tracing 根 span 生命周期
      - 统一终态写入（_finalize）
    """

    def __init__(self, agent: "ReActAgent"):
        # 关联的 ReActAgent 实例（提供 model / memory / hook_manager / tracer 等依赖）
        self.agent = agent
        # 本次 run() 的可变运行状态（iteration / empty_retries / trace_span 等）
        self.state = RunState()
        # 当前 run() 对应的 asyncio.Task，用于取消和并发锁检查
        self._run_task: asyncio.Task | None = None
        # LLM 调用封装（Provider 调用、流式解析、reasoning 提取等）
        self._llm = LLMHandler(
            agent.model,
            agent.api_key,
            agent.api_base,
            agent.api_type,
            max_tokens=agent.max_tokens,
            reasoning_effort=agent.reasoning_effort,
        )
        # 工具并发调度、取消传播和结果归并
        self._tool_handler = ToolHandler(agent.tool_registry, agent.hook_manager)

    @property
    def llm(self) -> LLMHandler:
        """LLM 调用封装实例。"""
        return self._llm

    @property
    def tool_handler(self) -> ToolHandler:
        """工具执行器实例。"""
        return self._tool_handler

    async def run(
        self,
        message,
        runtime_context: dict | None = None,
    ) -> AsyncGenerator[AgentStreamEvent, None]:
        """启动一次完整的 ReAct 执行。

        一次 run() 的生命周期：
          1. 并发锁检查 + 记录当前 Task
          2. 初始化 RunState（start()）
          3. 开启 Tracing 根 span
          4. 写入用户消息到 Memory
          5. 产出 ReplyStartEvent（只产一次）
          6. 驱动 _loop() 主循环
          7. 异常/取消/正常退出路径统一经过 _finalize()
          8. finally 中关闭 Tracing span + 释放运行锁

        Args:
            message: 用户消息（字符串或消息列表）。
            runtime_context: 调用方上下文（session_id、tracing 元数据等）。

        Yields:
            AgentStreamEvent: 回复过程中的所有流式事件。
        """
        # ── 并发锁：同一 Agent 禁止并发 run() ──
        if self._run_task is not None and not self._run_task.done():
            raise RuntimeError("Agent is already running")

        # 记录当前 Task，供 cancel_nowait() 使用
        self._run_task = asyncio.current_task()
        self.state.runtime_context = runtime_context or {}
        # max_iterations 放入 runtime_context 供 _decide() 读取
        self.state.runtime_context.setdefault(
            "max_iterations", self.agent.max_iterations,
        )
        self.state.start()

        # ── 准备 Tracing 元数据 ──
        # 调用方可通过 runtime_context 传入自定义 trace 元数据，
        # 这里兜底处理：非 dict 自动包一层，然后强制带上 model / api_type
        trace_metadata = self.state.runtime_context.get("trace_metadata") or {}
        if not isinstance(trace_metadata, dict):
            trace_metadata = {"value": trace_metadata}
        trace_metadata = {
            "model": self.agent.model,
            "api_type": self.agent.api_type,
            **trace_metadata,  # 调用方自定义字段可覆盖上面的默认值
        }
        # tags 允许传单个字符串，统一规整成 list
        trace_tags = self.state.runtime_context.get("trace_tags") or []
        if isinstance(trace_tags, str):
            trace_tags = [trace_tags]

        # 开启根 span（AGENT 节点）；tracer 未配置 exporter 时为空操作
        self.state.trace_span = self.agent.tracer.start_run(
            str(self.state.runtime_context.get("trace_name") or "react_agent"),
            RunType.AGENT,
            inputs={"message": message},
            metadata=trace_metadata,
            tags=list(trace_tags),
        )

        # ── 将用户消息写入 Memory ──
        if isinstance(message, str):
            self.agent.memory.add_user(message)
        else:
            # 列表形式：原样写入 memory（含 system 消息等）
            for msg in message:
                self.agent.memory.add_raw(msg)

        # ── 产出 ReplyStartEvent（一次 run() 只产一次）──
        reply_id = uuid.uuid4().hex[:16]
        self.state.reply_id = reply_id
        session_id = self.state.runtime_context.get("session_id", "")
        model_name = self.agent.model

        yield ReplyStartEvent(
            session_id=session_id, reply_id=reply_id, name=model_name,
        )

        # ── 主循环 + 异常/收尾处理 ──
        try:
            async for event in self._loop():
                yield event

        except asyncio.CancelledError:
            # 取消路径：_finalize 设置 INTERRUPTED 状态，产出 ReplyEnd
            self._finalize(ReplyFinishedReason.INTERRUPTED)
            yield ReplyEndEvent(
                session_id=session_id, reply_id=reply_id,
                finished_reason=ReplyFinishedReason.INTERRUPTED,
            )

        except Exception:
            # 异常路径：_finalize 设置 ERROR 状态，产出 ReplyEnd
            self._finalize(ReplyFinishedReason.ERROR)
            yield ReplyEndEvent(
                session_id=session_id, reply_id=reply_id,
                finished_reason=ReplyFinishedReason.ERROR,
                error={"message": str(self.state.error or "Unknown error")},
            )
            raise

        finally:
            # Tracing 收尾：依据 RunState.status 映射 trace 状态并关闭根 span
            if self.state.trace_span and not self.state.trace_span.ended:
                self.state.trace_span.end(
                    status=_TRACE_STATUS.get(self.state.status, TraceRunStatus.ERROR),
                    outputs={
                        "success": self.state.status == RunStatus.COMPLETED,
                        "done_reason": self.state.done_reason,
                        "iterations": self.state.iteration,
                    },
                    error=self.state.error if self.state.status == RunStatus.ERROR else None,
                )
            # 释放运行锁
            self._run_task = None

    def cancel_nowait(self) -> None:
        """外部调用：取消当前执行。

        对 self._run_task 执行 Task.cancel()，CancelledError 会沿
        await 调用栈传播到 _loop() 或 _execute_reasoning / _execute_acting
        中的当前 await 点，最终被 run() 的 except 捕获并转换为
        INTERRUPTED 结果。
        """
        if self._run_task is not None and not self._run_task.done():
            self._run_task.cancel()

    async def _loop(self) -> AsyncGenerator[AgentStreamEvent, None]:
        """ReAct 主循环：Reason → Act → Observe。

        循环逻辑：
          1. 调用 _decide(state, prev) 获取下一步动作
          2. 根据动作类型分发到对应执行器
          3. Reasoning → 递增 iteration，调 LLM，更新 prev
          4. Acting    → 执行工具，清除 prev（下一轮重新推理）
          5. Exit      → 产出结束事件，触发 on_turn_end，return

        iteration 计数规则：
          只在 Reasoning 时递增。一次"迭代"= 一次 LLM 调用，
          可能后跟一次 Acting（工具执行），但不额外计数。
          这样 max_iterations=N 表示最多调用 N 次 LLM。

        Exit + should_continue 的特殊路径：
          on_stop hook 返回 block 时，ExitExecutor 产出 HintBlockEvent
          但不产 ReplyEndEvent，返回 ExitOutcome(should_continue=True)。
          主循环注入续写提示到 Memory，清除 prev，继续循环。
        """
        prev: TurnResult | None = None
        max_iters = self.agent.max_iterations

        # 创建三个执行器实例（循环内复用）
        reasoning_executor = ReasoningExecutor(
            self.agent, self.state, self._llm, self.agent.hook_manager,
        )
        acting_executor = ActingExecutor(
            self.agent, self.state, self._tool_handler,
        )
        exit_executor = ExitExecutor(
            self.agent, self.state, self.agent.hook_manager,
        )

        try:
            while True:
                # 纯决策函数：只读状态，返回动作类型
                action = decide(self.state, prev)

                if isinstance(action, Reasoning):
                    # ── Reasoning：调 LLM ──
                    self.state.iteration += 1
                    # on_turn_start hook（可注入消息到 Memory）
                    await self._trigger_on_turn_start()
                    # 执行 LLM 调用 + 流式消费 + 重试
                    async for event in reasoning_executor.stream(action):
                        yield event
                    # 获取本轮推理的结构化产物，供下一轮 _decide() 消费
                    prev = reasoning_executor.result

                elif isinstance(action, Acting):
                    # ── Acting：执行工具 ──
                    # 并发执行所有工具调用，成组写入 Memory
                    async for event in acting_executor.stream(action):
                        yield event
                    # 清除 prev：工具执行后需要重新调 LLM 读取工具结果
                    prev = None

                elif isinstance(action, Exit):
                    # ── Exit：结束（或暂停）当前回复 ──
                    async for event in exit_executor.stream(action):
                        yield event
                    # ON_STOP hook 返回 block 时不退出
                    if exit_executor.outcome.should_continue:
                        # 注入续写提示已在 ExitExecutor 中完成
                        prev = None
                        continue
                    # 正常退出 → 触发 on_turn_end hook，结束循环
                    await self._trigger_on_turn_end()
                    return

        except CancelledError:
            # 内部取消异常（来自 _execute_acting 的 cancelled=True 路径）
            self._finalize(ReplyFinishedReason.INTERRUPTED)
            yield ReplyEndEvent(
                session_id=self.state.runtime_context.get("session_id", ""),
                reply_id=self.state.reply_id,
                finished_reason=ReplyFinishedReason.INTERRUPTED,
            )

    def _finalize(self, reason: ReplyFinishedReason) -> None:
        """统一终态写入。

        所有退出路径（正常、取消、异常、超限）都经过此函数，
        确保 RunState 的 status 和 done_reason 一致。

        Args:
            reason: 结束原因（COMPLETED / INTERRUPTED / ERROR / EXCEED_MAX_ITERS）
        """
        self.state.done_reason = reason
        self.state.status = _REASON_TO_STATUS.get(reason, RunStatus.ERROR)

    async def _trigger_on_turn_start(self) -> None:
        """触发 on_turn_start hook。

        在每次 Reasoning（LLM 调用）之前触发。
        hook 可以注入 system/user 消息（如每日提醒、上下文补充），
        注入的消息会追加到 Memory，Agent 在本轮迭代中可见。
        """
        from ...hooks import ON_TURN_START, TurnStartInput, TurnStartOutput

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

    async def _trigger_on_turn_end(self) -> None:
        """触发 on_turn_end hook（只读观察）。

        在 Agent 正常退出（Exit + ON_STOP allow）时触发。
        hook 不能阻止退出，decision 字段被忽略。
        用于遥测、日志、UI 通知。
        """
        from ...hooks import ON_TURN_END, TurnEndInput

        await self.agent.hook_manager.trigger(
            ON_TURN_END,
            lambda: TurnEndInput(
                session_id=self.state.runtime_context.get("session_id", ""),
                turn_id=self.state.turn_id,
                iteration=self.state.iteration,
                done_reason=str(self.state.done_reason or ReplyFinishedReason.COMPLETED),
                runtime_context=self.state.runtime_context,
            ),
        )
