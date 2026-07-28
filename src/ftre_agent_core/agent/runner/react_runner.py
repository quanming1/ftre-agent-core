"""ReActRunner — ReAct Agent 的核心执行引擎（状态机重构版）。

职责：
  - 驱动 Reason → Act → Observe 循环（通过 _decide + 执行器）
  - 管理运行锁（同一 Agent 禁止并发 run）
  - 取消入口（cancel_nowait → Task.cancel）
  - Tracing 根 span 生命周期
  - 统一终态写入（_finalize）
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
from ._actions import Reasoning, Acting, Exit, TurnResult
from ._decide import decide
from ._execute_acting import ActingExecutor
from ._execute_exit import ExitExecutor
from ._execute_reasoning import ReasoningExecutor
from ._state import RunState, RunStatus, CancelledError
from .tool_handler import ToolHandler

if TYPE_CHECKING:
    from ..react import ReActAgent

logger = logging.getLogger(__name__)

_REASON_TO_STATUS = {
    ReplyFinishedReason.COMPLETED: RunStatus.COMPLETED,
    ReplyFinishedReason.INTERRUPTED: RunStatus.CANCELLED,
    ReplyFinishedReason.ERROR: RunStatus.ERROR,
    ReplyFinishedReason.EXCEED_MAX_ITERS: RunStatus.COMPLETED,
}

_TRACE_STATUS = {
    RunStatus.COMPLETED: TraceRunStatus.COMPLETED,
    RunStatus.CANCELLED: TraceRunStatus.CANCELLED,
    RunStatus.ERROR: TraceRunStatus.ERROR,
}


class ReActRunner:
    """ReAct Agent 的核心执行引擎。"""

    def __init__(self, agent: "ReActAgent"):
        self.agent = agent
        self.state = RunState()
        self._run_task: asyncio.Task | None = None
        self._llm = LLMHandler(
            agent.model,
            agent.api_key,
            agent.api_base,
            agent.api_type,
            max_tokens=agent.max_tokens,
            reasoning_effort=agent.reasoning_effort,
        )
        self._tool_handler = ToolHandler(agent.tool_registry, agent.hook_manager)

    @property
    def llm(self) -> LLMHandler:
        return self._llm

    @property
    def tool_handler(self) -> ToolHandler:
        return self._tool_handler

    async def run(
        self,
        message,
        runtime_context: dict | None = None,
    ) -> AsyncGenerator[AgentStreamEvent, None]:
        """启动一次完整的 ReAct 执行。"""
        # 并发锁
        if self._run_task is not None and not self._run_task.done():
            raise RuntimeError("Agent is already running")

        self._run_task = asyncio.current_task()
        self.state.runtime_context = runtime_context or {}
        self.state.runtime_context.setdefault(
            "max_iterations", self.agent.max_iterations,
        )
        self.state.start()

        # 准备 tracing
        trace_metadata = self.state.runtime_context.get("trace_metadata") or {}
        if not isinstance(trace_metadata, dict):
            trace_metadata = {"value": trace_metadata}
        trace_metadata = {
            "model": self.agent.model,
            "api_type": self.agent.api_type,
            **trace_metadata,
        }
        trace_tags = self.state.runtime_context.get("trace_tags") or []
        if isinstance(trace_tags, str):
            trace_tags = [trace_tags]

        self.state.trace_span = self.agent.tracer.start_run(
            str(self.state.runtime_context.get("trace_name") or "react_agent"),
            RunType.AGENT,
            inputs={"message": message},
            metadata=trace_metadata,
            tags=list(trace_tags),
        )

        # 写入用户消息到 memory
        if isinstance(message, str):
            self.agent.memory.add_user(message)
        else:
            for msg in message:
                self.agent.memory.add_raw(msg)

        # ReplyStart（只产一次）
        reply_id = uuid.uuid4().hex[:16]
        self.state.reply_id = reply_id
        session_id = self.state.runtime_context.get("session_id", "")
        model_name = self.agent.model

        yield ReplyStartEvent(
            session_id=session_id, reply_id=reply_id, name=model_name,
        )

        # 主循环
        try:
            async for event in self._loop():
                yield event
        except asyncio.CancelledError:
            self._finalize(ReplyFinishedReason.INTERRUPTED)
            yield ReplyEndEvent(
                session_id=session_id, reply_id=reply_id,
                finished_reason=ReplyFinishedReason.INTERRUPTED,
            )
        except Exception:
            self._finalize(ReplyFinishedReason.ERROR)
            yield ReplyEndEvent(
                session_id=session_id, reply_id=reply_id,
                finished_reason=ReplyFinishedReason.ERROR,
                error={"message": str(self.state.error or "Unknown error")},
            )
            raise
        finally:
            # Tracing 收尾
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
            self._run_task = None

    def cancel_nowait(self) -> None:
        """外部调用：取消当前执行。"""
        if self._run_task is not None and not self._run_task.done():
            self._run_task.cancel()

    async def _loop(self) -> AsyncGenerator[AgentStreamEvent, None]:
        """ReAct 主循环。

        iteration 只在 Reasoning 时递增，一次迭代 = 一次 LLM 调用
        （可能后跟 Acting，但不额外计数）。
        """
        prev: TurnResult | None = None
        max_iters = self.agent.max_iterations

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
                action = decide(self.state, prev)

                if isinstance(action, Reasoning):
                    self.state.iteration += 1
                    # on_turn_start hook
                    await self._trigger_on_turn_start()
                    async for event in reasoning_executor.stream(action):
                        yield event
                    prev = reasoning_executor.result

                elif isinstance(action, Acting):
                    async for event in acting_executor.stream(action):
                        yield event
                    prev = None

                elif isinstance(action, Exit):
                    async for event in exit_executor.stream(action):
                        yield event
                    if exit_executor.outcome.should_continue:
                        prev = None
                        continue
                    # on_turn_end hook（正常退出时触发）
                    await self._trigger_on_turn_end()
                    return

        except CancelledError:
            self._finalize(ReplyFinishedReason.INTERRUPTED)
            yield ReplyEndEvent(
                session_id=self.state.runtime_context.get("session_id", ""),
                reply_id=self.state.reply_id,
                finished_reason=ReplyFinishedReason.INTERRUPTED,
            )

    def _finalize(self, reason: ReplyFinishedReason) -> None:
        """统一终态写入。"""
        self.state.done_reason = reason
        self.state.status = _REASON_TO_STATUS.get(reason, RunStatus.ERROR)

    async def _trigger_on_turn_start(self) -> None:
        """触发 on_turn_start hook。"""
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
        """触发 on_turn_end hook（只读观察）。"""
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
