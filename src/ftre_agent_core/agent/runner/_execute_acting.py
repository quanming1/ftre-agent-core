"""Acting 执行器 + Exit 执行器。

ActingExecutor：并发执行工具 + 成组写入 Memory + 产出事件。
ExitExecutor：ON_STOP Hook 检查 + 产出 ReplyEndEvent + 设置终态。
"""
from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, AsyncGenerator

from ...event import (
    AgentStreamEvent, ReplyEndEvent, HintBlockEvent,
    ToolResultStartEvent, ToolResultTextDeltaEvent, ToolResultEndEvent,
)
from ...message import ToolResultState
from ...types import ReplyFinishedReason
from ._state import Acting, Exit, ExitOutcome, RunStatus, CancelledError
from .tool_handler import ToolHandler

if TYPE_CHECKING:
    from ...hooks import FtreCoreHookManager
    from ._state import RunState


# ═══════════════════════════════════════════════════════════════
# ActingExecutor
# ═══════════════════════════════════════════════════════════════

class ActingExecutor:
    """执行 Acting 动作：并发执行工具，成组写入 memory，产出事件。"""

    def __init__(self, agent, state: "RunState", tool_handler: ToolHandler):
        self.agent = agent
        self.state = state
        self.tool_handler = tool_handler

    async def stream(self, action: Acting) -> AsyncGenerator[AgentStreamEvent, None]:
        """执行工具调用，yield 事件。"""
        reply_id = self.state.reply_id
        tool_calls = action.tool_calls

        # spawn 所有工具任务
        tool_tasks: dict[str, asyncio.Task] = {}
        for call in tool_calls:
            tool_tasks[call.id] = self.tool_handler.spawn(
                call,
                self.state,
                parent_span=self.state.trace_span,
            )

        # 等待全部完成
        results, cancelled = await self.tool_handler.gather_results(
            tool_calls, tool_tasks, self.state,
        )

        # 成组写入 memory：assistant(tool_calls) → tool(result_1) → ...
        self.agent.memory.add_raw(
            self.tool_handler.build_assistant_message(
                tool_calls=tool_calls,
            )
        )

        # 收集 pending hints（延后追加）
        pending_hints: list[AgentStreamEvent] = []

        for tc, result in zip(tool_calls, results):
            self.agent.memory.add_tool_result(
                tc.id, result.result or f"[{tc.name}] 已完成"
            )

            yield ToolResultStartEvent(
                reply_id=reply_id, tool_call_id=tc.id, tool_call_name=tc.name,
            )
            if result.result:
                yield ToolResultTextDeltaEvent(
                    reply_id=reply_id, tool_call_id=tc.id, delta=result.result,
                )
            state = ToolResultState.SUCCESS if not result.error else ToolResultState.ERROR
            yield ToolResultEndEvent(
                reply_id=reply_id, tool_call_id=tc.id,
                state=state, metadata=result.metadata or {},
            )

            if result.event is not None:
                pending_hints.append(result.event)

        # 统一追加 hints（确保在 tool_result 之后）
        for ev in pending_hints:
            if isinstance(ev, HintBlockEvent):
                content = ev.hint if isinstance(ev.hint, str) else str(ev.hint)
            else:
                content = str(ev)
            self.agent.memory.add_raw({"role": "user", "content": content})
            yield HintBlockEvent(
                reply_id=reply_id,
                block_id=uuid.uuid4().hex[:16],
                source="tool",
                hint=content,
            )

        if cancelled:
            raise CancelledError()


# ═══════════════════════════════════════════════════════════════
# ExitExecutor
# ═══════════════════════════════════════════════════════════════

class ExitExecutor:
    """执行 Exit 动作：ON_STOP Hook 检查 + 产出 ReplyEnd + 设置终态。"""

    def __init__(self, agent, state: "RunState", hook_manager: "FtreCoreHookManager"):
        self.agent = agent
        self.state = state
        self.hook_manager = hook_manager
        self.outcome: ExitOutcome = ExitOutcome()

    async def stream(self, action: Exit) -> AsyncGenerator[AgentStreamEvent, None]:
        """执行退出逻辑，yield 事件。"""
        session_id = self.state.runtime_context.get("session_id", "")
        reply_id = self.state.reply_id

        # 仅 COMPLETED 触发 ON_STOP
        if action.finished_reason == ReplyFinishedReason.COMPLETED:
            from ...hooks import ON_STOP, StopInput

            stop_output = await self.hook_manager.trigger(
                ON_STOP,
                lambda: StopInput(
                    session_id=session_id,
                    turn_id=self.state.turn_id,
                    iteration=self.state.iteration,
                    runtime_context=self.state.runtime_context,
                ),
            )

            if stop_output is not None and stop_output.decision == "block":
                # ON_STOP block → 不退出，返回 continue
                hint = stop_output.reason or "继续工作。"
                self.agent.memory.add_raw({"role": "user", "content": hint})
                yield HintBlockEvent(
                    reply_id=reply_id,
                    block_id=uuid.uuid4().hex[:16],
                    source="system",
                    hint=hint,
                    metadata={"hide": True, "internal": True, "reason": "stop_hook_block"},
                )
                self.outcome = ExitOutcome(should_continue=True, continue_hint=hint)
                return

        # 正常退出 → 设置终态 + yield ReplyEndEvent
        self._finalize(action.finished_reason, action.error, action.error_code)

        yield ReplyEndEvent(
            session_id=session_id,
            reply_id=reply_id,
            finished_reason=action.finished_reason,
            error={"message": action.error, "code": action.error_code} if action.error else None,
        )
        self.outcome = ExitOutcome()

    def _finalize(self, reason: ReplyFinishedReason, error: str | None, error_code: str | None) -> None:
        """设置终态。"""
        self.state.done_reason = reason
        self.state.status = (
            RunStatus.CANCELLED if reason == ReplyFinishedReason.INTERRUPTED
            else RunStatus.ERROR if reason == ReplyFinishedReason.ERROR
            else RunStatus.COMPLETED
        )
        if error:
            self.state.error = error
        if error_code:
            self.state.error_code = error_code
