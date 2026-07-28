"""退出执行器：ON_STOP Hook + 产出 ReplyEndEvent + 设置终态。"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, AsyncGenerator

from ...event import AgentStreamEvent, ReplyEndEvent, HintBlockEvent
from ...types import ReplyFinishedReason
from ._actions import Exit, ExitOutcome
from ._state import RunStatus

if TYPE_CHECKING:
    from ...hooks import FtreCoreHookManager
    from ._state import RunState


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
