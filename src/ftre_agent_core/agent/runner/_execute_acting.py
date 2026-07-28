"""工具执行器：并发执行工具 + 成组写入 Memory + 产出事件。"""
from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, AsyncGenerator

from ...event import (
    AgentStreamEvent,
    ToolResultStartEvent, ToolResultTextDeltaEvent, ToolResultEndEvent,
    HintBlockEvent,
)
from ...message import ToolResultState
from ._actions import Acting
from ._state import CancelledError
from .tool_handler import ToolHandler

if TYPE_CHECKING:
    from ._state import RunState


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
