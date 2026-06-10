"""
ToolHandler - 异步工具调用处理器
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncGenerator

from ftre_agent_core.agent.event import (
    AgentEvent,
    tool_call_event,
    tool_result_event,
)
from ftre_agent_core.tool import ToolRegistry
from ftre_agent_core.tool.registry import ToolContext

if TYPE_CHECKING:
    from .react_runner import RunState

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    call_id: str
    name: str
    result: str
    error: str | None = None
    status: str = "completed"
    metadata: dict = field(default_factory=dict)

    @property
    def cancelled(self) -> bool:
        return self.status == "cancelled"


class ToolHandler:

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def execute(
        self,
        parsed_calls: list[tuple[str, str, dict]],
        state: "RunState",
    ) -> AsyncGenerator[AgentEvent, None]:
        for call_id, name, arguments in parsed_calls:
            yield tool_call_event(id=call_id, name=name, arguments=arguments)

        async def _run_one(call_id: str, name: str, arguments: dict) -> tuple[str, ToolResult]:
            ctx = ToolContext(call_id=call_id, name=name, arguments=arguments)
            ctx.cancel_token = state.cancel_token
            ctx.metadata["runtime_context"] = state.runtime_context
            ctx = self._run_before(ctx)

            if ctx.skipped:
                result = ToolResult(call_id=call_id, name=name, result=ctx.skip_result)
                return call_id, self._run_after(ctx, result)

            try:
                tool = self.registry.get(name)
                if tool and tool.is_async():
                    raw = await tool.execute(**ctx.arguments)
                else:
                    raw = await asyncio.to_thread(
                        self.registry.execute, name,
                        runtime_context=ctx.metadata.get("runtime_context"),
                        **ctx.arguments
                    )
                result = ToolResult(call_id=call_id, name=name, result=str(raw))
            except asyncio.CancelledError:
                result = ToolResult(call_id=call_id, name=name, result="[用户取消]", status="cancelled")
            except Exception as exc:
                result = ToolResult(call_id=call_id, name=name, result=str(exc), error=str(exc), status="failed")

            return call_id, self._run_after(ctx, result)

        tasks = [
            asyncio.create_task(_run_one(cid, name, args))
            for cid, name, args in parsed_calls
        ]
        task_to_call_id = {t: cid for t, (cid, _, _) in zip(tasks, parsed_calls)}
        call_id_to_name = {cid: name for cid, name, _ in parsed_calls}

        pending = set(tasks)
        yielded = set()
        while pending:
            done, pending = await asyncio.wait(pending, timeout=0.05, return_when=asyncio.FIRST_COMPLETED)

            if state.cancel_token.is_cancelled():
                for t in pending:
                    t.cancel()
                remaining = set(pending)
                if remaining:
                    await asyncio.gather(*remaining, return_exceptions=True)
                for t in done | remaining:
                    cid = task_to_call_id[t]
                    if cid in yielded:
                        continue
                    yielded.add(cid)
                    yield tool_result_event(
                        id=cid, name=call_id_to_name[cid],
                        result="[用户取消]", status="cancelled",
                    )
                return

            for t in done:
                cid = task_to_call_id[t]
                if cid in yielded:
                    continue
                yielded.add(cid)
                try:
                    call_id, result = await t
                except asyncio.CancelledError:
                    yield tool_result_event(
                        id=cid, name=call_id_to_name[cid],
                        result="[用户取消]", status="cancelled",
                    )
                    continue
                yield tool_result_event(
                    id=call_id, name=result.name, result=result.result,
                    error=result.error, status=result.status,
                )
                if result.cancelled:
                    for t2 in pending:
                        t2.cancel()

    def _run_before(self, ctx: ToolContext) -> ToolContext:
        for mw in self.registry.middlewares:
            ctx = mw.before(ctx)
            if ctx.skipped:
                break
        return ctx

    def _run_after(self, ctx: ToolContext, result: ToolResult) -> ToolResult:
        for mw in reversed(self.registry.middlewares):
            result = mw.after(ctx, result)
        return result

    def parse_tool_call(self, tool_call) -> tuple[str, str, dict | None]:
        raw = tool_call.function.arguments
        try:
            return (tool_call.id, tool_call.function.name, json.loads(raw))
        except json.JSONDecodeError as e:
            logger.warning(
                f"[parse_tool_call] JSON 解析失败: tool={tool_call.function.name}, "
                f"error={e}, len={len(raw)}, raw[:200]={raw[:200]!r}"
            )
            return (tool_call.id, tool_call.function.name, None)

    @staticmethod
    def build_assistant_message(response) -> dict:
        msg: dict = {
            "role": "assistant",
            "content": response.content if response.content else None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in response.tool_calls
            ],
        }
        reasoning = getattr(response, "reasoning", None)
        if reasoning:
            msg["reasoning_content"] = reasoning
        return msg
