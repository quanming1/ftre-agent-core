"""Tool execution, Hook dispatch and concurrent result collection."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ftre_agent_core.event import AgentStreamEvent, EventBase
from ftre_agent_core.hooks import (
    TOOLS_EXECUTE_SPEC,
    TOOLS_POST_EXECUTE_SPEC,
    TOOLS_PRE_EXECUTE_SPEC,
    TOOLS_RESULT_SPEC,
    HookDispatcher,
    ToolArguments,
    ToolCallIdentity,
    ToolDeny,
    ToolExecutePayload,
    ToolExecutionResult,
    ToolPostExecutePayload,
    ToolPreExecutePayload,
    ToolResultPayload,
)
from ftre_agent_core.llm import ToolCall
from ftre_agent_core.tool import ToolRegistry
from ftre_agent_core.tool.registry import ToolContext
from ftre_agent_core.tracing import RunStatus as TraceRunStatus
from ftre_agent_core.tracing import RunType, TraceSpan

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
    event: AgentStreamEvent | None = None

    @property
    def cancelled(self) -> bool:
        return self.status == "cancelled"


class ToolHandler:
    """执行工具并在原始调用边界直接消费宿主 Hook Dispatcher。"""

    def __init__(
        self,
        registry: ToolRegistry,
        hooks: HookDispatcher | None = None,
        hook_context: object | None = None,
    ) -> None:
        self.registry = registry
        self.hooks = hooks
        self.hook_context = hook_context

    @staticmethod
    def _cancellation(state: RunState) -> asyncio.Event:
        value = state.runtime_context.get("cancellation")
        return value if isinstance(value, asyncio.Event) else asyncio.Event()

    @staticmethod
    def _identity(state: RunState, call_id: str, name: str) -> ToolCallIdentity:
        return ToolCallIdentity(
            call_id=call_id,
            name=name,
            session_id=str(state.runtime_context.get("session_id", "")),
            turn_id=state.turn_id,
            agent_id=str(state.runtime_context.get("agent_id", "")),
            iteration=state.iteration,
        )

    async def _dispatch(self, spec, payload):
        if self.hooks is None:
            result = spec.default(payload) if spec.default else None
            return await result if asyncio.iscoroutine(result) else result
        return await self.hooks.dispatch(spec, payload, context=self.hook_context)

    async def run_one(
        self,
        call_id: str,
        name: str,
        arguments: dict,
        state: RunState,
        parse_failed: bool = False,
    ) -> ToolResult:
        if parse_failed:
            return ToolResult(
                call_id, name,
                "[PARSE_ERROR] Tool call arguments were malformed JSON.",
                error="malformed JSON arguments",
                status="failed",
            )

        cancellation = self._cancellation(state)
        call = self._identity(state, call_id, name)
        pre = await self._dispatch(
            TOOLS_PRE_EXECUTE_SPEC,
            ToolPreExecutePayload(call, arguments, cancellation),
        )
        if isinstance(pre, ToolDeny):
            return ToolResult(call_id, name, pre.reason or "Tool denied", pre.reason or "Tool denied", "failed")
        if isinstance(pre, ToolArguments):
            arguments = dict(pre.arguments)

        ctx = ToolContext(call_id=call_id, name=name, arguments=arguments)
        ctx.cancel_token = state.cancel_token
        ctx.metadata["runtime_context"] = state.runtime_context

        async def invoke() -> ToolExecutionResult:
            tool = self.registry.get(name)
            if tool is not None and tool.is_async():
                resolved = self.registry._resolve_injections(
                    name, ctx.arguments, ctx.metadata.get("runtime_context")
                )
                raw = await tool._get_callable()(**resolved)
            else:
                raw = await asyncio.to_thread(
                    self.registry.execute,
                    name,
                    runtime_context=ctx.metadata.get("runtime_context"),
                    **ctx.arguments,
                )
            if isinstance(raw, EventBase):
                return ToolExecutionResult(value=raw)
            if (
                isinstance(raw, tuple)
                and len(raw) == 2
                and isinstance(raw[0], str)
                and isinstance(raw[1], dict)
            ):
                return ToolExecutionResult(output=raw[0], metadata=raw[1], value=raw)
            return ToolExecutionResult(output=str(raw), value=raw)

        try:
            execution = await self._dispatch(
                TOOLS_EXECUTE_SPEC,
                ToolExecutePayload(call, arguments, cancellation, invoke),
            )
            if not isinstance(execution, ToolExecutionResult):
                raise TypeError("tools/execute must return ToolExecutionResult")
            if execution.status == "cancelled":
                raise asyncio.CancelledError
            if execution.status == "failed":
                raise RuntimeError(execution.error or execution.output or "Tool failed")
            if isinstance(execution.value, EventBase):
                result = ToolResult(call_id, name, "", event=execution.value)
            else:
                result = ToolResult(
                    call_id,
                    name,
                    execution.output,
                    error=execution.error,
                    status=execution.status,
                    metadata=dict(execution.metadata),
                )
        except asyncio.CancelledError:
            result = ToolResult(
                call_id, name,
                "[CANCELLED] Tool execution was cancelled.",
                status="cancelled",
            )
        except Exception as exc:  # noqa: BLE001 - tool failures become ToolResult
            logger.warning("[tool] %s failed: %s", name, exc)
            result = ToolResult(call_id, name, str(exc), error=str(exc), status="failed")

        post = await self._dispatch(
            TOOLS_POST_EXECUTE_SPEC,
            ToolPostExecutePayload(
                call,
                arguments,
                ToolExecutionResult(
                    output=result.result,
                    status=result.status,
                    error=result.error,
                    metadata=result.metadata,
                    value=result.event,
                ),
                cancellation,
            ),
        )
        if isinstance(post, ToolExecutionResult):
            result.result = post.output
            result.error = post.error
            result.status = post.status
            result.metadata = dict(post.metadata)
        await self._dispatch(
            TOOLS_RESULT_SPEC,
            ToolResultPayload(
                call,
                arguments,
                ToolExecutionResult(
                    output=result.result,
                    status=result.status,
                    error=result.error,
                    metadata=result.metadata,
                    value=result.event,
                ),
            ),
        )
        return result

    def spawn(
        self,
        call: ToolCall,
        state: RunState,
        parent_span: TraceSpan | None = None,
    ) -> asyncio.Task:
        span = (
            parent_span.child(
                call.name,
                RunType.TOOL,
                inputs={"arguments": call.input},
                metadata={"call_id": call.id},
            )
            if parent_span
            else None
        )
        return asyncio.create_task(
            self._run_one_traced(
                call_id=call.id,
                name=call.name,
                arguments=call.input if call.input is not None else {},
                state=state,
                parse_failed=call.input is None,
                span=span,
            ),
            name=f"tool-{call.id}",
        )

    async def _run_one_traced(
        self,
        *,
        call_id: str,
        name: str,
        arguments: dict,
        state: RunState,
        parse_failed: bool,
        span: TraceSpan | None,
    ) -> ToolResult:
        try:
            result = await self.run_one(call_id, name, arguments, state, parse_failed)
        except BaseException as exc:
            if span and not span.ended:
                span.end(status=TraceRunStatus.CANCELLED if isinstance(exc, asyncio.CancelledError) else TraceRunStatus.ERROR)
            raise
        if span and not span.ended:
            status = {
                "cancelled": TraceRunStatus.CANCELLED,
                "failed": TraceRunStatus.ERROR,
            }.get(result.status, TraceRunStatus.COMPLETED)
            span.end(
                status=status,
                error=result.error if result.status == "failed" else None,
                outputs={"result": result.result, "status": result.status, "error": result.error},
            )
        return result

    @staticmethod
    async def drain(tasks: dict[str, asyncio.Task]) -> None:
        for task in tasks.values():
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks.values(), return_exceptions=True)

    async def gather_results(
        self,
        tool_calls: list[ToolCall],
        tasks: dict[str, asyncio.Task],
        state: RunState,
    ) -> tuple[list[ToolResult], bool]:
        if state.is_cancelled:
            for task in tasks.values():
                task.cancel()
        cancelled_externally = False
        try:
            raw = await asyncio.gather(*tasks.values(), return_exceptions=True)
        except asyncio.CancelledError:
            for task in tasks.values():
                task.cancel()
            raw = await asyncio.gather(*tasks.values(), return_exceptions=True)
            cancelled_externally = True

        finished: dict[str, ToolResult] = {}
        for call_id, item in zip(tasks, raw):
            if isinstance(item, BaseException):
                interrupted = state.is_cancelled or cancelled_externally
                finished[call_id] = ToolResult(
                    call_id,
                    next((call.name for call in tool_calls if call.id == call_id), call_id),
                    "[INTERRUPTED] Tool execution was interrupted.",
                    status="cancelled" if interrupted else "failed",
                )
            else:
                finished[call_id] = item
        results = [
            finished.get(call.id)
            or ToolResult(call.id, call.name, "[INTERRUPTED] Tool result was lost.", status="failed")
            for call in tool_calls
        ]
        any_cancelled = any(result.status == "cancelled" for result in results)
        return results, cancelled_externally or any_cancelled or state.is_cancelled
