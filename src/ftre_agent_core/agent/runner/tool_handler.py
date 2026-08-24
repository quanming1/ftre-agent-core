"""Tool execution, Hook dispatch and concurrent result collection."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ftre_agent_core.event import AgentStreamEvent, EventBase
from ftre_agent_core.hooks import (
    TOOL_AFTER_SPEC,
    TOOL_BEFORE_SPEC,
    HookDispatcher,
    ToolAfterPayload,
    ToolArguments,
    ToolBeforePayload,
    ToolCallIdentity,
    ToolDeny,
    ToolExecutionResult,
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
    """执行单个 Tool，并在原始调用边界直接消费宿主 Hook Dispatcher。

    一次调用的固定顺序是：

    ``tool/before → Core 私有执行 → tool/after``。

    before 可以改变决策/参数，after 可以替换归一化结果；真实 Tool 执行不再
    暴露为 around Hook，避免宿主拥有第二个执行器或重复执行 continuation。
    """

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
        # 没有宿主 Dispatcher 时直接走 Spec.default，保证 Core 可独立运行；
        # 有 Dispatcher 时把作用域交给宿主，Core 不解释 hook_context。
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
        """执行一个 Tool 调用并返回归一化结果。

        ``parse_failed`` 在真正执行前短路，避免把非法 JSON 当成 Tool 参数；
        其余异常被转换为失败 ToolResult，只有外部取消继续以 CancelledError
        语义传播到并发收集器。
        """
        if parse_failed:
            return ToolResult(
                call_id, name,
                "[PARSE_ERROR] Tool call arguments were malformed JSON.",
                error="malformed JSON arguments",
                status="failed",
            )

        cancellation = self._cancellation(state)
        call = self._identity(state, call_id, name)
        # pre Hook 可以拒绝调用或返回替换参数；原始 arguments 不被原地修改，
        # 使同一批并发 Tool 调用之间不会共享可变参数。
        pre = await self._dispatch(
            TOOL_BEFORE_SPEC,
            ToolBeforePayload(call, arguments, cancellation),
        )
        if isinstance(pre, ToolDeny):
            return ToolResult(call_id, name, pre.reason or "Tool denied", pre.reason or "Tool denied", "failed")
        if isinstance(pre, ToolArguments):
            arguments = dict(pre.arguments)

        ctx = ToolContext(call_id=call_id, name=name, arguments=arguments)
        ctx.cancel_token = state.cancel_token
        ctx.metadata["runtime_context"] = state.runtime_context

        async def invoke() -> ToolExecutionResult:
            # 这是 Core 私有执行边界。异步工具直接 await，同步工具放入线程池，
            # 避免阻塞事件循环；宿主只能通过 before/after 观察或改写边界。
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
            execution = await invoke()
            if not isinstance(execution, ToolExecutionResult):
                raise TypeError("Core Tool execution must return ToolExecutionResult")
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
            TOOL_AFTER_SPEC,
            ToolAfterPayload(
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
            # after Hook 只替换“最终展示/状态快照”，不重新执行 Tool。
            result.result = post.output
            result.error = post.error
            result.status = post.status
            result.metadata = dict(post.metadata)
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
