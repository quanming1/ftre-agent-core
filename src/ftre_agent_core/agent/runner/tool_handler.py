"""
工具执行器、并发调度器和 assistant 消息构造器。

本模块职责：
  - run_one(): 执行单个工具调用，返回 ToolResult。
  - spawn(): 为一个 ToolCall 创建并发执行任务，不阻塞 LLM 流消费。
  - drain(): 取消并回收一组工具任务（用于异常清理）。
  - gather_results(): 等待全部工具任务、处理取消、按 tool_calls 顺序归并结果。
  - build_assistant_message(): 根据 ToolCall 列表构造写入 memory 的 assistant 消息。
  - 执行工具中间件 before / after 链。

本模块不负责：
  - 决定何时向调用方 yield 事件。
  - 写入 memory。
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ftre_agent_core.llm import ToolCall
from ftre_agent_core.tool import ToolRegistry
from ftre_agent_core.tool.registry import ToolContext
from ftre_agent_core.agent.event import AgentEvent
from ftre_agent_core.reasoning import preserve_tool_call_reasoning
from ftre_agent_core.tracing import RunStatus as TraceRunStatus, RunType, TraceSpan

if TYPE_CHECKING:
    from .react_runner import RunState

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """单个工具调用的执行结果。"""
    call_id: str
    name: str
    result: str
    error: str | None = None
    status: str = "completed"   # completed / failed / cancelled
    metadata: dict = field(default_factory=dict)
    event: AgentEvent | None = None  # 工具返回了 AgentEvent（非 str）时设此字段

    @property
    def cancelled(self) -> bool:
        return self.status == "cancelled"


class ToolHandler:

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    # 执行单个工具调用。
    async def run_one(
        self,
        call_id: str,
        name: str,
        arguments: dict,
        state: "RunState",
        parse_failed: bool = False,
    ) -> ToolResult:
        """执行一个工具调用并返回 ToolResult。

        这个方法预期由 asyncio.create_task() 调用，这样多个工具可以并发执行。
        parse_failed=True 表示模型输出的工具参数 JSON 格式错误，直接返回失败结果。
        """
        if parse_failed:
            return ToolResult(
                call_id=call_id,
                name=name,
                result="[PARSE_ERROR] Tool call arguments were malformed JSON.",
                error="malformed JSON arguments",
                status="failed",
            )

        ctx = ToolContext(call_id=call_id, name=name, arguments=arguments)
        ctx.cancel_token = state.cancel_token
        ctx.metadata["runtime_context"] = state.runtime_context

        # 执行 before 中间件链。
        ctx = self._run_before(ctx)

        if ctx.skipped:
            result = ToolResult(call_id=call_id, name=name, result=ctx.skip_result)
            return self._run_after(ctx, result)

        try:
            tool = self.registry.get(name)
            if tool is not None and tool.is_async():
                # 异步工具必须直接 await 底层协程函数，不能走 Tool.execute()。
                # Tool.execute() 面向同步调用方，会使用 asyncio.run()。
                raw = await tool._get_callable()(**ctx.arguments)
            else:
                raw = await asyncio.to_thread(
                    self.registry.execute,
                    name,
                    runtime_context=ctx.metadata.get("runtime_context"),
                    **ctx.arguments,
                )
            if isinstance(raw, AgentEvent):
                result = ToolResult(call_id=call_id, name=name, result="", event=raw)
            else:
                result = ToolResult(call_id=call_id, name=name, result=str(raw))
        except asyncio.CancelledError:
            result = ToolResult(
                call_id=call_id, name=name,
                result="[CANCELLED] Tool execution was cancelled.",
                status="cancelled",
            )
        except Exception as exc:
            logger.warning("[tool] %s failed: %s", name, exc)
            result = ToolResult(
                call_id=call_id, name=name,
                result=str(exc), error=str(exc), status="failed",
            )

        return self._run_after(ctx, result)

    # ── 并发调度 ──────────────────────────────────────────────
    def spawn(
        self,
        call: ToolCall,
        state: "RunState",
        parent_span: TraceSpan | None = None,
    ) -> asyncio.Task:
        """为一个 ToolCall 创建并发执行任务。立即返回，不在此 await。"""
        span = parent_span.child(
            call.name,
            RunType.TOOL,
            inputs={"arguments": call.input},
            metadata={"call_id": call.id},
        ) if parent_span else None
        return asyncio.create_task(
            self._run_one_traced(
                call_id=call.id,
                name=call.name,
                arguments=call.input if call.input is not None else {},
                state=state,
                parse_failed=(call.input is None),
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
        state: "RunState",
        parse_failed: bool,
        span: TraceSpan | None,
    ) -> ToolResult:
        try:
            result = await self.run_one(
                call_id=call_id,
                name=name,
                arguments=arguments,
                state=state,
                parse_failed=parse_failed,
            )
        except BaseException as exc:
            if span and not span.ended:
                if isinstance(exc, asyncio.CancelledError):
                    span.end(status=TraceRunStatus.CANCELLED)
                else:
                    span.end(error=exc)
            raise

        if span and not span.ended:
            status = (
                TraceRunStatus.CANCELLED
                if result.status == "cancelled"
                else TraceRunStatus.ERROR
                if result.status == "failed"
                else TraceRunStatus.COMPLETED
            )
            span.end(
                status=status,
                error=result.error if result.status == "failed" else None,
                outputs={
                    "result": result.result,
                    "status": result.status,
                    "error": result.error,
                },
            )
        return result

    @staticmethod
    async def drain(tasks: dict[str, asyncio.Task]) -> None:
        """取消并等待一组工具任务，用于异常路径的清理。"""
        for t in tasks.values():
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks.values(), return_exceptions=True)

    async def gather_results(
        self,
        tool_calls: list[ToolCall],
        tasks: dict[str, asyncio.Task],
        state: "RunState",
    ) -> tuple[list[ToolResult], bool]:
        """等待全部工具任务完成，按 tool_calls 顺序返回 (results, cancelled)。

        cancelled 为 True 表示发生了外部取消或有工具被取消，调用方写完 memory
        后应抛出 CancelledError。任务异常会被归一成 INTERRUPTED 的 ToolResult。
        """
        if state.is_cancelled:
            for t in tasks.values():
                t.cancel()

        cancelled_externally = False
        try:
            raw = await asyncio.gather(*tasks.values(), return_exceptions=True)
        except asyncio.CancelledError:
            for t in tasks.values():
                t.cancel()
            raw = await asyncio.gather(*tasks.values(), return_exceptions=True)
            cancelled_externally = True

        finished: dict[str, ToolResult] = {}
        for call_id, item in zip(tasks.keys(), raw):
            if isinstance(item, BaseException):
                interrupted = state.is_cancelled or cancelled_externally
                finished[call_id] = ToolResult(
                    call_id=call_id,
                    name=next((c.name for c in tool_calls if c.id == call_id), call_id),
                    result="[INTERRUPTED] Tool execution was interrupted.",
                    status="cancelled" if interrupted else "failed",
                )
            else:
                finished[call_id] = item

        results = [
            finished.get(c.id) or ToolResult(
                call_id=c.id, name=c.name,
                result="[INTERRUPTED] Tool result was lost.", status="failed",
            )
            for c in tool_calls
        ]
        any_cancelled = any(r.status == "cancelled" for r in results)
        return results, (cancelled_externally or any_cancelled or state.is_cancelled)

    # 构造带 tool_calls 的 assistant 消息。
    @staticmethod
    def build_assistant_message(
        tool_calls: list[ToolCall],
        content: str | None = None,
        reasoning: str | None = None,
    ) -> dict:
        """构造要写入 memory.add_raw() 的 assistant 原始消息。

        assistant 消息只包含 tool_calls；对应的 role="tool" 结果消息由
        react_runner 在所有工具完成后统一写入。
        """
        msg: dict = {
            "role": "assistant",
            "content": content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.input, ensure_ascii=False)
                        if tc.input is not None else "{}",
                    },
                }
                for tc in tool_calls
            ],
        }
        preserve_tool_call_reasoning(msg, reasoning)
        return msg

    # 工具中间件
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
