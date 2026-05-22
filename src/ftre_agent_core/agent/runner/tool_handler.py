"""
ToolHandler - 工具调用处理器

职责：
- 解析 LLM 返回的 tool_call JSON
- 执行工具（支持并行、取消、中间件）
- 派发事件（tool_call / tool_result）
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator

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
    """工具执行结果"""
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
        from ftre_agent_core.threading import thread_pool
        self._executor = thread_pool.tool

    # =====================================================================
    #  核心：统一执行入口
    # =====================================================================

    def execute(
        self,
        parsed_calls: list[tuple[str, str, dict]],
        state: "RunState",
    ) -> Generator[AgentEvent, None, list[ToolResult]]:
        """
        执行一个或多个工具调用，统一处理并行、取消、中间件。

        流程：
        1. yield 所有 tool_call 事件
        2. 对每个工具：before 中间件 → 提交到线程池
        3. 主线程轮询：检查取消 + 收割完成的 future → yield tool_result
        """
        # 1. yield 所有 tool_call 事件
        for call_id, name, arguments in parsed_calls:
            yield tool_call_event(id=call_id, name=name, arguments=arguments)

        # 2. 构建 context + before 中间件 + 提交
        contexts: dict[str, ToolContext] = {}
        futures: dict[Future, str] = {}
        results: dict[str, ToolResult] = {}

        for call_id, name, arguments in parsed_calls:
            ctx = ToolContext(call_id=call_id, name=name, arguments=arguments)
            ctx.cancel_token = state.cancel_token
            ctx.metadata["runtime_context"] = state.runtime_context
            ctx = self._run_before(ctx)
            contexts[call_id] = ctx

            if ctx.skipped:
                result = ToolResult(call_id=call_id, name=name, result=ctx.skip_result)
                results[call_id] = self._run_after(ctx, result)
                yield tool_result_event(
                    id=call_id, name=name, result=results[call_id].result,
                    error=results[call_id].error, status=results[call_id].status,
                )
            else:
                futures[self._executor.submit(self._invoke, ctx)] = call_id

        # 3. 主线程轮询
        pending = set(futures.keys())

        while pending:
            if state.wait_or_cancelled(0.05):
                for f in pending:
                    f.cancel()
                for f in pending:
                    cid = futures[f]
                    ctx = contexts[cid]
                    result = ToolResult(call_id=cid, name=ctx.name, result="[用户取消]", status="cancelled")
                    results[cid] = self._run_after(ctx, result)
                    yield tool_result_event(id=cid, name=ctx.name, result=result.result, status="cancelled")
                break

            done = {f for f in pending if f.done()}
            for f in done:
                pending.discard(f)
                cid = futures[f]
                ctx = contexts[cid]
                try:
                    raw = f.result()
                    result = ToolResult(call_id=cid, name=ctx.name, result=str(raw))
                except Exception as exc:
                    result = ToolResult(call_id=cid, name=ctx.name, result=str(exc), error=str(exc), status="failed")
                results[cid] = self._run_after(ctx, result)
                yield tool_result_event(
                    id=cid, name=ctx.name, result=results[cid].result,
                    error=results[cid].error, status=results[cid].status,
                )

        return [results[cid] for cid, _, _ in parsed_calls if cid in results]

    # =====================================================================
    #  内部方法
    # =====================================================================

    def _invoke(self, ctx: ToolContext) -> str:
        """在子线程中执行工具函数。"""
        return self.registry.execute(ctx.name, runtime_context=ctx.metadata.get("runtime_context"), **ctx.arguments)

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

    # =====================================================================
    #  消息解析
    # =====================================================================

    def parse_tool_call(self, tool_call) -> tuple[str, str, dict | None]:
        """解析 tool_call JSON。失败返回 (id, name, None)。"""
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
        return {
            "role": "assistant",
            "content": response.content,
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
