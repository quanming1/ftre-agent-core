"""
ToolHandler - 工具调用处理器

职责：
- 解析 LLM 返回的 tool_call JSON
- 在子线程中执行工具，主线程轮询取消信号
- 中间件链：before → execute → after
- 构建 assistant message
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator

from ftre_agent_core.agent.event import (
    AgentEvent,
    tool_call_event,
    tool_result_event,
)
from ftre_agent_core.tool import ToolRegistry
from ftre_agent_core.tool.middleware import ToolContext
from ftre_agent_core.tool_system import ToolCancelledError

if TYPE_CHECKING:
    from ..state import RunState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  ToolResult
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
#  ToolHandler
# ---------------------------------------------------------------------------

class ToolHandler:

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        from ftre_agent_core.threading import thread_pool
        self._executor = thread_pool.tool

    # =====================================================================
    #  执行核心
    # =====================================================================

    def _execute(self, ctx: ToolContext) -> str:
        """在子线程中执行工具，返回字符串结果。"""
        return self.registry.execute(ctx.name, **ctx.arguments)

    def _run_middlewares_before(self, ctx: ToolContext) -> ToolContext:
        for mw in self.registry.middlewares:
            ctx = mw.before(ctx)
            if ctx.skipped:
                break
        return ctx

    def _run_middlewares_after(self, ctx: ToolContext, result: ToolResult) -> ToolResult:
        for mw in reversed(self.registry.middlewares):
            result = mw.after(ctx, result)
        return result

    # =====================================================================
    #  可取消执行
    # =====================================================================

    def execute_cancellable(
        self,
        call_id: str,
        name: str,
        arguments: dict,
        state: "RunState",
    ) -> ToolResult:
        """
        执行工具，支持取消。

        流程：middleware.before → 子线程执行(轮询取消) → middleware.after
        无论成功、失败、取消，都返回 ToolResult，不抛异常。
        """
        from ..state import CancelledError

        ctx = ToolContext(call_id=call_id, name=name, arguments=arguments)
        ctx.cancel_token = state.cancel_token
        ctx = self._run_middlewares_before(ctx)

        # 中间件短路
        if ctx.skipped:
            result = ToolResult(call_id=call_id, name=name, result=ctx.skip_result)
            return self._run_middlewares_after(ctx, result)

        # 提交到线程池执行
        future = self._executor.submit(self._execute, ctx)

        # 轮询等待，期间检查取消
        raw, error = "", None
        try:
            while not future.done():
                if state.wait_or_cancelled(0.1):
                    future.cancel()
                    raise CancelledError()
            raw = future.result()
        except CancelledError:
            error = "cancelled"
            raw = "[用户取消]"
        except Exception as exc:
            error = str(exc)
            raw = str(exc)

        # 构建结果
        status = "cancelled" if error == "cancelled" else ("failed" if error else "completed")
        result = ToolResult(
            call_id=call_id,
            name=name,
            result=str(raw),
            error=error if error != "cancelled" else None,
            status=status,
            metadata=dict(ctx.metadata),
        )
        return self._run_middlewares_after(ctx, result)

    # =====================================================================
    #  事件派发 API
    # =====================================================================

    def execute_and_emit(
        self,
        call_id: str,
        name: str,
        arguments: dict,
        state: "RunState",
    ) -> Generator[AgentEvent, None, ToolResult]:
        """执行工具 + yield 事件流（tool_call → tool_result）"""
        yield tool_call_event(id=call_id, name=name, arguments=arguments)
        result = self.execute_cancellable(call_id, name, arguments, state)
        yield tool_result_event(
            id=call_id, name=name, result=result.result,
            error=result.error, status=result.status,
            metadata=result.metadata,
        )
        return result

    def execute_parallel(
        self,
        parsed_calls: list[tuple[str, str, dict]],
        state: "RunState",
    ) -> Generator[AgentEvent, None, list[ToolResult]]:
        """并行执行多个工具，谁先完成谁先 yield 事件。"""

        # 1. yield 所有 tool_call 事件
        for call_id, name, arguments in parsed_calls:
            yield tool_call_event(id=call_id, name=name, arguments=arguments)

        # 2. 并发执行
        results_map: dict[str, ToolResult] = {}

        with ThreadPoolExecutor(max_workers=len(parsed_calls), thread_name_prefix="ftre-parallel") as pool:
            futures = {
                pool.submit(self.execute_cancellable, call_id, name, arguments, state): (call_id, name)
                for call_id, name, arguments in parsed_calls
            }

            # 3. 谁先完成谁先 yield
            for future in as_completed(futures):
                call_id, name = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = ToolResult(call_id=call_id, name=name, result=str(e), error=str(e), status="failed")
                results_map[call_id] = result
                yield tool_result_event(
                    id=call_id, name=name, result=result.result,
                    error=result.error, status=result.status,
                    metadata=result.metadata,
                )

        return [results_map[cid] for cid, _, _ in parsed_calls]

    # =====================================================================
    #  消息解析
    # =====================================================================

    def parse_tool_call(self, tool_call) -> tuple[str, str, dict | None]:
        """
        解析 tool_call 的 JSON 参数。

        Returns:
            成功: (call_id, tool_name, arguments)
            失败: (call_id, tool_name, None)
        """
        raw = tool_call.function.arguments
        try:
            args = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"[parse_tool_call] JSON 解析失败: tool={tool_call.function.name}, error={e}")
            return (tool_call.id, tool_call.function.name, None)
        return (tool_call.id, tool_call.function.name, args)

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
