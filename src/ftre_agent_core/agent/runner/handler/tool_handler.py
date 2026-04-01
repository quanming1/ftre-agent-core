from __future__ import annotations

"""
ToolHandler - 工具调用处理器

职责：
- 解析 LLM 返回的 tool_call
- 在子线程中执行工具，主线程轮询取消信号
- 中间件链：before → execute → after
- 构建 assistant message（含 tool_calls）

执行模型：
    所有工具统一在 ThreadPoolExecutor 子线程中执行。
    - sync  工具：子线程直接调用
    - async 工具：子线程通过 run_coroutine_threadsafe 提交回主事件循环
      （Motor 等异步驱动绑定在主循环，不能 asyncio.run 新建循环）

    主线程（调用方）通过 _poll_or_cancel 轮询取消信号，
    取消时通过 CancellationToken 通知子线程中的工具。
"""

import json
import logging
import threading
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generator

from ftre_agent_core.agent.event import (
    AgentEvent,
    tool_call_event,
    tool_result_event,
)
from ftre_agent_core.tool import ToolRegistry
from ftre_agent_core.tool.middleware import ToolContext
from ftre_agent_core.tool_system import (
    ToolCancelledError,
    ToolExecutionHandle,
    ToolOutput,
    ToolResult as StandardToolResult,
)

if TYPE_CHECKING:
    from ..state import RunState

logger = logging.getLogger(__name__)


class _DefaultOutputGuard:
    """默认的输出过滤器（不做任何过滤）"""
    def sanitize_tool_result(self, tool_name: str, result: str) -> str:
        return result


def _get_output_guard():
    """获取输出过滤器，优先使用外部注入的实现"""
    try:
        from packages.workspace import get_output_guard
        return get_output_guard()
    except ImportError:
        return _DefaultOutputGuard()


# ---------------------------------------------------------------------------
#  ToolResult — 面向上层的轻量结果
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """工具执行结果（上层消费的简化版本）。"""
    call_id: str
    name: str
    result: str
    error: str | None = None
    status: str = "completed"

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def timed_out(self) -> bool:
        return self.status == "timed_out"

    @property
    def cancelled(self) -> bool:
        return self.status == "cancelled"


# ---------------------------------------------------------------------------
#  ToolHandler
# ---------------------------------------------------------------------------

class ToolHandler:

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        # 使用全局线程池，不再每个 agent 各建一个（避免 N*8 线程爆炸）
        from ftre_agent_core.threading import thread_pool
        self._executor = thread_pool.tool
        self._output_guard = _get_output_guard()
        self._active_handles: dict[str, ToolExecutionHandle] = {}
        self._context_local = threading.local()
        self.registry.provide("tool_context", self._get_current_context)

    # =====================================================================
    #  1. 中间件 & thread-local
    # =====================================================================

    def _get_current_context(self) -> ToolContext | None:
        return getattr(self._context_local, "context", None)

    def _set_current_context(self, ctx: ToolContext | None) -> None:
        self._context_local.context = ctx

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
    #  2. 子线程执行（核心）
    # =====================================================================

    def _invoke(self, ctx: ToolContext) -> Any:
        """
        在子线程中执行工具函数。

        sync  → 直接调用
        async → run_coroutine_threadsafe 提交回主事件循环
        """
        tool = self.registry.get(ctx.name)
        if tool is None:
            raise ValueError(f"Tool '{ctx.name}' not found")

        if not tool.is_async():
            return self.registry.execute(ctx.name, **ctx.arguments)

        # async 工具：提交回主事件循环
        from packages.storage.database import get_event_loop
        main_loop = get_event_loop()
        if main_loop is not None and main_loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(
                self.registry.execute_async(ctx.name, **ctx.arguments),
                main_loop,
            )
            return fut.result()
        # fallback：无主循环（单元测试等）
        return asyncio.run(
            self.registry.execute_async(ctx.name, **ctx.arguments)
        )

    def _run_in_thread(self, ctx: ToolContext) -> Any:
        """子线程入口：设置 thread-local context → 执行 → 清理。"""
        self._set_current_context(ctx)
        try:
            return self._invoke(ctx)
        finally:
            self._set_current_context(None)

    # =====================================================================
    #  3. 取消轮询
    # =====================================================================

    @staticmethod
    def _poll_or_cancel(
        future: Future,
        state: "RunState",
        handle: ToolExecutionHandle,
        interval: float = 0.1,
    ) -> Any:
        """
        等待 future 完成，期间轮询取消信号。

        取消时通知 handle（进而通知 CancellationToken），
        然后抛 CancelledError 给调用方。
        """
        from ..state import CancelledError
        while not future.done():
            if state.wait_or_cancelled(interval):
                handle.request_cancel("user_cancelled")
                raise CancelledError()
        return future.result()

    # =====================================================================
    #  4. 结果构建
    # =====================================================================

    def _sanitize(self, name: str, raw: Any) -> str:
        return self._output_guard.sanitize_tool_result(name, str(raw))

    def _build_result(
        self,
        ctx: ToolContext,
        handle: ToolExecutionHandle,
        raw: Any,
        error: Exception | None,
    ) -> ToolResult:
        """
        raw + error → StandardToolResult → handle.finish → ToolResult → after 中间件

        这是结果构建的唯一入口，不管工具成功、失败、取消、超时，都走这里。
        """
        preview = self._sanitize(handle.name, raw)
        output = ToolOutput(preview=preview, channels={"result": preview})

        # 根据异常类型决定 status
        if isinstance(error, ToolCancelledError):
            std = StandardToolResult.cancelled(str(error))
        elif isinstance(error, TimeoutError):
            std = StandardToolResult.timed_out(str(error))
        elif error is not None:
            std = StandardToolResult.failed("tool_execution_failed", str(error))
        else:
            std = StandardToolResult.completed(value=raw, output=output)
        std.output = output

        handle.finish(std)

        # StandardToolResult → 轻量 ToolResult
        text = preview or (
            std.error.message if std.error else ""
        )
        compat = ToolResult(
            call_id=ctx.call_id,
            name=ctx.name,
            result=text,
            error=std.error.message if std.error else None,
            status=std.status,
        )
        return self._run_after(ctx, compat)

    # =====================================================================
    #  5. 对外 API
    # =====================================================================

    def execute_cancellable(
        self,
        call_id: str,
        name: str,
        arguments: dict,
        state: "RunState",
    ) -> ToolResult:
        """
        执行工具（生产路径）。

        流程：before → 子线程执行(轮询取消) → 构建结果 → after → 清理

        无论成功、失败、取消、超时，都返回 ToolResult，不抛异常。
        调用方通过 result.cancelled / result.timed_out 判断状态。
        """
        from ..state import CancelledError

        ctx = ToolContext(call_id=call_id, name=name, arguments=arguments)
        ctx.cancel_token = state.cancel_token
        ctx = self._run_before(ctx)

        # 中间件 skip
        if ctx.skipped:
            return self._run_after(
                ctx,
                ToolResult(call_id=call_id, name=name,
                           result=ctx.skip_result, status="completed"),
            )

        # 创建 handle，提交子线程
        handle = ToolExecutionHandle(call_id=call_id, name=name)
        handle.cancel_token = ctx.cancel_token
        handle.resources = ctx.resources
        self._active_handles[call_id] = handle
        handle.transition_to(handle.status.RUNNING)

        future = self._executor.submit(self._run_in_thread, ctx)

        raw, error = "", None
        try:
            raw = self._poll_or_cancel(future, state, handle)
        except CancelledError:
            logger.info(f"[tool] {name} 被取消")
            error, raw = ToolCancelledError("user_cancelled"), "[用户取消]"
        except Exception as exc:
            error, raw = exc, str(exc)

        result = self._build_result(ctx, handle, raw, error)

        # 清理
        self._active_handles.pop(call_id, None)
        handle.resources.cleanup_all()

        return result

    def execute_and_emit(
        self,
        call_id: str,
        name: str,
        arguments: dict,
        state: "RunState",
    ) -> Generator[AgentEvent, None, ToolResult]:
        """
        执行工具 + yield 完整事件流。

        事件流保证完整：tool_call → (执行) → tool_result。
        无论工具成功、失败还是取消，tool_result 都会被 yield。
        """
        yield tool_call_event(id=call_id, name=name, arguments=arguments)

        logger.info(f"[execute_and_emit] 开始执行工具 {name} (call_id={call_id})")
        result = self.execute_cancellable(call_id, name, arguments, state)

        yield tool_result_event(
            id=call_id, name=name, result=result.result,
            error=result.error, status=result.status,
        )
        return result

    def execute_parallel(
        self,
        parsed_calls: list[tuple[str, str, dict]],
        state: "RunState",
    ) -> Generator[AgentEvent, None, list[ToolResult]]:
        """串行 execute_and_emit 多个工具（当前未真正并行，保留接口语义）。"""
        results: list[ToolResult] = []
        for call_id, name, arguments in parsed_calls:
            result = yield from self.execute_and_emit(
                call_id=call_id, name=name, arguments=arguments, state=state,
            )
            results.append(result)
        return results

    # =====================================================================
    #  6. 消息解析（纯工具函数）
    # =====================================================================

    def parse_tool_call(self, tool_call) -> tuple[str, str, dict]:
        """
        解析 tool_call 的 JSON 参数。

        Returns:
            成功: (call_id, tool_name, arguments)
            失败: (call_id, tool_name, None)  # arguments=None 表示解析失败

        调用方检查 arguments is None 来判断是否解析失败，
        失败时应跳过执行并返回错误结果给 LLM，让其在下一轮重试。
        """
        raw = tool_call.function.arguments
        try:
            args = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(
                f"[parse_tool_call] JSON 解析失败（可能是 streaming 截断）: "
                f"tool={tool_call.function.name}, call_id={tool_call.id}, "
                f"error={e}, raw_len={len(raw)}, raw[:500]={raw[:500]!r}"
            )
            # 返回 None 作为 arguments，让调用方处理
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
