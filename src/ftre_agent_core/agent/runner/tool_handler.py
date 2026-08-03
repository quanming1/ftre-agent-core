"""
工具执行器和并发调度器。

本模块职责：
  - run_one(): 执行单个工具调用，返回 ToolResult。
  - spawn(): 为一个 ToolCall 创建并发执行任务，不阻塞 LLM 流消费。
  - drain(): 取消并回收一组工具任务（用于异常清理）。
  - gather_results(): 等待全部工具任务、处理取消、按 tool_calls 顺序归并结果。
  - on_pre_tool / on_post_tool hook 集成。

本模块不负责：
  - 决定何时向调用方 yield 事件。
  - 写入 memory。
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ftre_agent_core.llm import ToolCall
from ftre_agent_core.tool import ToolRegistry
from ftre_agent_core.tool.registry import ToolContext
from ftre_agent_core.event import AgentStreamEvent, EventBase
from ftre_agent_core.tracing import RunStatus as TraceRunStatus, RunType, TraceSpan

if TYPE_CHECKING:
    from .react_runner import RunState
    from ..hooks import FtreCoreHookManager

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """单个工具调用的执行结果。

    每一次 ``ToolHandler.run_one`` 执行完毕后都会产出一个 ``ToolResult``，
    它会被 ``gather_results`` 收集并按 ``tool_calls`` 顺序返回给 react_runner，
    最终由 react_runner 转成 role="tool" 的消息写入 memory。

    字段说明：
      - call_id: 对应 LLM 输出 ToolCall 的 id，用于把结果与调用一一配对。
      - name: 工具名称，便于日志与 tracing。
      - result: 工具返回的可读字符串，将原样作为 tool 消息 content 回传给 LLM。
      - error: 失败时的错误信息（成功时为 None）。
      - status: 执行状态，取值 completed / failed / cancelled。
      - metadata: 工具返回的附加元数据（例如 edit/write 的 diff 信息）。
      - event: 当工具返回的是 EventBase（事件对象而非纯字符串）时存放于此，
        表示该结果不是给 LLM 的文本，而是一个需要向上冒泡的流事件。
    """
    call_id: str
    name: str
    result: str
    error: str | None = None
    status: str = "completed"   # completed / failed / cancelled
    metadata: dict = field(default_factory=dict)
    event: AgentStreamEvent | None = None  # 工具返回了 EventBase（非 str）时设此字段

    @property
    def cancelled(self) -> bool:
        """便捷判断：该结果是否为被取消状态。"""
        return self.status == "cancelled"


class ToolHandler:
    """工具执行处理器。

    负责把 LLM 输出的 ToolCall 真正派发给已注册的工具并收集结果，
    同时串联 on_pre_tool / on_post_tool 两个 hook，允许外部拦截或改写工具调用。
    一个 ToolHandler 实例通常在整个 agent 运行期共享。
    """

    def __init__(self, registry: ToolRegistry, hook_manager: "FtreCoreHookManager | None" = None):
        """初始化工具处理器。

        参数：
          - registry: 工具注册表，提供按名查找工具、解析依赖注入、同步执行工具的能力。
          - hook_manager: 可选的 hook 管理器；为 None 时跳过所有 on_pre_tool / on_post_tool hook。
        """
        self.registry = registry
        self.hook_manager = hook_manager

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

        调用时机：
          通常由 ``spawn`` 包装成 ``_run_one_traced`` 后通过 ``asyncio.create_task``
          并发调度；多个工具调用会同时运行、互不阻塞。也可直接 await 调用。

        参数：
          - call_id: LLM 输出的工具调用 id，用于结果配对。
          - name: 工具名称（必须在 registry 中已注册，否则视为失败）。
          - arguments: 工具参数字典，已从 LLM 输出的 JSON 字符串解析而来。
          - state: 当前 ReAct 循环的运行状态（RunState），提供 session_id、turn_id、
            iteration、cancel_token、runtime_context 等。
          - parse_failed: 为 True 表示 LLM 输出的工具参数 JSON 格式错误，
            无法解析成 dict；此时跳过实际执行，直接返回失败的 ToolResult。

        返回：
          一个 ToolResult，描述成功/失败/被取消及其结果文本。
        """
        # —— 参数解析失败短路 ——
        # LLM 有时会输出格式错误的 JSON 参数，此时上层会把 parse_failed=True 传入，
        # 这里直接构造一个失败结果返回给 LLM，让它看到错误并自我纠正，而不去执行工具。
        if parse_failed:
            return ToolResult(
                call_id=call_id,
                name=name,
                result="[PARSE_ERROR] Tool call arguments were malformed JSON.",
                error="malformed JSON arguments",
                status="failed",
            )

        # ── on_pre_tool hook ──
        # 在真正执行工具前给 hook 一次机会拦截或改写本次调用。
        # hook 可返回三种决策：
        #   - block: 拦截，不执行工具，直接以 hook 给出的 reason 作为失败结果返回。
        #   - modify: 改写参数，用 hook 返回的 modified_args 替换原始 arguments 后再执行。
        #   - (其他/None): 放行，按原始参数继续执行。
        if self.hook_manager is not None:
            from ...hooks import ON_PRE_TOOL, PreToolInput, PreToolOutput
            pre_output = await self.hook_manager.trigger(
                ON_PRE_TOOL,
                lambda: PreToolInput(
                    session_id=state.runtime_context.get("session_id", ""),
                    turn_id=state.turn_id,
                    iteration=state.iteration,
                    tool_call_id=call_id,
                    tool_name=name,
                    tool_args=arguments,
                    runtime_context=state.runtime_context,
                ),
            )
            if pre_output is not None:
                # block：直接拦截，不执行工具，返回失败结果
                if pre_output.decision == "block":
                    return ToolResult(
                        call_id=call_id,
                        name=name,
                        result=pre_output.reason or "被 Hook 拦截",
                        error=pre_output.reason or "被 Hook 拦截",
                        status="failed",
                    )
                # modify：用 hook 改写后的参数替换原始参数，后续执行使用新参数
                if pre_output.decision == "modify" and isinstance(pre_output, PreToolOutput):
                    if pre_output.modified_args is not None:
                        arguments = pre_output.modified_args

        # 构造工具执行上下文 ToolContext，把 call_id、参数、取消令牌、runtime_context
        # 打包传给底层工具，工具可通过 ctx.cancel_token 响应取消、通过 runtime_context 拿到 session 信息。
        ctx = ToolContext(call_id=call_id, name=name, arguments=arguments)
        ctx.cancel_token = state.cancel_token
        ctx.metadata["runtime_context"] = state.runtime_context

        try:
            tool = self.registry.get(name)
            # —— 异步工具 vs 同步工具的执行分支 ——
            # 异步工具（is_async()=True）原生是协程函数，直接 await 它的底层 callable，
            # 这样它内部可以正常使用 asyncio 原语（如 asyncio.sleep、await 网络请求）；
            # 但需要先调用 _resolve_injections 解析 @Injected 依赖注入占位符。
            # 同步工具则用 asyncio.to_thread 丢到线程池执行，避免阻塞事件循环；
            # registry.execute 内部会处理依赖注入与异常包装。
            if tool is not None and tool.is_async():
                # 异步工具直接 await 底层协程函数，但需要先解析 Injected 依赖注入。
                resolved_kwargs = self.registry._resolve_injections(
                    name, ctx.arguments, ctx.metadata.get("runtime_context"),
                )
                raw = await tool._get_callable()(**resolved_kwargs)
            else:
                raw = await asyncio.to_thread(
                    self.registry.execute,
                    name,
                    runtime_context=ctx.metadata.get("runtime_context"),
                    **ctx.arguments,
                )
            # —— 结果类型归一化 ——
            # 工具可能返回三种形态，统一包装成 ToolResult：
            #   1) EventBase：工具返回的是事件对象（非给 LLM 的文本），存入 event 字段，
            #      result 置空，由上层把事件向上冒泡。
            #   2) (str, dict) 二元组：工具同时返回结果文本和元数据（如 edit/write 的 diff），
            #      分别填入 result 和 metadata。
            #   3) 其他：直接 str() 转成文本结果。
            if isinstance(raw, EventBase):
                result = ToolResult(call_id=call_id, name=name, result="", event=raw)
            elif isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], str) and isinstance(raw[1], dict):
                # 工具返回 (result_str, metadata) — edit/write 返回 diff metadata
                result = ToolResult(call_id=call_id, name=name, result=raw[0], metadata=raw[1])
            else:
                result = ToolResult(call_id=call_id, name=name, result=str(raw))
        except asyncio.CancelledError:
            # —— 取消（CancelledError）处理 ——
            # 工具执行过程中被外部 cancel（如用户中断、drain 清理）会抛 CancelledError。
            # 这里捕获并标记为 cancelled 状态，产出带 [CANCELLED] 文本的结果，
            # 不让取消冒泡成异常，保证 gather 能正常收集到结果。
            result = ToolResult(
                call_id=call_id, name=name,
                result="[CANCELLED] Tool execution was cancelled.",
                status="cancelled",
            )
        except Exception as exc:
            # —— 普通异常（Exception）处理 ——
            # 工具内部抛出的非取消异常视为执行失败：记录 warning 日志、
            # 把异常信息同时写入 result 与 error，status 标记为 failed，
            # 让 LLM 能看到错误文本并决定是否重试或换路。
            # 注意：CancelledError 不是 Exception 的子类，因此不会误入此分支。
            logger.warning("[tool] %s failed: %s", name, exc)
            result = ToolResult(
                call_id=call_id, name=name,
                result=str(exc), error=str(exc), status="failed",
            )

        # ── on_post_tool hook ──
        # 工具执行完成后给 hook 一次机会改写结果文本。
        # 目前只支持 decision=="modify"：用 hook 返回的 modified_result 替换 result 字段，
        # 用于对工具输出做后处理（如脱敏、追加说明）。
        if self.hook_manager is not None:
            from ...hooks import ON_POST_TOOL, PostToolInput, PostToolOutput
            post_output = await self.hook_manager.trigger(
                ON_POST_TOOL,
                lambda: PostToolInput(
                    session_id=state.runtime_context.get("session_id", ""),
                    turn_id=state.turn_id,
                    iteration=state.iteration,
                    tool_call_id=call_id,
                    tool_name=name,
                    tool_args=arguments,
                    result=result.result,
                    error=result.error,
                    status=result.status,
                    metadata=result.metadata,
                    runtime_context=state.runtime_context,
                ),
            )
            if post_output is not None and post_output.decision == "modify":
                if isinstance(post_output, PostToolOutput) and post_output.modified_result is not None:
                    result.result = post_output.modified_result

        return result

    # ── 并发调度 ──────────────────────────────────────────────
    def spawn(
        self,
        call: ToolCall,
        state: "RunState",
        parent_span: TraceSpan | None = None,
    ) -> asyncio.Task:
        """为一个 ToolCall 创建并发执行任务，立即返回 asyncio.Task，不在此 await。

        调用时机：
          react_runner 在一轮 ReAct 迭代中收到 LLM 输出的若干 ToolCall 后，
          对每一个调用本方法，把它们全部丢到事件循环并发执行，
          这样可以在等待工具的同时不阻塞对 LLM 流的继续消费。

        参数：
          - call: LLM 输出的工具调用（含 id、name、input）。
          - state: 当前 RunState。
          - parent_span: 可选的父 tracing span，用于把本次工具调用挂到调用链上。

        返回：
          包装了 _run_one_traced 的 asyncio.Task，名字形如 "tool-<call_id>"。

        关于 parse_failed：
          call.input 为 None 表示 LLM 输出的参数 JSON 解析失败（registry 无法解析），
          此时把 parse_failed=True 传给 _run_one_traced -> run_one，
          run_one 会短路返回 [PARSE_ERROR] 结果，而不真正执行工具。
        """
        # 为本次工具调用创建子 span（若提供了父 span），记录工具名、输入参数与 call_id
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
                # call.input 为 None 表示 JSON 解析失败，标记 parse_failed 让 run_one 短路
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
        """``run_one`` 的 tracing 包装版本，负责把执行过程映射到 tracing span。

        本方法由 ``spawn`` 创建为 asyncio.Task 调用，不直接对外暴露。
        它在 run_one 之外包了一层异常处理与 span 状态收尾：

          - run_one 抛 BaseException（含 CancelledError）时：按异常类型结束 span
            （CancelledError -> CANCELLED，其他 -> ERROR），然后重新抛出，交由上层处理。
          - run_one 正常返回 ToolResult 时：根据 result.status 把 span 映射为
            COMPLETED / ERROR / CANCELLED，并写入结果快照。

        参数含义同 run_one，额外多一个 span 用于 tracing。
        """
        try:
            result = await self.run_one(
                call_id=call_id,
                name=name,
                arguments=arguments,
                state=state,
                parse_failed=parse_failed,
            )
        except BaseException as exc:
            # —— 异常路径：结束 span 并重新抛出 ——
            # run_one 内部已捕获 CancelledError 与普通 Exception，正常情况下不会抛到这里；
            # 但若 hook 或 registry 自身抛出未预期异常（BaseException 含 KeyboardInterrupt 等），
            # 需要先把 span 收尾再往上抛，避免 span 悬挂未关闭。
            if span and not span.ended:
                if isinstance(exc, asyncio.CancelledError):
                    span.end(status=TraceRunStatus.CANCELLED)
                else:
                    span.end(error=exc)
            raise

        # —— 正常路径：根据 ToolResult.status 映射 span 状态 ——
        # status 映射规则：
        #   result.status == "cancelled" -> TraceRunStatus.CANCELLED
        #   result.status == "failed"    -> TraceRunStatus.ERROR
        #   其他（completed）            -> TraceRunStatus.COMPLETED
        # 同时把结果文本、状态、错误信息写入 span 的 outputs，便于事后排查。
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
        """取消并等待一组工具任务，用于异常路径的清理。

        使用场景：
          react_runner 在 gather_results 之前若遇到致命错误（如 LLM 流异常、
          上层主动取消整轮迭代），需要把已经 spawn 出去、尚未完成的工具任务
          全部取消并回收，避免这些 task 在事件循环中残留成为孤儿。
          本方法先对每个 task 调 cancel()，再用 gather(return_exceptions=True)
          等它们都结束（异常被吞掉，只保证不抛出），从而安全地完成清理。
        """
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

        调用时机：
          react_runner 在 spawn 完所有工具任务后调用本方法阻塞等待全部完成，
          拿到与 LLM 输出 tool_calls 顺序一致的结果列表，随后构造 tool 消息写入 memory。

        返回：
          - results: 与 tool_calls 同序的 ToolResult 列表。
          - cancelled: 是否发生了取消（外部取消 / 有工具被取消 / state 已取消），
            为 True 时调用方写完 memory 后应抛出 CancelledError 终止本轮。

        关键逻辑：
          1) state.is_cancelled 预检查：若进入时整轮已被取消，先 cancel 所有 task；
          2) asyncio.CancelledError 捕获：gather 本身被外部取消时，再次 cancel 并重新
             gather 拿到所有结果，标记 cancelled_externally；
          3) 异常归一化：gather(return_exceptions=True) 会让任务异常以返回值形式出现，
             这里把 BaseException 统一转成 INTERRUPTED 的 ToolResult（取消场景->cancelled，
             其他异常->failed），保证 results 里不会有裸异常；
          4) 结果顺序保证：按 tool_calls 顺序从 finished 字典取结果，缺失的补一条
             "[INTERRUPTED] Tool result was lost."，确保与 LLM 输出严格对齐。
        """
        # —— 预检查：进入时整轮已被取消，主动 cancel 所有 task ——
        # 避免在已取消状态下还去 await，尽快让任务进入取消收尾。
        if state.is_cancelled:
            for t in tasks.values():
                t.cancel()

        cancelled_externally = False
        try:
            # return_exceptions=True：任务异常不会抛出，而是作为返回值，便于后续归一化
            raw = await asyncio.gather(*tasks.values(), return_exceptions=True)
        except asyncio.CancelledError:
            # —— gather 自身被外部取消（如用户中断整轮）——
            # 先把所有 task 都 cancel 掉，再重新 gather 一次拿到它们的最终状态，
            # 标记 cancelled_externally 让调用方知道本轮是被外部取消的。
            for t in tasks.values():
                t.cancel()
            raw = await asyncio.gather(*tasks.values(), return_exceptions=True)
            cancelled_externally = True

        # —— 异常归一化：把 gather 返回的裸异常转成 ToolResult ——
        # 用 call_id -> ToolResult 的字典暂存，保证后续能按 tool_calls 顺序取出。
        finished: dict[str, ToolResult] = {}
        for call_id, item in zip(tasks.keys(), raw):
            if isinstance(item, BaseException):
                # 任务以异常结束（CancelledError 或其他）：
                # 若整轮被取消或外部取消，视为 interrupted->cancelled；
                # 否则视为执行失败->failed。结果文本统一为 [INTERRUPTED]。
                interrupted = state.is_cancelled or cancelled_externally
                finished[call_id] = ToolResult(
                    call_id=call_id,
                    name=next((c.name for c in tool_calls if c.id == call_id), call_id),
                    result="[INTERRUPTED] Tool execution was interrupted.",
                    status="cancelled" if interrupted else "failed",
                )
            else:
                # 正常完成的 ToolResult，直接收录
                finished[call_id] = item

        # —— 结果顺序保证：严格按 tool_calls 顺序输出 ——
        # 从 finished 按 call_id 取结果；若某个 call_id 在 finished 中缺失（理论上不应发生），
        # 补一条 [INTERRUPTED] Tool result was lost. 的 failed 结果，保证列表长度与 tool_calls 一致。
        results = [
            finished.get(c.id) or ToolResult(
                call_id=c.id, name=c.name,
                result="[INTERRUPTED] Tool result was lost.", status="failed",
            )
            for c in tool_calls
        ]
        # 任一结果为 cancelled 即认为本轮发生了取消，通知调用方终止
        any_cancelled = any(r.status == "cancelled" for r in results)
        return results, (cancelled_externally or any_cancelled or state.is_cancelled)
