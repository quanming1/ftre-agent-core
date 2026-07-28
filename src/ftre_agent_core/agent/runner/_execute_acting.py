"""Acting 执行器 + Exit 执行器。

本模块实现 ReAct 状态机中两个“动作执行器”：

ActingExecutor：
  负责 Acting 动作的执行——并发跑工具、成组写入 Memory、产出流事件。
  它是工具调用的“编排层”：决定 spawn 顺序、写 memory 的顺序、事件的产出顺序，
  以及取消信号的传播。真正的“如何执行单个工具”由 ToolHandler 负责。

ExitExecutor：
  负责 Exit 动作的执行——在 Agent 准备结束回复时，先过一遍 ON_STOP Hook，
  根据返回值决定是真的退出还是注入续写提示继续下一轮；真正退出时设置终态并
  产出 ReplyEndEvent。
"""
from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, AsyncGenerator

from ...event import (
    AgentStreamEvent, ReplyEndEvent, HintBlockEvent,
    ToolResultStartEvent, ToolResultTextDeltaEvent, ToolResultEndEvent,
)
from ...message import ToolResultState
from ...types import ReplyFinishedReason
from ._state import Acting, Exit, ExitOutcome, RunStatus, CancelledError
from .tool_handler import ToolHandler

if TYPE_CHECKING:
    from ...hooks import FtreCoreHookManager
    from ._state import RunState


# ═══════════════════════════════════════════════════════════════
# ActingExecutor
# ═══════════════════════════════════════════════════════════════

class ActingExecutor:
    """执行 Acting 动作：并发跑工具、成组写入 Memory、产出流事件。

    职责定位（与 ToolHandler 的分工）：
      - 本类是“编排层”：拿到一轮 LLM 输出的 tool_calls 后，调度它们的并发执行、
        决定 assistant(tool_calls) 与 tool(result) 写入 memory 的先后顺序、
        向上层 yield 工具结果事件与 hint 事件、在取消时抛出 CancelledError。
      - ToolHandler 是“执行层”：负责单个工具的真正派发、on_pre_tool/on_post_tool
        hook 集成、异常归一化、tracing span 管理；它不关心 memory 写入与事件产出。
      本类不直接 await 单个工具，而是通过 ToolHandler.spawn / gather_results 间接驱动。
    """

    def __init__(self, agent, state: "RunState", tool_handler: ToolHandler):
        """初始化 Acting 执行器。

        参数：
          - agent: 宿主 Agent 实例，提供 memory（写入消息）、tool_registry 等。
          - state: 当前 run() 的运行状态（RunState），提供 reply_id、trace_span、
            is_cancelled 等运行期上下文。
          - tool_handler: 工具执行处理器（ToolHandler），负责 spawn / gather_results /
            build_assistant_message，通常在整个 agent 运行期共享一个实例。
        """
        self.agent = agent
        self.state = state
        self.tool_handler = tool_handler

    async def stream(self, action: Acting) -> AsyncGenerator[AgentStreamEvent, None]:
        """执行一轮工具调用，按工具顺序产出流事件。

        整体流程分四个阶段：
          1) spawn：为每个 tool_call 创建并发任务（不 await，立即返回 Task）；
          2) gather_results：阻塞等待全部完成，归一化异常与取消，按原序拿回结果；
          3) 成组写入 memory：先写 assistant(tool_calls)，再逐条写 tool(result)；
          4) 延后追加 pending_hints 并 yield 对应事件；若发生取消则抛出 CancelledError。

        取消传播：
          gather_results 返回 cancelled=True 表示本轮被取消（外部取消 / 某工具被取消 /
          state 已取消）。本方法在写完 memory 与事件后抛出内部的 CancelledError，
          由 react_runner._loop 捕获，转成 INTERRUPTED 终态并产出 ReplyEndEvent。
        """
        reply_id = self.state.reply_id
        tool_calls = action.tool_calls

        # ── 阶段 1：spawn 所有工具任务 ──
        # 对每个 tool_call 调 tool_handler.spawn 创建 asyncio.Task，这里【不 await】——
        # spawn 只是把任务丢进事件循环并立即返回 Task 句柄。之所以不在循环里等待，
        # 是为了让所有工具任务能并发跑起来（一个慢工具不阻塞其它工具的启动），
        # 同时保持本循环纯同步、不阻塞对 LLM 流的继续消费。真正的等待在阶段 2 完成。
        tool_tasks: dict[str, asyncio.Task] = {}
        for call in tool_calls:
            tool_tasks[call.id] = self.tool_handler.spawn(
                call,
                self.state,
                parent_span=self.state.trace_span,
            )

        # ── 阶段 2：等待全部完成 + 取消处理 ──
        # gather_results 会 await 所有任务结束（return_exceptions=True 收敛异常），
        # 把 CancelledError / Exception 归一化成 cancelled / failed 的 ToolResult，
        # 并返回 cancelled 标志。这一步阻塞直到所有工具都有结果（含失败/取消）。
        results, cancelled = await self.tool_handler.gather_results(
            tool_calls, tool_tasks, self.state,
        )

        # ── 阶段 3：成组写入 memory ──
        # 顺序约束：assistant(tool_calls) 消息必须先于所有 tool(result) 写入。
        # OpenAI / Anthropic 协议要求 role="tool" 的结果消息紧跟在带 tool_calls 的
        # assistant 消息之后、且同组的 tool 结果必须连续；若 assistant 消息缺失或
        # 顺序颠倒，后续 LLM 上下文会出现“无主 tool 消息”，直接违反协议导致报错。
        self.agent.memory.add_raw(
            self.tool_handler.build_assistant_message(
                tool_calls=tool_calls,
            )
        )

        # 工具可能返回事件对象（ToolResult.event，如 HintBlockEvent）而非纯文本，
        # 这些事件需要向上冒泡给调用方。但它们不能在下面的循环里立即写入 memory，
        # 否则会被插在 tool(result) 序列中间——见阶段 4 的说明。先收集起来。
        pending_hints: list[AgentStreamEvent] = []

        for tc, result in zip(tool_calls, results):
            # 写入这一条 tool 结果（role="tool"），与上面的 assistant 消息配对
            self.agent.memory.add_tool_result(
                tc.id, result.result or f"[{tc.name}] 已完成"
            )

            # 产出工具结果事件三元组：Start → (TextDelta) → End
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

            # 工具返回了事件对象（非文本）→ 暂存，阶段 4 统一处理
            if result.event is not None:
                pending_hints.append(result.event)

        # ── 阶段 4：延后追加 pending_hints ──
        # 之所以把 hints 放到所有 tool(result) 之后再写入，而不是在循环里即时写：
        # hint 本质是 role="user" 的注入消息，若把它插在某两条 tool(result) 之间，
        # 会打破“同组 tool 结果必须连续”的消息序列合法性，后续 LLM 调用会因消息
        # 顺序非法而报错。因此必须等全部 tool 结果写完，再统一把 hints 追加到末尾。
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

        # ── 取消传播 ──
        # cancelled=True 时抛出内部 CancelledError（与 asyncio.CancelledError 区分）。
        # 传播路径：本方法抛出 → react_runner._loop 的 except CancelledError 捕获 →
        # 调 react_runner._finalize(INTERRUPTED) 设置终态 → yield ReplyEndEvent(INTERRUPTED)。
        # 注意：此时 memory 与事件已写完，取消只影响“本轮之后是否继续 / 以何终态收尾”。
        if cancelled:
            raise CancelledError()


# ═══════════════════════════════════════════════════════════════
# ExitExecutor
# ═══════════════════════════════════════════════════════════════

class ExitExecutor:
    """执行 Exit 动作：ON_STOP Hook 检查 + 产出 ReplyEnd + 设置终态。

    ON_STOP Hook 的两种路径：
      - block：Hook 决定不让 Agent 停下。此时不产出 ReplyEndEvent、不设终态，
        而是把 Hook 的 reason 作为续写提示注入 memory、yield 一个 HintBlockEvent，
        并把 outcome 置为 should_continue=True，让 react_runner._loop 进入下一轮迭代。
      - allow（或无 Hook / 非 COMPLETED 退出）：正常退出路径——_finalize 设置终态，
        yield ReplyEndEvent，outcome 保持 should_continue=False，主循环收到后 return。

    仅当 finished_reason == COMPLETED 时才触发 ON_STOP：ERROR / EXCEED_MAX_ITERS 属于
    异常 / 超限退出，没有“要不要让 Agent 继续”的语义，不需要问 Hook，直接走正常退出。
    """

    def __init__(self, agent, state: "RunState", hook_manager: "FtreCoreHookManager"):
        """初始化 Exit 执行器。

        参数：
          - agent: 宿主 Agent 实例，提供 memory（注入续写提示）等。
          - state: 当前 run() 的运行状态（RunState），退出时由 _finalize 写入终态。
          - hook_manager: Hook 管理器，用于触发 ON_STOP 挂点；无注册 Hook 时
            trigger 返回 None，等价于 allow。

        实例属性：
          - outcome: 本次 Exit 的结果载体（ExitOutcome），初始 should_continue=False。
            ON_STOP block 时被置为 should_continue=True，供 react_runner._loop 判断
            是否继续下一轮迭代。
        """
        self.agent = agent
        self.state = state
        self.hook_manager = hook_manager
        self.outcome: ExitOutcome = ExitOutcome()

    async def stream(self, action: Exit) -> AsyncGenerator[AgentStreamEvent, None]:
        """执行退出逻辑：先过 ON_STOP Hook，再决定续写还是真正退出。

        - finished_reason == COMPLETED：触发 ON_STOP，Hook 可 block（续写）或 allow（退出）。
        - 其它 reason（ERROR / EXCEED_MAX_ITERS / INTERRUPTED）：跳过 Hook，直接退出。
        """
        session_id = self.state.runtime_context.get("session_id", "")
        reply_id = self.state.reply_id

        # ── 仅 COMPLETED 触发 ON_STOP ──
        # ERROR（LLM 调用失败 / 空响应耗尽重试）和 EXCEED_MAX_ITERS（超过最大迭代数）
        # 都是“被迫退出”，没有“让 Hook 决定是否继续”的语义——Hook 没法把一个失败的
        # LLM 调用变成成功，也无法突破迭代上限。故只在 Agent 主动完成（COMPLETED）时，
        # 给 Hook 一次“拦下并要求继续干活”的机会。
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

            # ── ON_STOP block：不退出，注入续写提示 ──
            # Hook 返回 decision="block" 表示“别停，接着干”。行为：
            #   1) 把 Hook 的 reason（或默认“继续工作。”）作为 role="user" 消息写入 memory，
            #      让下一轮 LLM 调用能看到这条续写指令；
            #   2) yield 一个 HintBlockEvent（hide=True / internal=True），用于向上层 / 前端
            #      传达“本次停止被 Hook 拦截”，但标记为内部隐藏事件，不直接展示给用户；
            #   3) 置 outcome.should_continue=True，react_runner._loop 据此不 return 而是
            #      continue 进入下一轮迭代；本方法提前 return，不产出 ReplyEndEvent、不设终态。
            if stop_output is not None and stop_output.decision == "block":
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

        # ── 正常退出路径 ──
        # 走到这里说明：非 COMPLETED 退出，或 COMPLETED 但 ON_STOP allow / 无 Hook。
        # 1) _finalize 把 reason / error / error_code 写入 RunState，并把 status 映射成
        #    COMPLETED / ERROR / CANCELLED 终态；
        # 2) yield ReplyEndEvent 通知上层本轮回复正式结束；
        # 3) outcome 保持默认（should_continue=False），react_runner._loop 收到后 return。
        self._finalize(action.finished_reason, action.error, action.error_code)

        yield ReplyEndEvent(
            session_id=session_id,
            reply_id=reply_id,
            finished_reason=action.finished_reason,
            error={"message": action.error, "code": action.error_code} if action.error else None,
        )
        self.outcome = ExitOutcome()

    def _finalize(self, reason: ReplyFinishedReason, error: str | None, error_code: str | None) -> None:
        """把退出原因与错误信息写入 RunState，并设置终态。

        reason → RunStatus 映射：
          - INTERRUPTED → CANCELLED（被取消 / 中断）
          - ERROR       → ERROR（LLM 调用失败或空响应耗尽重试）
          - 其它（COMPLETED / EXCEED_MAX_ITERS）→ COMPLETED

        EXCEED_MAX_ITERS 归入 COMPLETED 而非 ERROR：它不是错误，而是达到迭代上限的正常停止。
        error / error_code 仅在非空时写入，供上层 tracing 与 ReplyEndEvent 使用。
        """
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
