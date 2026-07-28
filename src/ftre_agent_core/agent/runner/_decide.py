"""纯决策函数：根据当前状态和上一轮结果决定下一步动作。"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ...types import ReplyFinishedReason
from ._actions import Reasoning, Acting, Exit, TurnResult

if TYPE_CHECKING:
    from ._state import RunState

MAX_EMPTY_RESPONSE_RETRIES = 2

FINALIZATION_RETRY_PROMPT = "请根据上面的对话，直接给出回复用户的最终内容。"
EMPTY_FINAL_RESPONSE_MESSAGE = "模型多次重试后仍未生成可见的最终文本回复。"


def decide(state: "RunState", prev: TurnResult | None) -> Reasoning | Acting | Exit:
    """根据当前状态和上一轮 TurnResult 决定下一步动作。

    纯函数：不执行 I/O，不 yield 事件。
    副作用仅限修改 state.empty_retries 和 state.in_finalization。

    判断优先级：
        1. prev.error 非空 → Exit(ERROR)
        2. prev.tool_calls 非空 → Acting
        3. prev.text 非空且无工具调用 → Exit(COMPLETED)
        4. 空响应 + in_finalization → Exit(ERROR)
        5. 空响应 + empty_retries < MAX → Reasoning() 重试
        6. 空响应 + 重试耗尽 → Reasoning(最终化提示, force_no_tools)
        7. iteration >= max_iterations → Exit(EXCEED_MAX_ITERS)
        8. 默认 → Reasoning()
    """
    max_iterations = state.runtime_context.get("max_iterations")

    # 1. LLM 错误 → 直接退出
    if prev is not None and prev.error is not None:
        return Exit(
            finished_reason=ReplyFinishedReason.ERROR,
            error=f"[{prev.error.code}] {prev.error.message}",
            error_code=prev.error.code,
        )

    # 2. 有工具调用 → 执行工具
    if prev is not None and prev.tool_calls:
        return Acting(tool_calls=prev.tool_calls)

    # 3. 有非空文本且无工具调用 → 正常完成
    if prev is not None and prev.text.strip():
        return Exit(finished_reason=ReplyFinishedReason.COMPLETED)

    # 4-6. 空响应处理（prev 非空但文本为空）
    if prev is not None and not prev.text.strip():
        # 4. 已在最终化阶段还是空 → 彻底失败
        if state.in_finalization:
            return Exit(
                finished_reason=ReplyFinishedReason.ERROR,
                error=EMPTY_FINAL_RESPONSE_MESSAGE,
                error_code="empty_response",
            )

        # 5. 重试次数未达上限 → 继续 Reasoning
        if state.empty_retries < MAX_EMPTY_RESPONSE_RETRIES:
            state.empty_retries += 1
            return Reasoning()

        # 6. 重试耗尽 → 进入强制最终化
        state.in_finalization = True
        return Reasoning(
            hint=FINALIZATION_RETRY_PROMPT,
            force_no_tools=True,
        )

    # 7. 达到最大迭代次数
    if max_iterations is not None and state.iteration >= max_iterations:
        return Exit(finished_reason=ReplyFinishedReason.EXCEED_MAX_ITERS)

    # 8. 默认 → 继续推理
    return Reasoning()
