"""纯决策函数：根据当前状态和上一轮结果决定下一步动作。

本模块是 ReAct 状态机的核心——不执行任何 I/O，不 yield 事件，
只读取 RunState 和 TurnResult，返回 Reasoning / Acting / Exit 三种动作之一。

设计灵感来自 AgentScope 的 _next_action()：决策与执行分离，
决策函数可独立单测，执行器各自独立处理副作用。

所有副作用仅限修改 state.empty_retries 和 state.in_finalization 两个计数器。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ...types import ReplyFinishedReason
from ._actions import Reasoning, Acting, Exit, TurnResult

if TYPE_CHECKING:
    from ._state import RunState

# 空响应最多重试次数（不含强制最终化那一次）。
# 超过此次数后进入"强制最终化"阶段：去掉工具，注入提示，让模型只输出文本。
MAX_EMPTY_RESPONSE_RETRIES = 2

# 强制最终化时注入 Memory 的提示词，要求模型直接给出最终回复。
FINALIZATION_RETRY_PROMPT = "请根据上面的对话，直接给出回复用户的最终内容。"

# 强制最终化后仍然返回空响应时的错误提示。
EMPTY_FINAL_RESPONSE_MESSAGE = "模型多次重试后仍未生成可见的最终文本回复。"


def decide(state: "RunState", prev: TurnResult | None) -> Reasoning | Acting | Exit:
    """根据当前状态和上一轮 TurnResult 决定下一步动作。

    纯函数：不执行 I/O，不 yield 事件。
    副作用仅限修改 state.empty_retries 和 state.in_finalization。

    判断优先级（从高到低）：
        1. prev.error 非空 → Exit(ERROR)
           LLM 调用重试耗尽或遇到不可重试错误，直接退出。
        2. prev.tool_calls 非空 → Acting(tool_calls)
           模型本轮产生了工具调用，需要执行这些工具。
        3. prev.text 非空且无工具调用 → Exit(COMPLETED)
           模型本轮给出了纯文本回答，正常完成。
        4. 空响应 + in_finalization → Exit(ERROR)
           已在强制最终化阶段还是空响应，彻底失败。
        5. 空响应 + empty_retries < MAX → Reasoning() 重试
           模型返回空内容，重试计数 +1，继续推理。
        6. 空响应 + 重试耗尽 → Reasoning(最终化提示, force_no_tools)
           重试次数用尽，进入强制最终化：去掉工具，注入提示。
        7. iteration >= max_iterations → Exit(EXCEED_MAX_ITERS)
           达到最大迭代次数，防止无限循环。
        8. 默认 → Reasoning()
           首轮（prev=None）或工具执行后（prev=None），继续推理。

    Args:
        state: 当前 RunState（只读 iteration / empty_retries /
               in_finalization / runtime_context）。
        prev: 上一轮推理的 TurnResult，None 表示首轮或刚执行完工具。

    Returns:
        Reasoning: 继续调用大模型进行推理
        Acting:    执行模型产生的工具调用
        Exit:      结束（或暂停）当前回复
    """
    max_iterations = state.runtime_context.get("max_iterations")

    # ===========================================================
    # 第一步：LLM 错误检查（最高优先级）
    # ===========================================================
    # 如果上一轮 LLM 调用重试耗尽或遇到不可重试错误，
    # 直接退出，不再尝试推理或执行工具。
    if prev is not None and prev.error is not None:
        return Exit(
            finished_reason=ReplyFinishedReason.ERROR,
            error=f"[{prev.error.code}] {prev.error.message}",
            error_code=prev.error.code,
        )

    # ===========================================================
    # 第二步：工具调用检查
    # ===========================================================
    # 模型本轮产生了工具调用 → 执行这些工具。
    # 这一步优先于文本检查：即使模型同时输出了文本和工具调用，
    # 也先执行工具，下一轮再让模型根据工具结果生成最终文本。
    if prev is not None and prev.tool_calls:
        return Acting(tool_calls=prev.tool_calls)

    # ===========================================================
    # 第三步：纯文本回答检查
    # ===========================================================
    # 模型本轮给出了非空文本且无工具调用 → 正常完成本次回复。
    # 注意：空白文本（如 " \n "）不视为有效回答，走空响应处理。
    if prev is not None and prev.text.strip():
        return Exit(finished_reason=ReplyFinishedReason.COMPLETED)

    # ===========================================================
    # 第四步：空响应处理（prev 非空但文本为空）
    # ===========================================================
    # 模型返回了空内容（只有 reasoning 或完全空），需要特殊处理。
    if prev is not None and not prev.text.strip():
        # 4a. 已在强制最终化阶段还是空 → 彻底失败
        # 前一轮已经去掉工具并注入了最终化提示，模型仍然返回空，
        # 说明模型无法生成有效回复，直接报错退出。
        if state.in_finalization:
            return Exit(
                finished_reason=ReplyFinishedReason.ERROR,
                error=EMPTY_FINAL_RESPONSE_MESSAGE,
                error_code="empty_response",
            )

        # 4b. 重试次数未达上限 → 继续 Reasoning
        # 模型可能只是偶发性返回空，重试一次通常能正常输出。
        # 计数 +1 是本函数唯一的副作用之一。
        if state.empty_retries < MAX_EMPTY_RESPONSE_RETRIES:
            state.empty_retries += 1
            return Reasoning()

        # 4c. 重试耗尽 → 进入强制最终化
        # 去掉工具（force_no_tools=True），注入最终化提示，
        # 让模型只能输出文本、不能调用工具。
        # in_finalization 标志是本函数的另一个副作用。
        state.in_finalization = True
        return Reasoning(
            hint=FINALIZATION_RETRY_PROMPT,
            force_no_tools=True,
        )

    # ===========================================================
    # 第五步：最大迭代次数检查
    # ===========================================================
    # prev 为 None（首轮或刚执行完工具）时检查是否已达上限。
    # iteration 在每次 Reasoning 时递增，max_iterations=N 表示
    # 最多调用 N 次 LLM。
    if max_iterations is not None and state.iteration >= max_iterations:
        return Exit(finished_reason=ReplyFinishedReason.EXCEED_MAX_ITERS)

    # ===========================================================
    # 第六步：默认 → 继续推理
    # ===========================================================
    # 首轮（prev=None，iteration=0）或工具执行后（prev=None），
    # 调用 LLM 进行下一轮推理。
    return Reasoning()
