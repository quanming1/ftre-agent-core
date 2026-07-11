"""
Core Hook 系统 — 让外部代码挂到 ReActRunner 的关键决策点上。

与 Gateway 层的 HookManager（filter chain，只能改写 ctx）不同，本系统支持
block 决策：hook 可以阻止 Agent 停止、阻止工具执行、修改工具参数或结果。

执行模型：
  - 一个挂点上可注册多个 hook，按注册顺序依次执行
  - hook 可以是 async def 或普通 def（trigger 自动 await）
  - hook 返回 None 视为 allow，不中断链
  - hook 返回 decision='block' 立即终止链，调用方据此决定是否阻止主流程
  - hook 返回 decision='modify' 不终止链，修改后的 input 传递给后续 hook
  - hook 抛异常 → 捕获 + log，跳过该 hook 继续链

挂点一览：

  on_turn_start  — 每轮迭代开始前，可注入消息
  on_pre_tool    — 每个工具执行前，可拒绝/改参数
  on_post_tool   — 每个工具执行后，可改结果
  on_stop        — Agent 想停下时，可阻止停止并注入 continuation prompt ★
  on_turn_end    — 每轮迭代结束后，只读观察
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 挂点常量
# ═══════════════════════════════════════════════════════════════════

ON_TURN_START = "on_turn_start"
ON_PRE_TOOL = "on_pre_tool"
ON_POST_TOOL = "on_post_tool"
ON_STOP = "on_stop"
ON_TURN_END = "on_turn_end"


# ═══════════════════════════════════════════════════════════════════
# 基类
# ═══════════════════════════════════════════════════════════════════

@dataclass
class HookInput:
    """所有 hook 输入的基类。

    子类按需追加自己的字段，但不应移除基类字段。
    """
    session_id: str = ""
    turn_id: str = ""
    iteration: int = 0
    runtime_context: dict = field(default_factory=dict)


@dataclass
class HookOutput:
    """所有 hook 输出的基类。

    decision 三态：
      'allow'  — 放行（默认）
      'block'  — 阻止当前操作
      'modify' — 修改后放行（配合子类的 modified_* 字段）
    """
    decision: str = "allow"
    reason: str = ""
    system_message: str = ""


# ═══════════════════════════════════════════════════════════════════
# on_turn_start
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TurnStartInput(HookInput):
    """每轮迭代开始前。

    触发位置：_loop() 中 iteration += 1 之后、_run_turn() 之前。
    用途：注入 system/user 消息（如每日提醒、上下文补充）。
    """
    messages: list[dict] = field(default_factory=list)


@dataclass
class TurnStartOutput(HookOutput):
    """inject_messages 中的消息会追加到 memory，Agent 在本轮迭代中可见。"""
    inject_messages: list[dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# on_stop ★
# ═══════════════════════════════════════════════════════════════════

@dataclass
class StopInput(HookInput):
    """Agent 想停下时（无工具调用、有文本输出、finish_reason 正常）。

    触发位置：_stream_turn() 阶段 3，在 length/unknown/pending_user_messages
    分支之后、正常结束之前。

    这是唯一能"阻止 Agent 停下"的挂点。decision='block' 时，
    reason 会作为 continuation prompt 注入 memory，Agent 进入下一轮迭代。
    """
    last_assistant_text: str = ""
    finish_reason: str = ""
    token_usage: dict = field(default_factory=dict)


# StopOutput 无额外字段，直接使用 HookOutput。
# decision='block' + reason=continuation prompt → 阻止停止
# decision='allow' → 正常停止


# ═══════════════════════════════════════════════════════════════════
# on_pre_tool
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PreToolInput(HookInput):
    """每个工具执行前。

    触发位置：_stream_turn() 阶段 4，工具任务创建之前。
    用途：权限审批、参数修改、危险操作拦截。

    decision='block' → 工具不执行，reason 作为 tool_result 的错误内容。
    decision='modify' → modified_args 替换原始参数后执行。
    """
    tool_call_id: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)


@dataclass
class PreToolOutput(HookOutput):
    modified_args: dict | None = None


# ═══════════════════════════════════════════════════════════════════
# on_post_tool
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PostToolInput(HookInput):
    """每个工具执行后。

    触发位置：run_one() 中工具执行完毕后（含异常路径）。
    用途：结果审计、敏感信息脱敏、结果改写。

    decision='modify' → modified_result 替换原始 result 字符串。
    decision='block' → 当前无特殊语义，等同于 allow。
    """
    tool_call_id: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    result: str = ""
    error: str | None = None
    status: str = "completed"
    metadata: dict = field(default_factory=dict)


@dataclass
class PostToolOutput(HookOutput):
    modified_result: str | None = None


# ═══════════════════════════════════════════════════════════════════
# on_turn_end
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TurnEndInput(HookInput):
    """每轮迭代结束后（只读观察，不能阻止）。

    触发位置：_loop() 中 _run_turn() 返回后。
    用途：遥测、日志、UI 通知。decision 字段被忽略。
    """
    done_reason: str = ""


# TurnEndOutput 无额外字段，直接使用 HookOutput。


# ═══════════════════════════════════════════════════════════════════
# HookManager
# ═══════════════════════════════════════════════════════════════════

HookFunc = Callable[[HookInput], "HookOutput | None"]


class FtreCoreHookManager:
    """Core 层 Hook 注册与调度中心。

    设计要点：
      - 多 hook 按注册顺序链式执行
      - 任一 hook 返回 block 立即终止链
      - hook 异常被捕获跳过，不拖垮主流程
      - hook 可以是 async def 或普通 def
    """

    def __init__(self) -> None:
        self._hooks: dict[str, list[HookFunc]] = {}

    def register(self, point: str, fn: HookFunc) -> None:
        """在指定挂点注册一个 hook（按注册顺序执行）。"""
        if not callable(fn):
            raise TypeError(f"hook 必须可调用，收到 {type(fn).__name__}")
        self._hooks.setdefault(point, []).append(fn)
        logger.info(
            "[core-hook] 注册: point=%s fn=%s",
            point,
            getattr(fn, "__qualname__", repr(fn)),
        )

    def unregister(self, point: str, fn: HookFunc) -> bool:
        """移除指定挂点上的一个 hook。返回是否找到并移除。"""
        hooks = self._hooks.get(point)
        if not hooks:
            return False
        try:
            hooks.remove(fn)
            logger.info(
                "[core-hook] 移除: point=%s fn=%s",
                point,
                getattr(fn, "__qualname__", repr(fn)),
            )
            return True
        except ValueError:
            return False

    def has_hooks(self, point: str) -> bool:
        """该挂点是否有已注册的 hook。"""
        return bool(self._hooks.get(point))

    def clear(self, point: str | None = None) -> None:
        """清除指定挂点的全部 hook，或清除所有挂点（point=None）。"""
        if point is None:
            self._hooks.clear()
        else:
            self._hooks.pop(point, None)

    async def trigger(
        self,
        point: str,
        input: HookInput | Callable[[], HookInput],
    ) -> HookOutput | None:
        """触发一条 hook 链。

        input 可以是 HookInput 实例，也可以是零参 callable（工厂函数）。
        传 callable 时，没有 hook 则不会调用它 —— 避免无谓的 input 构造开销。

        返回：
          - None：没有 hook
          - HookOutput：最后一个非 None 的 hook 输出（可能是 allow/modify/block）
          调用方根据 decision 字段决定后续行为。

        行为：
          - hook 返回 None → 视为 allow，继续链
          - hook 返回 decision='block' → 立即返回该 output，终止链
          - hook 返回 decision='modify' → 将修改写回 input，继续链
          - hook 返回 decision='allow' → 记录为最后输出，继续链
          - hook 抛异常 → 捕获 + log，跳过该 hook 继续
        """
        hooks = self._hooks.get(point)
        if not hooks:
            return None

        # 懒构造：只有真的有 hook 时才实例化 input。
        if callable(input):
            input = input()

        last_output: HookOutput | None = None

        for fn in hooks:
            try:
                result = fn(input)
                if asyncio.iscoroutine(result):
                    result = await result
            except Exception:
                logger.exception(
                    "[core-hook] 执行异常，已跳过: point=%s fn=%s",
                    point,
                    getattr(fn, "__qualname__", repr(fn)),
                )
                continue

            if result is None:
                continue

            last_output = result

            if result.decision == "block":
                return result

            if result.decision == "modify":
                _apply_modification(input, result)

        return last_output


# ═══════════════════════════════════════════════════════════════════
# 内部工具
# ═══════════════════════════════════════════════════════════════════

def _apply_modification(input: HookInput, output: HookOutput) -> None:
    """将 modify 输出写回 input（就地修改）。

    目前只有 PreToolInput / PostToolInput 有 modify 语义：
      - PreToolOutput.modified_args → PreToolInput.tool_args
      - PostToolOutput.modified_result → PostToolInput.result
    """
    if isinstance(output, PreToolOutput) and output.modified_args is not None:
        if isinstance(input, PreToolInput):
            input.tool_args = output.modified_args
    elif isinstance(output, PostToolOutput) and output.modified_result is not None:
        if isinstance(input, PostToolInput):
            input.result = output.modified_result
