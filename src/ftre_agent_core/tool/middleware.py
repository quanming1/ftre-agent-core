"""
Tool 中间件

在 tool 执行前后插入自定义逻辑，不侵入 ReActRunner 核心代码。

执行顺序：
    before 链（按注册顺序）→ 实际执行（或短路）→ after 链（按注册逆序，洋葱模型）

设计要点：
    - ToolContext 贯穿整个调用生命周期，中间件间通过 metadata 通信
    - before 可短路（skip），跳过实际执行直接返回结果
    - CancelledError 不经过 after，属于异常流
"""
from abc import ABC
from dataclasses import dataclass, field
from typing import Any

from ftre_agent_core.tool_system import CancellationToken


@dataclass
class ToolContext:
    """
    单次 tool 调用的上下文

    贯穿 before → execute → after 全程。
    中间件通过 metadata 传递数据，外层（runner / workflow）也可读取。
    """
    call_id: str
    name: str
    arguments: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    cancel_token: CancellationToken = field(default_factory=CancellationToken)

    _skipped: bool = field(default=False, repr=False)
    _skip_result: str = field(default="", repr=False)

    @property
    def skipped(self) -> bool:
        return self._skipped

    @property
    def skip_result(self) -> str:
        return self._skip_result

    def skip(self, result: str = "") -> None:
        self._skipped = True
        self._skip_result = result


class ToolMiddleware(ABC):
    """
    Tool 中间件基类

    子类按需覆写 before / after，不需要的钩子保持默认即可。
    """

    def before(self, context: ToolContext) -> ToolContext:
        """
        tool 执行前调用

        可以：
        - 修改 arguments: context.arguments["key"] = ...
        - 短路执行: context.skip(result="cached value")
        - 写元数据: context.metadata["key"] = value

        Returns:
            context（支持链式传递）
        """
        return context

    def after(self, context: ToolContext, result: "ToolResult") -> "ToolResult":
        """
        tool 执行后调用（逆序，洋葱模型）

        可以：
        - 修改 result
        - 读写 context.metadata
        - 抛异常中断流程

        Returns:
            result（可替换）
        """
        return result
