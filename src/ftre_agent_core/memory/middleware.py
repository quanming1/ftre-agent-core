"""
Memory 中间件

在消息读写的关键节点插入自定义逻辑，不侵入 Manager 核心代码。

三个钩子:
    on_get_messages  → get_messages() 组装完消息列表后，发给 LLM 前
    on_add_message   → 每条消息写入后
    on_loop_end      → ReAct 循环结束后（剪枝、压缩等后处理）

设计要点:
    - 与 ToolMiddleware 对称，ABC 基类 + 默认空实现
    - 子类按需覆写，不需要的钩子保持默认
    - Manager 持有 list[MemoryMiddleware]，按注册顺序调用
"""
from __future__ import annotations

from abc import ABC
from typing import Any


class MemoryMiddleware(ABC):
    """
    Memory 中间件基类

    子类按需覆写钩子方法，不需要的保持默认即可。
    """

    def on_get_messages(self, messages: list[dict]) -> list[dict]:
        """
        get_messages() 组装完消息列表后调用（含 system）

        用途:
        - 过滤/变换消息（如注入额外上下文）
        - 按 token 预算截断

        Args:
            messages: 完整消息列表 (含 system prompt)

        Returns:
            变换后的消息列表
        """
        return messages

    def on_add_message(self, message: Any) -> Any | None:
        """
        每条消息写入 Manager 后调用

        用途:
        - 拦截/修改新消息（如截断过长的工具输出）
        - 返回 None 则丢弃该消息

        Args:
            message: 刚写入的消息对象

        Returns:
            消息对象（可修改），或 None 丢弃
        """
        return message

    def on_loop_end(self, messages: list) -> int:
        """
        ReAct 循环结束后调用

        用途:
        - 剪枝旧工具输出 (PruneMiddleware)
        - 上下文压缩 (CompactMiddleware)
        - 统计/日志

        Args:
            messages: Manager 内部消息列表（可原地修改）

        Returns:
            受影响的消息数量（用于日志/统计）
        """
        return 0
