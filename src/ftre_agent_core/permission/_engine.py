# -*- coding: utf-8 -*-
"""纯权限决策引擎。

引擎把「一次请求 + 一组规则 + 兜底行为」转换成「一个决策」，是无副作用的纯函数：
不碰 Runner / ToolHandler / AgentState，不修改传入参数，返回可序列化的决策。

规则的持久化事实源是 AgentState.permission_context，由 Core（编排层）取出后
以 ``rules`` / ``default_behavior`` 显式传入本引擎；引擎自身不反向依赖 state 包。

冲突 / 默认语义：
  - 在所有「启用且命中」的规则里，priority 最高者胜出。
  - 若多条规则并列最高优先级但行为不一致，安全失败，返回 DENY。
  - 没有任何规则命中时，使用调用方传入的 default_behavior。
"""
from __future__ import annotations

from ._types import (
    PermissionBehavior,
    PermissionDecision,
    PermissionRequest,
    PermissionRule,
)


class PermissionEngine:
    """纯权限决策引擎。

    ``evaluate`` 只依赖显式传入的参数，不修改它们，
    返回可序列化的 :class:`PermissionDecision`。
    """

    def evaluate(
        self,
        request: PermissionRequest,
        rules: list[PermissionRule],
        default_behavior: PermissionBehavior = PermissionBehavior.ASK,
    ) -> PermissionDecision:
        """为 ``request`` 求解应采取的行为。

        参数：
            request: 待决策的工具调用。
            rules: 参与匹配的规则集合（来自 AgentState.permission_context）。
            default_behavior: 无规则命中时的兜底行为，默认 ASK。

        返回：
            一个决策，携带最终行为、可读原因，以及（当由某条规则决定时）
            命中的规则 id。
        """
        # 1. 找出所有「启用 且 命中（精确名或通配）」的规则
        matched = [
            rule
            for rule in rules
            if rule.enabled and rule.tool_name in ("*", request.tool_name)
        ]

        # 2. 没有任何规则命中 → 用传入的默认行为兜底
        if not matched:
            return PermissionDecision(
                behavior=default_behavior,
                reason="No permission rule matched",
            )

        # 3. 取命中规则里的最高优先级
        highest_priority = max(rule.priority for rule in matched)
        candidates = [
            rule for rule in matched if rule.priority == highest_priority
        ]

        # 4. 并列最高优先级但行为不一致 → 安全失败，返回 DENY
        behaviors = {rule.behavior for rule in candidates}
        if len(behaviors) > 1:
            return PermissionDecision(
                behavior=PermissionBehavior.DENY,
                reason="Conflicting permission rules at the same priority",
            )

        # 5. 唯一（或行为一致）的最高优先级规则 → 采用其行为
        rule = candidates[0]
        return PermissionDecision(
            behavior=rule.behavior,
            reason=f"Matched permission rule: {rule.id}",
            rule_id=rule.id,
        )
