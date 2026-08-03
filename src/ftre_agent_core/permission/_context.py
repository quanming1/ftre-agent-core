# -*- coding: utf-8 -*-
"""类型化权限上下文（AgentState.permission_context 的持久化数据模型）。

规则与默认行为的唯一事实源是 ``AgentState.permission_context``，本模型
即为该字段的类型。字段名与历史 JSON 约定一致（``permission_rules`` /
``default_behavior``），旧的 ``state.json`` 对象可由 Pydantic 直接校验，
无需迁移。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ._types import PermissionBehavior, PermissionRule


class PermissionContext(BaseModel):
    """一条 Agent 的权限配置：规则列表 + 无规则命中时的兜底行为。

    ``default_behavior`` 默认 ``ALLOW``：空规则 + ALLOW 表达“不启用拦截”，
    等价于旧设计中未注入 ``PermissionEngine`` 时工具直接执行的行为。
    """

    permission_rules: list[PermissionRule] = Field(default_factory=list)
    default_behavior: PermissionBehavior = PermissionBehavior.ALLOW
