"""权限领域模型（纯数据结构，不含决策逻辑）。

本模块定义工具调用权限决策的"词汇表"：请求、规则与决策结果。
规则以 ``list[PermissionRule]`` 的形式持久化在 AgentState 上，本模块不提供聚合容器。
决策逻辑由同包的 :class:`PermissionEngine`（``_engine.py``）负责，本模块只放模型。

第一版范围：
  - 只有三种行为：ALLOW（放行）/ DENY（拒绝）/ ASK（询问用户）。
  - 规则按工具名精确匹配，或用 "*" 通配任意工具。
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PermissionBehavior(StrEnum):
    """工具调用权限决策的三种可能结果。"""

    ALLOW = "allow"  # 放行：直接执行工具
    DENY = "deny"    # 拒绝：不执行，产生拒绝结果
    ASK = "ask"      # 询问：暂停执行，等待用户确认


class PermissionRequest(BaseModel):
    """一次待决策的工具调用（描述"谁想执行什么"）。"""

    tool_name: str  # 工具名
    arguments: dict[str, Any] = Field(default_factory=dict)  # 调用参数


class PermissionRule(BaseModel):
    """一条按工具名及可选参数正则匹配的权限规则。

    ``tool_name`` 是精确工具名，或用 ``"*"`` 匹配任意工具。
    ``argument_regex`` 为空时只匹配工具名；非空时每个参数都必须通过
    对应正则的 ``fullmatch``。
    """

    id: str                       # 规则唯一标识（也用于决策溯源）
    tool_name: str                # 精确工具名，或 "*" 通配
    argument_regex: dict[str, str] = Field(default_factory=dict)
    behavior: PermissionBehavior  # 命中后采取的行为
    priority: int = 0             # 优先级，数值越大越优先
    enabled: bool = True          # 是否启用；关闭的规则不参与匹配


class PermissionDecision(BaseModel):
    """对某次请求按策略求值后的结果。"""

    model_config = ConfigDict(use_enum_values=True)

    behavior: PermissionBehavior  # 最终行为
    reason: str                   # 可读的判定原因（便于日志与 UI 展示）
    rule_id: str | None = None    # 命中的规则 id；走默认或冲突时为 None
