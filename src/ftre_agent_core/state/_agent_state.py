"""Serializable state owned by a ReAct agent."""
from typing import Any

from pydantic import BaseModel, Field

from ..message import Msg


class AgentState(BaseModel):
    """Persistent agent data that can be injected into a new agent instance."""

    context: list[Msg] = Field(default_factory=list)
    # 权限上下文（路线 B：规则的持久化事实源）。
    # 约定 key：
    #   - "permission_rules": list[dict]  规则序列化后的列表（PermissionRule.model_dump）
    #   - "default_behavior": str         无规则命中时的兜底行为（PermissionBehavior 值）
    # 用户点"永久允许/拒绝"时由 ActingExecutor 往 permission_rules 追加规则。
    permission_context: dict[str, Any] = Field(default_factory=dict)
