"""Serializable state owned by a ReAct agent."""
from pydantic import BaseModel, Field

from ..message import Msg
from ..permission import PermissionContext


class AgentState(BaseModel):
    """Persistent agent data that can be injected into a new agent instance."""

    context: list[Msg] = Field(default_factory=list)
    # 权限上下文（规则的持久化事实源，类型化模型）。
    # 用户点"永久允许/拒绝"时由 ActingExecutor 往 permission_rules 追加规则。
    permission_context: PermissionContext = Field(default_factory=PermissionContext)
