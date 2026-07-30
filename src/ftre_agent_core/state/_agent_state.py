"""Serializable state owned by a ReAct agent."""
from pydantic import BaseModel, Field

from ..message import Msg


class AgentState(BaseModel):
    """Persistent agent data that can be injected into a new agent instance."""

    context: list[Msg] = Field(default_factory=list)
