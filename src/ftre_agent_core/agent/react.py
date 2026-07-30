"""ReAct Agent - 异步推理与行动循环。"""
import logging
from typing import AsyncGenerator

from ftre_agent_core.message_context import MessageContext
from ftre_agent_core.state import AgentState
from ftre_agent_core.tool import ToolRegistry
from ftre_agent_core.tracing import Tracer

from ..event import AgentStreamEvent
from ..hooks import FtreCoreHookManager
from .runner import ReActRunner

logger = logging.getLogger(__name__)


class ReActAgent:
    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str | None = None,
        api_type: str = "completions",
        system_prompt: str = "",
        tool_registry: ToolRegistry | None = None,
        max_iterations: int | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str = "",
        state: AgentState | None = None,
        max_retries: int = 5,
        retry_delay: float = 3.0,
        tracer: Tracer | None = None,
        hook_manager: FtreCoreHookManager | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.api_type = api_type
        self._system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.tracer = tracer or Tracer()
        self.hook_manager = hook_manager or FtreCoreHookManager()
        self._state = state if state is not None else AgentState()
        self._registry = tool_registry if tool_registry is not None else ToolRegistry()
        self._runner = ReActRunner(self)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        self._system_prompt = value

    @property
    def state(self) -> AgentState:
        """Persistent, injectable agent state."""
        return self._state

    @property
    def run_state(self):
        """Temporary state for the current or most recent run."""
        return self._runner.state

    @property
    def messages(self) -> list[dict]:
        """Provider-compatible view of the persistent message context."""
        return MessageContext.messages(self._state.context)

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._registry

    @property
    def runner(self) -> ReActRunner:
        return self._runner

    async def run(
        self, message, runtime_context: dict | None = None
    ) -> AsyncGenerator[AgentStreamEvent, None]:
        async for event in self._runner.run(message, runtime_context=runtime_context):
            yield event

    def cancel_nowait(self) -> None:
        self._runner.cancel_nowait()
