"""
ReAct Agent - 异步推理与行动循环
"""
from typing import AsyncGenerator

from ftre_agent_core.tool import ToolRegistry
from ftre_agent_core.memory import MemoryManager
from ftre_agent_core.tracing import Tracer
from ..event import AgentStreamEvent
from .runner import ReActRunner
from ..hooks import FtreCoreHookManager
import logging

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
        memory: MemoryManager | None = None,
        max_retries: int = 5,
        retry_delay: float = 3.0,
        tracer: Tracer | None = None,
        hook_manager: FtreCoreHookManager | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.api_type = api_type
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.tracer = tracer or Tracer()
        self.hook_manager = hook_manager or FtreCoreHookManager()

        if memory is not None:
            self.memory = memory
            if system_prompt:
                self.memory.system_prompt = system_prompt
        else:
            self.memory = MemoryManager({"system_prompt": system_prompt})

        self._registry = tool_registry if tool_registry is not None else ToolRegistry()

        self._runner = ReActRunner(self)

    @property
    def system_prompt(self) -> str:
        return self.memory.system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        self.memory.system_prompt = value

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._registry

    @property
    def runner(self) -> ReActRunner:
        return self._runner

    @property
    def state(self):
        return self._runner.state

    async def run(
        self, message, runtime_context: dict | None = None
    ) -> AsyncGenerator[AgentStreamEvent, None]:
        async for event in self._runner.run(message, runtime_context=runtime_context):
            yield event

    def cancel_nowait(self) -> None:
        self._runner.cancel()
