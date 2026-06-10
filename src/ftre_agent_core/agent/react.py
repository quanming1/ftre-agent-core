"""
ReAct Agent - 异步推理与行动循环
"""
from typing import AsyncGenerator

from ftre_agent_core.tool import Tool, ToolRegistry
from ftre_agent_core.memory import MemoryManager
from .event import AgentEvent
from .runner import ReActRunner
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
        tools: list[Tool] = None,
        max_iterations: int | None = None,
        memory: MemoryManager | None = None,
        max_retries: int = 5,
        retry_delay: float = 3.0,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.api_type = api_type
        self.max_iterations = max_iterations
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if memory is not None:
            self.memory = memory
            if system_prompt:
                self.memory.system_prompt = system_prompt
        else:
            self.memory = MemoryManager({"system_prompt": system_prompt})

        self._registry = ToolRegistry()
        if tools:
            for t in tools:
                self._registry.register(t)

        self._runner = ReActRunner(self)

    @property
    def system_prompt(self) -> str:
        return self.memory.system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        self.memory.system_prompt = value

    @property
    def tools(self) -> ToolRegistry:
        return self._registry

    @property
    def runner(self) -> ReActRunner:
        return self._runner

    @property
    def state(self):
        return self._runner.state

    def add_tool(self, tool: Tool) -> None:
        self._registry.register(tool)

    def remove_tool(self, name: str) -> None:
        self._registry.unregister(name)

    async def run(
        self, message, runtime_context: dict | None = None
    ) -> AsyncGenerator[AgentEvent, None]:
        async for event in self._runner.run(message, runtime_context=runtime_context):
            yield event

    def cancel_nowait(self) -> None:
        self._runner.cancel()
