"""
ReAct Agent - 推理与行动循环
"""
import asyncio
from typing import Generator
from ftre_agent_core.tool import Tool, ToolRegistry
from ftre_agent_core.memory import MemoryManager
from .event import AgentEvent
from .runner import ReActRunner
import logging

logger = logging.getLogger(__name__)


class ReActAgent:
    """
    ReAct Agent

    实现 Reasoning + Acting 循环：
    1. 思考：分析问题，决定是否需要工具
    2. 行动：调用工具获取信息
    3. 观察：处理工具返回结果
    4. 重复：直到能给出最终答案
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str | None = None,
        api_type: str = "completions",
        system_prompt: str = "",
        tools: list[Tool] = None,
        max_iterations: int = 10,
        memory: MemoryManager | None = None,
    ):
        """
        Args:
            model:            模型名称（LiteLLM 格式，如 "openai/gpt-4"）
            api_key:          API 密钥
            api_base:         自定义端点（可选）
            api_type:         协议类型，"completions"（默认）或 "responses"
            system_prompt:    系统提示词
            tools:            工具列表
            max_iterations:   最大迭代次数
            memory:           自定义 MemoryManager
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.api_type = api_type
        self.max_iterations = max_iterations

        # Memory
        if memory is not None:
            self.memory = memory
            if system_prompt:
                self.memory.system_prompt = system_prompt
        else:
            self.memory = MemoryManager({"system_prompt": system_prompt})

        # 工具注册表
        self._registry = ToolRegistry()
        if tools:
            for t in tools:
                self._registry.register(t)

        # 执行引擎
        self._runner = ReActRunner(self)

    # ============================================================
    # 属性
    # ============================================================

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

    # ============================================================
    # 工具管理
    # ============================================================

    def add_tool(self, tool: Tool) -> None:
        self._registry.register(tool)

    def remove_tool(self, name: str) -> None:
        self._registry.unregister(name)

    # ============================================================
    # 执行
    # ============================================================

    def run(self, message) -> Generator[AgentEvent, None, None]:
        """
        运行 ReAct 循环，返回事件迭代器。

        Args:
            message: str 或 list[dict]
                - str: 单条用户消息
                - list: 完整消息列表（含历史 + 当前用户消息）
        """
        yield from self._runner.run(message)

    # ============================================================
    # 取消
    # ============================================================

    async def cancel(self) -> None:
        """异步取消（等待善后完成）"""
        await asyncio.to_thread(self._runner.cancel)

    def cancel_sync(self) -> None:
        """同步取消（阻塞等待善后完成）"""
        self._runner.cancel()

    def cancel_nowait(self) -> None:
        """仅发取消信号，不等待善后"""
        self._runner.state.cancel()
        self._runner.llm.cancel()
