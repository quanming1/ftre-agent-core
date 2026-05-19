"""
ReAct Agent - 推理与行动循环
"""
import asyncio
from typing import Generator
from ftre_agent_core.tool import Tool
from ftre_agent_core.prompt import prompts as core_prompts
from .base import Agent
from .event import AgentEvent
from .runner import ReActRunner
import logging

logger = logging.getLogger(__name__)

class ReActAgent(Agent):
    """
    ReAct Agent

    实现 Reasoning + Acting 循环：
    1. 思考：分析问题，决定是否需要工具
    2. 行动：调用工具获取信息
    3. 观察：处理工具返回结果
    4. 重复：直到能给出最终答案

    Agent 本身只做容器和代理，实际执行逻辑委托给 ReActRunner
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str | None = None,
        api_type: str = "completions",
        system_prompt: str = None,
        tools: list[Tool] = None,
        max_iterations: int = 10,
        memory=None,
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
        default_prompt = core_prompts.get("react_system")

        super().__init__(
            model=model,
            api_key=api_key,
            api_base=api_base,
            api_type=api_type,
            system_prompt=system_prompt or default_prompt,
            tools=tools,
            memory=memory,
        )
        self.max_iterations = max_iterations

        # 执行引擎
        self._runner = ReActRunner(self)

    @property
    def runner(self) -> ReActRunner:
        """获取执行引擎"""
        return self._runner

    @property
    def state(self):
        """获取运行状态（代理到 runner）"""
        return self._runner.state

    def run(self, message) -> Generator[AgentEvent, None, None]:
        """
        运行 ReAct 循环，返回事件迭代器。

        Args:
            message: str 或 list[dict]
                - str: 单条用户消息
                - list: 完整消息列表（含历史 + 当前用户消息）
        """
        yield from self._runner.run(message)

    async def cancel(self) -> None:
        """
        用户主动取消当前执行（异步，等待善后完成）。

        await 返回时，runner 已完成所有善后工作（Memory 写入、tool results 补齐）。
        """
        await asyncio.to_thread(self._runner.cancel)

    def cancel_sync(self) -> None:
        """
        同步版取消（阻塞等待善后完成）。

        适用于 CLI 等非 async 场景。
        """
        self._runner.cancel()

    def cancel_nowait(self) -> None:
        """
        只发取消信号，不等待善后完成。

        适用于 CompiledGraph 等不能阻塞的场景，
        善后由 generator 消费链路自然完成。
        """
        self._runner.state.cancel()
        self._runner.llm.cancel()
