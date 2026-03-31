"""
ReAct Agent - 推理与行动循环

支持功能：
- ReAct 循环（Reasoning + Acting）
- Checkpoint（快照/回退）
- Interrupt/Resume（中断/恢复）

中断配置：
    agent = ReActAgent(..., interrupt_before=["dangerous_tool"])
    # 或者
    agent = ReActAgent(..., interrupt_all=True)
"""
import asyncio
from typing import Generator
from packages.core.tool import Tool
from packages.core.prompt import prompts as core_prompts
from packages.core.checkpoint import Checkpoint
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
    2. 行动：调用工具获取信息（可能在此中断等待确认）
    3. 观察：处理工具返回结果
    4. 重复：直到能给出最终答案

    Agent 本身只做容器和代理，实际执行逻辑委托给 ReActRunner
    """

    def __init__(
        self,
        client,
        model: str,
        system_prompt: str = None,
        tools: list[Tool] = None,
        max_iterations: int = 10,
        interrupt_before: list[str] = None,
        interrupt_all: bool = False,
        memory=None,
    ):
        """
        Args:
            client:           OpenAI 客户端
            model:            模型名称
            system_prompt:    系统提示词
            tools:            工具列表
            max_iterations:   最大迭代次数
            interrupt_before: 需要中断确认的工具名列表
            interrupt_all:    是否所有工具都需要中断确认
            memory:           自定义 Memory 管理器 (实现 MemoryProtocol)
        """
        default_prompt = core_prompts.get("react_system")

        super().__init__(
            client=client,
            model=model,
            system_prompt=system_prompt or default_prompt,
            tools=tools,
            memory=memory,
        )
        self.max_iterations = max_iterations

        # 中断配置
        self.interrupt_before: list[str] = interrupt_before or []
        self.interrupt_all: bool = interrupt_all

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

    def resume(self, approved: bool = True) -> Generator[AgentEvent, None, None]:
        """
        从中断处恢复执行

        Args:
            approved: 是否批准执行被中断的工具
                True  → 执行工具，继续 ReAct 循环
                False → 跳过工具（告知 LLM 用户拒绝），继续循环
        """
        yield from self._runner.resume(approved)

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
        同时强关 LLM HTTP 连接，避免线程卡在等待 chunk 上。
        """
        self._runner.state.cancel()
        self._runner.llm.cancel()


    # ============================================================
    # Checkpoint API
    # ============================================================

    def list_checkpoints(self) -> list[Checkpoint]:
        """列出所有快照"""
        return self.memory.checkpoints.list()

    def rollback(self, checkpoint_id: str) -> Checkpoint:
        """
        回退到指定快照

        恢复消息历史和 token 统计到快照时的状态，
        该快照之后的所有消息和快照都会被丢弃。
        """
        cp = self.memory.restore_checkpoint(checkpoint_id)
        self._runner.state.reset()
        return cp

    def rollback_last(self) -> Checkpoint | None:
        """
        回退到上一个快照（撤销最后一轮对话）
        """
        checkpoints = self.memory.checkpoints.list()
        if not checkpoints:
            return None

        if len(checkpoints) >= 2:
            target = checkpoints[-2]
        else:
            target = checkpoints[0]

        return self.rollback(target.id)
