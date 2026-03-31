"""
Agent 基类
"""
from abc import ABC, abstractmethod
from ftre_agent_core.tool import Tool, ToolRegistry
from ftre_agent_core.memory import MemoryManager
from ftre_agent_core.memory.protocol import MemoryProtocol


class Agent(ABC):
    """Agent 基类"""

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str | None = None,
        system_prompt: str = "你是一个有帮助的助手。",
        tools: list[Tool] = None,
        memory: MemoryProtocol | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base

        # 记忆管理器: 外部传入或使用默认实现
        if memory is not None:
            self.memory: MemoryProtocol = memory
        else:
            self.memory: MemoryProtocol = MemoryManager({
                "system_prompt": system_prompt
            })

        # 如果外部传了 memory 但没设 system_prompt，补上
        if memory and system_prompt != "你是一个有帮助的助手。":
            self.memory.system_prompt = system_prompt

        # 内部构建 ToolRegistry
        self._registry = ToolRegistry()
        if tools:
            for tool in tools:
                self._registry.register(tool)

    @property
    def system_prompt(self) -> str:
        """获取系统提示词"""
        return self.memory.system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        """设置系统提示词"""
        self.memory.system_prompt = value

    @property
    def tools(self) -> ToolRegistry:
        """获取工具注册表"""
        return self._registry

    def add_tool(self, tool: Tool) -> None:
        """添加工具"""
        self._registry.register(tool)

    def remove_tool(self, name: str) -> None:
        """移除工具"""
        self._registry.unregister(name)

    @abstractmethod
    def run(self, message) -> str:
        """
        运行 Agent 处理消息。

        Args:
            message: 用户消息，支持两种格式：
                - str: 单条用户消息（自动包装为 messages 列表）
                - list[dict]: 完整消息列表（含历史），每条为 openai 格式 dict
                  或 FtreMessage 对象
        """
        pass
