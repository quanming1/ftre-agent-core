"""
MCPAdapter - 将 MCP 工具转换为本地 Tool 对象

职责：
- 连接 MCP Server
- 把 MCP 工具描述转成 core.tool.Tool
- 注册到 ToolRegistry
- Agent 调用时，透明地通过 MCP 协议远程执行

对上层（Agent / ToolRegistry）完全透明，
MCP 工具和本地 @tool() 工具用法完全一致。

async → sync 桥接策略：
    MCP 的 call_tool 是 async 的，但 ToolRegistry.execute() 是 sync 的。
    桥接方式：connect() 时记住事件循环，call 时用 run_coroutine_threadsafe
    把协程提交回原 loop 执行（因为 stdio transport 绑定在那个 loop 上）。
"""
import asyncio
import logging
from typing import Any

from ..base import Tool, ToolParameter
from .client import MCPClient, MCPServerConfig

logger = logging.getLogger(__name__)


class MCPAdapter:
    """
    MCP 适配器

    将一个或多个 MCP Server 的工具转换为本地 Tool 对象。

    用法：
        adapter = MCPAdapter()
        adapter.add_server(MCPServerConfig(name="github", command="uvx", args=["mcp-github"]))

        # 连接所有 Server，获取工具
        tools = await adapter.connect()
        # tools 是 list[Tool]，可以直接传给 ReActAgent

        # 用完断开
        await adapter.disconnect()
    """

    def __init__(self):
        self._clients: dict[str, MCPClient] = {}
        self._tools: list[Tool] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def add_server(self, config: MCPServerConfig) -> None:
        """添加一个 MCP Server 配置"""
        if config.name in self._clients:
            raise ValueError(f"MCP Server '{config.name}' 已存在")
        self._clients[config.name] = MCPClient(config)

    async def connect(self) -> list[Tool]:
        """
        连接所有 MCP Server，获取并转换工具

        Returns:
            所有 MCP Server 提供的工具列表（已转换为 Tool 对象）
        """
        # 记住当前事件循环，后续 sync 调用时提交回这个 loop
        self._loop = asyncio.get_running_loop()
        self._tools = []

        for name, client in self._clients.items():
            try:
                await client.connect()
                mcp_tools = await client.list_tools()
                converted = self._convert_tools(client, mcp_tools)
                self._tools.extend(converted)
                logger.info(f"MCP [{name}]: 加载了 {len(converted)} 个工具")
            except Exception as e:
                logger.error(f"MCP [{name}] 连接失败: {e}")

        return self._tools

    async def disconnect(self) -> None:
        """断开所有 MCP Server"""
        for client in self._clients.values():
            try:
                await client.disconnect()
            except Exception as e:
                logger.error(f"MCP [{client.name}] 断开失败: {e}")
        self._loop = None

    @property
    def tools(self) -> list[Tool]:
        return self._tools

    @property
    def server_names(self) -> list[str]:
        return list(self._clients.keys())

    # ============================================================
    # 工具转换
    # ============================================================

    def _convert_tools(self, client: MCPClient, mcp_tools: list) -> list[Tool]:
        """将 MCP 工具列表转换为本地 Tool 对象"""
        tools = []
        for mcp_tool in mcp_tools:
            t = self._convert_one(client, mcp_tool)
            if t:
                tools.append(t)
        return tools

    def _convert_one(self, client: MCPClient, mcp_tool) -> Tool | None:
        """将单个 MCP 工具转换为 Tool"""
        try:
            name = mcp_tool.name
            description = mcp_tool.description or ""
            parameters = self._convert_parameters(mcp_tool.inputSchema)

            # 闭包：调用时通过 MCP 协议远程执行
            def make_caller(c: MCPClient, tool_name: str, adapter: "MCPAdapter"):
                def caller(**kwargs) -> str:
                    return adapter._call_sync(c, tool_name, kwargs)
                return caller

            return Tool(
                name=name,
                description=description,
                parameters=parameters,
                func=make_caller(client, name, self),
            )
        except Exception as e:
            logger.warning(f"MCP 工具转换失败 [{mcp_tool.name}]: {e}")
            return None

    def _convert_parameters(self, input_schema: dict | None) -> list[ToolParameter]:
        """将 MCP inputSchema (JSON Schema) 转换为 ToolParameter 列表"""
        if not input_schema:
            return []

        properties = input_schema.get("properties", {})
        required_set = set(input_schema.get("required", []))
        params = []

        for name, prop in properties.items():
            params.append(ToolParameter(
                name=name,
                type=prop.get("type", "string"),
                description=prop.get("description", f"参数 {name}"),
                required=name in required_set,
                enum=prop.get("enum"),
            ))

        return params

    # ============================================================
    # async → sync 桥接
    # ============================================================

    def _call_sync(self, client: MCPClient, tool_name: str, arguments: dict) -> str:
        """
        同步调用 MCP 工具

        策略：把 async 协程提交回 connect() 时记住的事件循环。
        因为 stdio transport 的 read/write 绑定在那个 loop 上，
        必须在同一个 loop 里执行才能正常通信。
        """
        if self._loop is None or self._loop.is_closed():
            return "[错误] MCP 未连接"

        coro = client.call_tool(tool_name, arguments)

        # 如果当前线程就在目标 loop 里（比如在 async 上下文中同步调用）
        # 不能用 run_coroutine_threadsafe（会死锁），直接报错提示
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is self._loop:
            # 当前就在目标 loop 里，不能阻塞等待，需要特殊处理
            # 这种情况出现在 async 代码里直接调 tool.execute()
            # 解决方案：用 nest_asyncio 或改为 await
            try:
                import nest_asyncio
                nest_asyncio.apply()
                return self._loop.run_until_complete(coro)
            except ImportError:
                return "[错误] 在 async 上下文中同步调用 MCP 工具需要安装 nest_asyncio"
        else:
            # 从其他线程提交到目标 loop
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            return future.result(timeout=60)

    def __repr__(self) -> str:
        return f"MCPAdapter(servers={self.server_names}, tools={len(self._tools)})"