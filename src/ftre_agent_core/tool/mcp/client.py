"""
MCPClient - MCP 服务器客户端

支持三种传输模式：
1. stdio  — 启动本地子进程通信（Cursor / Claude Desktop 主流方式）
2. sse    — 连接远程 SSE 服务（阿里云 DashScope 等）
3. http   — Streamable HTTP（MCP 新协议）

配置示例：

    # stdio
    MCPServerConfig(name="github", command="uvx", args=["mcp-github"])

    # sse（带认证 headers）
    MCPServerConfig(name="thinking", type="sse",
        url="https://dashscope.aliyuncs.com/api/v1/mcps/Sequential_Thinking/sse",
        headers={"Authorization": "Bearer sk-xxx"})

    # http
    MCPServerConfig(name="weather", type="http",
        url="http://localhost:8000/mcp")
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any
from contextlib import AsyncExitStack

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp import types

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """
    MCP Server 配置

    三种模式：
    1. stdio:  指定 command + args（type 可省略，自动推断）
    2. sse:    指定 url + type="sse"，可带 headers
    3. http:   指定 url + type="http"（或不指定 type，默认 http）

    配置格式兼容 Cursor / Cherry Studio / Claude Desktop 的写法。
    """
    name: str

    # 传输类型：stdio / sse / http，None 时自动推断
    type: str | None = None

    # stdio 模式
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None

    # sse / http 模式
    url: str | None = None
    headers: dict[str, str] | None = None

    @property
    def transport_type(self) -> str:
        """推断传输类型"""
        if self.type:
            return self.type
        if self.command:
            return "stdio"
        if self.url:
            return "http"
        raise ValueError(f"MCPServerConfig '{self.name}': 无法推断传输类型")

    def validate(self) -> None:
        t = self.transport_type
        if t == "stdio" and not self.command:
            raise ValueError(f"MCPServerConfig '{self.name}': stdio 模式必须指定 command")
        if t in ("sse", "http") and not self.url:
            raise ValueError(f"MCPServerConfig '{self.name}': {t} 模式必须指定 url")


class MCPClient:
    """
    MCP 客户端

    管理与一个 MCP Server 的连接，提供工具发现和调用能力。
    """

    def __init__(self, config: MCPServerConfig):
        config.validate()
        self.config = config
        self.name = config.name

        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """连接到 MCP Server"""
        if self._connected:
            return

        # AsyncExitStack 管理所有异步上下文（transport、session）
        # 断开时会按 LIFO 顺序自动清理，不用手动一个个 close
        self._exit_stack = AsyncExitStack()
        transport = self.config.transport_type

        try:
            # --------------------------------------------------------
            # 第一步：根据传输类型建立底层通信通道
            # 三种方式最终都产出 (read, write) 两个流对象，
            # 后面统一交给 ClientSession 做 MCP 协议层通信
            # --------------------------------------------------------

            if transport == "stdio":
                # stdio: 启动子进程，通过 stdin/stdout 通信
                # 比如 command="uvx", args=["mcp-github"]
                # 会启动 `uvx mcp-github` 子进程，用管道收发 JSON-RPC
                import sys
                params = StdioServerParameters(
                    command=self.config.command,
                    args=self.config.args,
                    env=self.config.env,
                )
                read, write = await self._exit_stack.enter_async_context(
                    stdio_client(params, errlog=sys.stderr)
                )

            elif transport == "sse":
                # SSE: 连接远程 HTTP 服务，用 Server-Sent Events 接收响应
                # 典型场景：阿里云 DashScope 托管的 MCP 服务
                # headers 用于传 Bearer Token 认证
                read, write = await self._exit_stack.enter_async_context(
                    sse_client(
                        url=self.config.url,
                        headers=self.config.headers,
                    )
                )

            elif transport == "http":
                # Streamable HTTP: MCP 新协议，替代 SSE
                # 返回三个值，第三个是 session_id，暂时用不到
                read, write, _ = await self._exit_stack.enter_async_context(
                    streamablehttp_client(
                        url=self.config.url,
                        headers=self.config.headers,
                    )
                )

            else:
                raise ValueError(f"不支持的传输类型: {transport}")

            # --------------------------------------------------------
            # 第二步：在通信通道上建立 MCP 会话
            # ClientSession 封装了 MCP 协议（JSON-RPC），
            # 提供 list_tools() / call_tool() 等高层 API
            # initialize() 完成 MCP 握手（交换版本和能力）
            # --------------------------------------------------------
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await self._session.initialize()
            self._connected = True
            logger.info(f"MCP 已连接: {self.name} ({transport})")

        except Exception as e:
            # 连接失败时清理已建立的资源，避免泄漏
            await self._cleanup()
            raise ConnectionError(f"MCP 连接失败 [{self.name}]: {e}") from e

    async def disconnect(self) -> None:
        """断开连接"""
        await self._cleanup()
        logger.info(f"MCP 已断开: {self.name}")

    async def list_tools(self) -> list[types.Tool]:
        """获取 MCP Server 提供的工具列表"""
        self._ensure_connected()
        result = await self._session.list_tools()
        return result.tools

    async def call_tool(self, name: str, arguments: dict[str, Any] = None) -> str:
        """
        调用 MCP 工具

        Returns:
            工具执行结果（文本）
        """
        self._ensure_connected()
        result = await self._session.call_tool(name, arguments or {})

        texts = []
        for content in result.content:
            if isinstance(content, types.TextContent):
                texts.append(content.text)
            elif isinstance(content, types.ImageContent):
                texts.append(f"[图片: {content.mimeType}]")
            elif isinstance(content, types.EmbeddedResource):
                texts.append(f"[资源: {content.resource.uri}]")
            else:
                texts.append(str(content))

        return "\n".join(texts) if texts else "(无输出)"

    def _ensure_connected(self) -> None:
        if not self._connected or self._session is None:
            raise RuntimeError(f"MCP 未连接: {self.name}，请先调用 connect()")

    async def _cleanup(self) -> None:
        self._session = None
        self._connected = False
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception:
                pass
            self._exit_stack = None

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"MCPClient(name={self.name!r}, transport={self.config.transport_type}, {status})"