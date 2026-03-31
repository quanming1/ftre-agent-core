"""
Tool 注册表 - 管理所有工具
"""
import inspect
from typing import Any, Callable

from .base import Tool
from .inject import Injected
from .middleware import ToolMiddleware


class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._middlewares: list[ToolMiddleware] = []
        self._injections: dict[str, Callable[[], Any]] = {}
        self._inject_map: dict[str, dict[str, str]] = {}

    def add_middleware(self, middleware: ToolMiddleware) -> None:
        """注册中间件（按注册顺序执行 before，逆序执行 after）"""
        self._middlewares.append(middleware)

    def remove_middleware(self, middleware: ToolMiddleware) -> None:
        """移除中间件"""
        self._middlewares.remove(middleware)

    @property
    def middlewares(self) -> list[ToolMiddleware]:
        return list(self._middlewares)

    def register(self, tool: Tool) -> None:
        """注册工具"""
        self._tools[tool.name] = tool
        self._inject_map[tool.name] = self._parse_injections(tool)

    def unregister(self, name: str) -> None:
        """注销工具"""
        if name in self._tools:
            del self._tools[name]
        self._inject_map.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """获取工具"""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """检查工具是否存在"""
        return name in self._tools

    def provide(self, key: str, provider: Callable[[], Any]) -> None:
        """
        注册注入源

        Args:
            key: 注入键名，与 Injected("key") 对应
            provider: 工厂函数，每次 tool 执行时调用取值
        """
        self._injections[key] = provider

    def _resolve_injections(self, name: str, kwargs: dict) -> dict:
        """解析注入参数，合并到 kwargs 中"""
        inject_map = self._inject_map.get(name)
        if not inject_map:
            return kwargs

        merged = dict(kwargs)
        for param_name, inject_key in inject_map.items():
            if param_name in merged:
                continue
            provider = self._injections.get(inject_key)
            if provider is not None:
                merged[param_name] = provider()
        return merged

    @staticmethod
    def _parse_injections(tool: Tool) -> dict[str, str]:
        """
        解析 tool 的函数签名，提取 Injected 参数映射。

        Returns:
            {param_name: inject_key} 映射，无注入参数时返回空 dict
        """
        func = None
        if tool.func is not None:
            func = tool.func
        elif type(tool)._run is not Tool._run:
            func = type(tool)._run

        if func is None:
            return {}

        result = {}
        try:
            sig = inspect.signature(func)
            for param_name, param in sig.parameters.items():
                if isinstance(param.default, Injected):
                    result[param_name] = param.default.key
        except (ValueError, TypeError):
            pass
        return result

    def execute(self, name: str, **kwargs) -> Any:
        """执行指定工具（自动注入依赖）"""
        tool = self.get(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found")
        merged = self._resolve_injections(name, kwargs)
        return tool.execute(**merged)

    async def execute_async(self, name: str, **kwargs) -> Any:
        """异步执行指定工具（自动注入依赖），兼容 sync / async。"""
        tool = self.get(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found")
        merged = self._resolve_injections(name, kwargs)
        return await tool.execute_async(**merged)

    def to_openai_tools(self) -> list[dict]:
        """转换所有工具为 OpenAI 格式"""
        return [tool.to_openai_dict() for tool in self._tools.values()]

    @property
    def names(self) -> list[str]:
        """获取所有工具名称"""
        return list(self._tools.keys())

    @property
    def tools(self) -> list[Tool]:
        """获取所有工具"""
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())
