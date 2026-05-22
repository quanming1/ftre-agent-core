"""
ToolRegistry - 工具注册表 + 中间件
"""
import inspect
from abc import ABC
from dataclasses import dataclass, field
from typing import Any

from .base import Tool, Injected
from .cancellation import CancellationToken


# ============================================================
# 中间件
# ============================================================

@dataclass
class ToolContext:
    """单次 tool 调用的上下文"""
    call_id: str
    name: str
    arguments: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    cancel_token: CancellationToken = field(default_factory=CancellationToken)

    _skipped: bool = field(default=False, repr=False)
    _skip_result: str = field(default="", repr=False)

    @property
    def skipped(self) -> bool:
        return self._skipped

    @property
    def skip_result(self) -> str:
        return self._skip_result

    def skip(self, result: str = "") -> None:
        self._skipped = True
        self._skip_result = result


class ToolMiddleware(ABC):
    """Tool 中间件基类"""

    def before(self, context: ToolContext) -> ToolContext:
        return context

    def after(self, context: ToolContext, result) -> Any:
        return result


# ============================================================
# 注册表
# ============================================================

class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._middlewares: list[ToolMiddleware] = []
        self._inject_map: dict[str, dict[str, str]] = {}

    # --- 中间件 ---

    def add_middleware(self, middleware: ToolMiddleware) -> None:
        self._middlewares.append(middleware)

    def remove_middleware(self, middleware: ToolMiddleware) -> None:
        self._middlewares.remove(middleware)

    @property
    def middlewares(self) -> list[ToolMiddleware]:
        return list(self._middlewares)

    # --- 工具管理 ---

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._inject_map[tool.name] = self._parse_injections(tool)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._inject_map.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    # --- 注入 ---

    def _resolve_injections(self, name: str, kwargs: dict, runtime_context: dict | None) -> dict:
        inject_map = self._inject_map.get(name)
        if not inject_map:
            return kwargs
        ctx = runtime_context or {}
        merged = dict(kwargs)
        for param_name, inject_key in inject_map.items():
            if param_name not in merged:
                merged[param_name] = ctx.get(inject_key)
        return merged

    @staticmethod
    def _parse_injections(tool: Tool) -> dict[str, str]:
        func = tool.func if tool.func is not None else (type(tool)._run if type(tool)._run is not Tool._run else None)
        if func is None:
            return {}
        result = {}
        try:
            for param_name, param in inspect.signature(func).parameters.items():
                if isinstance(param.default, Injected):
                    result[param_name] = param.default.key
        except (ValueError, TypeError):
            pass
        return result

    # --- 执行 ---

    def execute(self, name: str, runtime_context: dict | None = None, **kwargs) -> Any:
        tool = self.get(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found")
        return tool.execute(**self._resolve_injections(name, kwargs, runtime_context))

    # --- 导出 ---

    def to_openai_tools(self) -> list[dict]:
        return [tool.to_openai_dict() for tool in self._tools.values()]

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())
