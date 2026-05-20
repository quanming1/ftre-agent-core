"""
Tool 定义 - 基类、参数、装饰器、依赖注入
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Callable, Any, get_type_hints
from dataclasses import dataclass


# ============================================================
# 依赖注入标记
# ============================================================

class Injected:
    """注入标记。作为参数默认值，标记该参数需要从 ToolRegistry 注入，不暴露给 LLM。"""

    def __init__(self, key: str):
        self.key = key

    def __repr__(self) -> str:
        return f"Injected({self.key!r})"


# ============================================================
# 参数定义
# ============================================================

@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: str  # string, number, boolean, array, object
    description: str
    required: bool = True
    enum: list = None


# ============================================================
# Tool 基类
# ============================================================

class Tool:
    """
    工具基类

    用法：
    1) 装饰器: @tool() def fn(...) -> str: ...
    2) 手动构造: Tool(name=..., func=fn, parameters=[...])
    3) 子类继承: class MyTool(Tool): def _run(self, **kwargs): ...
    """

    name: str = ""
    description: str = ""
    parameters: list[ToolParameter] = []

    def __init__(self, name: str = None, description: str = None, parameters: list[ToolParameter] = None, func: Callable[..., Any] = None):
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if parameters is not None:
            self.parameters = parameters
        self.func = func

    def to_openai_dict(self) -> dict:
        """转换为 OpenAI function calling 格式"""
        properties = {}
        required = []
        for param in self.parameters:
            prop = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        }

    def _get_callable(self) -> Callable[..., Any]:
        if type(self)._run is not Tool._run:
            return self._run
        if self.func is not None:
            return self.func
        raise NotImplementedError(f"Tool '{self.name}' 必须实现 _run() 或提供 func")

    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self._get_callable())

    def execute(self, **kwargs) -> Any:
        fn = self._get_callable()
        if inspect.iscoroutinefunction(fn):
            return asyncio.run(fn(**kwargs))
        return fn(**kwargs)

    def _run(self, **kwargs) -> Any:
        raise NotImplementedError


# ============================================================
# @tool() 装饰器
# ============================================================

def tool(name: str = None, description: str = None, parameters: list[ToolParameter] = None):
    """装饰器：将函数转换为 Tool"""
    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or ""
        tool_params = parameters if parameters is not None else _infer_parameters(func)
        return Tool(name=tool_name, description=tool_desc.strip(), parameters=tool_params, func=func)
    return decorator


def _infer_parameters(func: Callable) -> list[ToolParameter]:
    """从函数签名推断参数"""
    params = []
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if isinstance(param.default, Injected):
            continue

        param_type = "string"
        if param_name in hints:
            param_type = _python_type_to_json_type(hints[param_name])

        required = param.default is inspect.Parameter.empty
        params.append(ToolParameter(name=param_name, type=param_type, description=f"参数 {param_name}", required=required))

    return params


_TYPE_MAP = {str: "string", int: "number", float: "number", bool: "boolean", list: "array", dict: "object"}


def _python_type_to_json_type(python_type) -> str:
    origin = getattr(python_type, "__origin__", None)
    if origin is list:
        return "array"
    if origin is dict:
        return "object"
    return _TYPE_MAP.get(python_type, "string")
