"""
Tool 定义：基类、参数定义、装饰器和依赖注入标记。
"""
from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints


# 依赖注入标记
class Injected:
    """标记一个参数需要从 ToolRegistry 的 runtime_context 注入，不暴露给 LLM。"""

    def __init__(self, key: str):
        self.key = key

    def __repr__(self) -> str:
        return f"Injected({self.key!r})"


# 工具参数定义
@dataclass
class ToolParameter:
    """单个工具参数的 OpenAI schema 描述。"""
    name: str
    type: str  # string / number / boolean / array / object
    description: str
    required: bool = True
    enum: list = None


# 工具基类
class Tool:
    """
    工具基类。

    支持三种使用方式：
    1. 使用 @tool() 装饰普通函数。
    2. 手动构造 Tool(name=..., func=..., parameters=...)。
    3. 继承 Tool 并实现 _run()。
    """

    name: str = ""
    description: str = ""
    parameters: list[ToolParameter] = []

    def __init__(
        self,
        name: str = None,
        description: str = None,
        parameters: list[ToolParameter] = None,
        func: Callable[..., Any] = None,
    ):
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if parameters is not None:
            self.parameters = parameters
        self.func = func

    def to_openai_dict(self) -> dict:
        """转换成 OpenAI function calling 工具定义。"""
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
        """同步执行工具。

        对异步工具来说，这个方法只适合在没有运行中 event loop 的同步上下文调用。
        如果已经处在异步上下文中，应该使用 ``await tool_handler.run_one(...)``。
        run_one() 会对同步工具走线程池，对异步工具直接 await。
        """
        fn = self._get_callable()
        if inspect.iscoroutinefunction(fn):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None and loop.is_running():
                raise RuntimeError(
                    f"Tool '{self.name}' is async. "
                    "Call it via ToolHandler.run_one() inside an async context, "
                    "not Tool.execute() which is for synchronous callers only."
                )
            return asyncio.run(fn(**kwargs))
        return fn(**kwargs)

    def _run(self, **kwargs) -> Any:
        raise NotImplementedError


# @tool() 装饰器
def tool(name: str = None, description: str = None, parameters: list[ToolParameter] = None):
    """把普通函数转换成 Tool 对象。"""
    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or ""
        tool_params = parameters if parameters is not None else _infer_parameters(func)
        return Tool(name=tool_name, description=tool_desc.strip(), parameters=tool_params, func=func)
    return decorator


def _infer_parameters(func: Callable) -> list[ToolParameter]:
    """根据函数签名推断工具参数。"""
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
        params.append(
            ToolParameter(
                name=param_name,
                type=param_type,
                description=f"参数 {param_name}",
                required=required,
            )
        )

    return params


_TYPE_MAP = {str: "string", int: "number", float: "number", bool: "boolean", list: "array", dict: "object"}


def _python_type_to_json_type(python_type) -> str:
    """把 Python 类型映射成 JSON schema 类型。"""
    origin = getattr(python_type, "__origin__", None)
    if origin is list:
        return "array"
    if origin is dict:
        return "object"
    return _TYPE_MAP.get(python_type, "string")
