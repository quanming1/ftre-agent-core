"""
Tool 装饰器 - 简化工具创建
"""
import inspect
from typing import Callable, get_type_hints
from .base import Tool, ToolParameter
from .inject import Injected

def tool(
    name: str = None,
    description: str = None,
    parameters: list[ToolParameter] = None
):
    """
    装饰器：将函数转换为 Tool

    用法1 - 手动指定参数:
        @tool(
            name="get_weather",
            description="获取天气",
            parameters=[
                ToolParameter(name="city", type="string", description="城市名")
            ]
        )
        def get_weather(city: str) -> str:
            return f"{city}的天气是晴天"

    用法2 - 自动推断参数:
        @tool()
        def get_weather(city: str) -> str:
            '''获取指定城市的天气'''
            return f"{city}的天气是晴天"
    """
    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or ""

        # 如果没有指定参数，尝试从函数签名推断
        tool_params = parameters
        if tool_params is None:
            tool_params = _infer_parameters(func)

        return Tool(
            name=tool_name,
            description=tool_desc.strip(),
            parameters=tool_params,
            func=func
        )

    return decorator

def _infer_parameters(func: Callable) -> list[ToolParameter]:
    """从函数签名推断参数"""
    params = []
    sig = inspect.signature(func)

    # 尝试获取类型注解
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    for param_name, param in sig.parameters.items():
        # 跳过 self, *args, **kwargs
        if param_name in ("self", "cls"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue

        # 跳过 Injected 标记的参数（依赖注入，不暴露给 LLM）
        if isinstance(param.default, Injected):
            continue

        # 推断类型
        param_type = "string"  # 默认类型
        if param_name in hints:
            hint = hints[param_name]
            param_type = _python_type_to_json_type(hint)

        # 判断是否必需
        required = param.default is inspect.Parameter.empty

        params.append(ToolParameter(
            name=param_name,
            type=param_type,
            description=f"参数 {param_name}",
            required=required
        ))

    return params

def _python_type_to_json_type(python_type) -> str:
    """Python 类型转 JSON Schema 类型"""
    type_map = {
        str: "string",
        int: "number",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    # 处理 Optional, Union 等
    origin = getattr(python_type, "__origin__", None)
    if origin is not None:
        # 如 list[str] -> array
        if origin is list:
            return "array"
        # 如 dict[str, Any] -> object
        if origin is dict:
            return "object"

    return type_map.get(python_type, "string")
