"""
Tool 基础类定义

支持两种定义工具的方式：

1. 装饰器方式（适合简单工具）：
    @tool(name="grep", description="搜索文件内容")
    def grep(pattern: str) -> str: ...

2. 继承方式（适合复杂工具，参考 LangChain BaseTool）：
    class MyTool(Tool):
        name = "my_tool"
        description = "做一些事情"
        parameters = [ToolParameter(...)]

        def _run(self, **kwargs) -> str:
            return "result"
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Callable, Any
from dataclasses import dataclass


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: str  # string, number, boolean, array, object
    description: str
    required: bool = True
    enum: list = None


class Tool:
    """
    工具基类

    两种用法：

    1) 数据实例（装饰器 / 手动构造）：
       tool = Tool(name="x", description="y", parameters=[...], func=fn)

    2) 子类继承：
       class MyTool(Tool):
           name = "my_tool"
           description = "..."
           parameters = [ToolParameter(...)]
           def _run(self, **kwargs) -> str: ...
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
        """转换为 OpenAI 标准格式"""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
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
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def _get_callable(self) -> Callable[..., Any]:
        if type(self)._run is not Tool._run:
            return self._run
        if self.func is not None:
            return self.func
        raise NotImplementedError(
            f"Tool '{self.name}' 必须实现 _run() 方法或提供 func 参数"
        )

    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self._get_callable())

    def execute(self, **kwargs) -> Any:
        callable_obj = self._get_callable()
        if inspect.iscoroutinefunction(callable_obj):
            try:
                import nest_asyncio
                nest_asyncio.apply()
                return asyncio.get_event_loop().run_until_complete(callable_obj(**kwargs))
            except Exception as e:
                return f"[错误] async tool bridge failed: {e}"
        return callable_obj(**kwargs)

    async def execute_async(self, **kwargs) -> Any:
        callable_obj = self._get_callable()
        if inspect.iscoroutinefunction(callable_obj):
            return await callable_obj(**kwargs)
        return callable_obj(**kwargs)

    def _run(self, **kwargs) -> Any:
        raise NotImplementedError
