"""
测试 Tool 基类
"""
from typing import ClassVar

import pytest

from ftre_agent_core.tool import Tool, ToolParameter


class TestToolBase:
    """Tool 基类测试"""

    def test_manual_construction(self):
        """手动构造 Tool"""
        def my_func(x: str) -> str:
            return f"Got: {x}"
        
        t = Tool(
            name="my_tool",
            description="A simple tool",
            parameters=[
                ToolParameter(name="x", type="string", description="输入值")
            ],
            func=my_func
        )
        
        assert t.name == "my_tool"
        assert t.description == "A simple tool"
        assert len(t.parameters) == 1
        assert t.execute(x="hello") == "Got: hello"

    def test_subclass_tool(self):
        """继承方式定义 Tool"""
        class CalculatorTool(Tool):
            name = "calculator"
            description = "简单计算器"
            parameters: ClassVar[list[ToolParameter]] = [
                ToolParameter(name="expression", type="string", description="数学表达式")
            ]
            
            def _run(self, expression: str) -> str:
                try:
                    result = eval(expression)
                    return str(result)
                except Exception as e:  # noqa: BLE001 - test tool reports its error
                    return f"Error: {e}"
        
        calc = CalculatorTool()
        assert calc.name == "calculator"
        assert calc.execute(expression="1 + 2 * 3") == "7"

    def test_tool_parameter_enum(self):
        """带枚举值的参数"""
        param = ToolParameter(
            name="color",
            type="string",
            description="颜色选择",
            enum=["red", "green", "blue"]
        )
        
        t = Tool(
            name="set_color",
            description="设置颜色",
            parameters=[param],
            func=lambda color: f"Color: {color}"
        )
        
        openai_dict = t.to_openai_dict()
        assert openai_dict["function"]["parameters"]["properties"]["color"]["enum"] == ["red", "green", "blue"]

    def test_tool_without_implementation_raises(self):
        """没有实现的 Tool 应该抛出错误"""
        t = Tool(name="empty", description="empty tool", parameters=[])
        
        with pytest.raises(NotImplementedError):
            t.execute()

    def test_is_async(self):
        """检测异步工具"""
        def sync_func():
            return "sync"
        
        async def async_func():
            return "async"
        
        sync_tool = Tool(name="sync", description="", parameters=[], func=sync_func)
        async_tool = Tool(name="async", description="", parameters=[], func=async_func)
        
        assert sync_tool.is_async() is False
        assert async_tool.is_async() is True
