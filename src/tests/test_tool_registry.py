"""
测试 ToolRegistry
"""
from typing import ClassVar

from ftre_agent_core.tool import Tool, ToolRegistry, tool


class TestToolRegistry:
    """ToolRegistry 测试"""

    def test_register_and_get(self):
        """注册和获取工具"""
        registry = ToolRegistry()
        
        @tool()
        def my_tool(x: str) -> str:
            """测试工具"""
            return x
        
        registry.register(my_tool)
        
        assert registry.get("my_tool") is my_tool
        assert "my_tool" in registry.names

    def test_register_multiple_tools(self):
        """注册多个工具"""
        registry = ToolRegistry()
        
        @tool()
        def tool_a() -> str:
            """工具A"""
            return "a"
        
        @tool()
        def tool_b() -> str:
            """工具B"""
            return "b"
        
        registry.register(tool_a)
        registry.register(tool_b)
        
        assert len(registry.names) == 2
        assert set(registry.names) == {"tool_a", "tool_b"}

    def test_get_nonexistent_tool(self):
        """获取不存在的工具返回 None"""
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_to_openai_tools(self):
        """转换为 OpenAI 工具列表"""
        registry = ToolRegistry()
        
        @tool()
        def greet(name: str) -> str:
            """打招呼"""
            return f"Hi {name}"
        
        registry.register(greet)
        
        openai_list = registry.to_openai_tools()
        
        assert len(openai_list) == 1
        assert openai_list[0]["type"] == "function"
        assert openai_list[0]["function"]["name"] == "greet"

    def test_register_tool_class(self):
        """注册 Tool 子类"""
        registry = ToolRegistry()
        
        class CustomTool(Tool):
            name = "custom"
            description = "自定义工具"
            parameters: ClassVar[list] = []
            
            def _run(self) -> str:
                return "custom result"
        
        registry.register(CustomTool())
        
        assert registry.get("custom") is not None
        assert registry.get("custom").execute() == "custom result"

    def test_unregister_tool(self):
        """注销工具"""
        registry = ToolRegistry()
        
        @tool()
        def temp_tool() -> str:
            """临时工具"""
            return "temp"
        
        registry.register(temp_tool)
        assert len(registry) == 1
        
        registry.unregister("temp_tool")
        assert len(registry) == 0

    def test_has_tool(self):
        """检查工具是否存在"""
        registry = ToolRegistry()
        
        @tool()
        def my_tool() -> str:
            """测试"""
            return "test"
        
        assert registry.has("my_tool") is False
        registry.register(my_tool)
        assert registry.has("my_tool") is True

    def test_execute_tool(self):
        """通过注册表执行工具"""
        registry = ToolRegistry()
        
        @tool()
        def add(a: int, b: int) -> int:
            """加法"""
            return a + b
        
        registry.register(add)
        
        result = registry.execute("add", a=1, b=2)
        assert result == 3

    def test_len_and_iter(self):
        """测试 __len__ 和 __iter__"""
        registry = ToolRegistry()
        
        @tool()
        def tool_a() -> str:
            """工具A"""
            return "a"
        
        @tool()
        def tool_b() -> str:
            """工具B"""
            return "b"
        
        registry.register(tool_a)
        registry.register(tool_b)
        
        assert len(registry) == 2
        assert {t.name for t in registry} == {"tool_a", "tool_b"}
