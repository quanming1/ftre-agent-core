"""
测试 @tool 装饰器
"""
import pytest
from ftre_agent_core.tool import tool, Tool, ToolParameter


class TestToolDecorator:
    """@tool 装饰器测试"""

    def test_basic_decorator(self):
        """基础用法：自动推断参数"""
        @tool()
        def greet(name: str) -> str:
            """向用户打招呼"""
            return f"Hello, {name}!"
        
        assert isinstance(greet, Tool)
        assert greet.name == "greet"
        assert greet.description == "向用户打招呼"
        assert len(greet.parameters) == 1
        assert greet.parameters[0].name == "name"
        assert greet.parameters[0].type == "string"

    def test_custom_name_and_description(self):
        """自定义名称和描述"""
        @tool(name="say_hello", description="Say hello to someone")
        def greet(name: str) -> str:
            return f"Hello, {name}!"
        
        assert greet.name == "say_hello"
        assert greet.description == "Say hello to someone"

    def test_multiple_parameters(self):
        """多参数推断"""
        @tool()
        def search(query: str, limit: int, exact: bool) -> str:
            """搜索内容"""
            return f"Searching: {query}"
        
        assert len(search.parameters) == 3
        
        param_map = {p.name: p for p in search.parameters}
        assert param_map["query"].type == "string"
        assert param_map["limit"].type == "number"
        assert param_map["exact"].type == "boolean"

    def test_optional_parameter(self):
        """可选参数（有默认值）"""
        @tool()
        def read_file(path: str, encoding: str = "utf-8") -> str:
            """读取文件"""
            return f"Reading {path}"
        
        param_map = {p.name: p for p in read_file.parameters}
        assert param_map["path"].required is True
        assert param_map["encoding"].required is False

    def test_list_and_dict_types(self):
        """列表和字典类型"""
        @tool()
        def process(items: list, config: dict) -> str:
            """处理数据"""
            return "done"
        
        param_map = {p.name: p for p in process.parameters}
        assert param_map["items"].type == "array"
        assert param_map["config"].type == "object"

    def test_tool_execution(self):
        """工具执行"""
        @tool()
        def add(a: int, b: int) -> int:
            """加法"""
            return a + b
        
        result = add.execute(a=1, b=2)
        assert result == 3

    def test_to_openai_dict(self):
        """转换为 OpenAI 格式"""
        @tool()
        def get_weather(city: str) -> str:
            """获取天气"""
            return f"{city}: sunny"
        
        openai_dict = get_weather.to_openai_dict()
        
        assert openai_dict["type"] == "function"
        assert openai_dict["function"]["name"] == "get_weather"
        assert openai_dict["function"]["description"] == "获取天气"
        assert "city" in openai_dict["function"]["parameters"]["properties"]
        assert "city" in openai_dict["function"]["parameters"]["required"]
