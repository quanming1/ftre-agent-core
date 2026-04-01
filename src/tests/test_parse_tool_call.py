"""
测试 parse_tool_call 的 JSON 截断容错处理
"""
import pytest
from unittest.mock import Mock
from ftre_agent_core.tool import ToolRegistry
from ftre_agent_core.agent.runner.handler.tool_handler import ToolHandler


class TestParseToolCallTruncation:
    """测试 JSON 截断时的容错处理"""

    def setup_method(self):
        """每个测试方法前初始化"""
        self.registry = ToolRegistry()
        self.handler = ToolHandler(self.registry)

    def test_parse_valid_json(self):
        """正常 JSON 解析"""
        tool_call = Mock()
        tool_call.id = "call_123"
        tool_call.function.name = "write"
        tool_call.function.arguments = '{"filePath": "test.py", "content": "print(1)"}'

        call_id, name, args = self.handler.parse_tool_call(tool_call)

        assert call_id == "call_123"
        assert name == "write"
        assert args == {"filePath": "test.py", "content": "print(1)"}

    def test_parse_truncated_json_returns_none_args(self):
        """截断的 JSON 应返回 arguments=None"""
        tool_call = Mock()
        tool_call.id = "call_456"
        tool_call.function.name = "write"
        # 模拟 streaming 截断：JSON 不完整
        tool_call.function.arguments = '{"filePath": "src/createStores.ts'

        call_id, name, args = self.handler.parse_tool_call(tool_call)

        assert call_id == "call_456"
        assert name == "write"
        assert args is None  # 解析失败时返回 None

    def test_parse_malformed_json_returns_none_args(self):
        """畸形 JSON 应返回 arguments=None"""
        tool_call = Mock()
        tool_call.id = "call_789"
        tool_call.function.name = "read"
        tool_call.function.arguments = '{"filePath": test.py}'  # 缺少引号

        call_id, name, args = self.handler.parse_tool_call(tool_call)

        assert call_id == "call_789"
        assert name == "read"
        assert args is None

    def test_parse_empty_json_returns_empty_dict(self):
        """空 JSON 对象应正常解析"""
        tool_call = Mock()
        tool_call.id = "call_empty"
        tool_call.function.name = "think"
        tool_call.function.arguments = '{}'

        call_id, name, args = self.handler.parse_tool_call(tool_call)

        assert call_id == "call_empty"
        assert name == "think"
        assert args == {}

    def test_parse_very_short_truncation(self):
        """极短截断（如只有开头几个字符）"""
        tool_call = Mock()
        tool_call.id = "call_short"
        tool_call.function.name = "write"
        tool_call.function.arguments = '{"f'

        call_id, name, args = self.handler.parse_tool_call(tool_call)

        assert call_id == "call_short"
        assert name == "write"
        assert args is None
