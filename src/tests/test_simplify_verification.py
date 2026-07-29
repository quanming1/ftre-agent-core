"""
代码精简后的验证测试套件

覆盖范围：
   1. import 可达性：所有公开模块/类/函数都能正常导入
   2. LLMError.classify 分类正确性
   3. LLMError.UNRETRYABLE_CODES 与旧常量一致
   4. Tool 基类不受影响
   5. ftre 后端引用无断裂
"""

import sys
import pytest


# ─── 1. import 可达性 ──────────────────────────────────────────

class TestImportReachability:
    """所有公开导出都能正常导入，不报 ImportError"""

    def test_llm_handler(self):
        from ftre_agent_core.llm import LLMHandler
        assert LLMHandler is not None

    def test_llm_error(self):
        from ftre_agent_core.llm import LLMError
        assert LLMError is not None

    def test_llm_event_types(self):
        from ftre_agent_core.llm import (
            LLMEvent, TextDelta, ReasoningDelta,
            ToolInputDelta, ToolCall, StepFinish,
        )
        # 全部非空
        for cls in [TextDelta, ReasoningDelta, ToolInputDelta,
                     ToolCall, StepFinish]:
            assert cls is not None
        assert LLMEvent is not None

    def test_agent_event_module(self):
        from ftre_agent_core.event import EventType
        assert EventType is not None

    def test_react_runner(self):
        from ftre_agent_core.agent.runner import ReActRunner, RunState, RunStatus
        assert ReActRunner is not None

    def test_tool_handler(self):
        from ftre_agent_core.agent.runner import ToolHandler, ToolResult
        assert ToolHandler is not None

    def test_tool_base(self):
        from ftre_agent_core.tool import Tool, ToolParameter, Injected, tool
        assert Tool is not None

    def test_tool_registry(self):
        from ftre_agent_core.tool import ToolRegistry
        assert ToolRegistry is not None

    def test_cancellation(self):
        from ftre_agent_core.tool import CancellationToken, ToolCancelledError
        assert CancellationToken is not None


# ─── 2. LLMError.classify 分类 ──────────────────────────────────

class TestLLMErrorClassify:
    """精简后的 classify 行为与旧版一致"""

    def test_unknown_error(self):
        from ftre_agent_core.llm import LLMError
        err = LLMError.classify(Exception("test"))
        assert err.code == "unknown"
        assert err.message == "test"

    def test_rate_limit(self):
        import openai
        from ftre_agent_core.llm import LLMError
        # 构造 openai 异常需要 response 对象，用 mock 绕过
        from unittest.mock import MagicMock
        resp = MagicMock()
        resp.request = MagicMock()
        err = LLMError.classify(openai.RateLimitError("too many", response=resp, body=None))
        assert err.code == "rate_limit"

    def test_timeout(self):
        import openai
        from ftre_agent_core.llm import LLMError
        err = LLMError.classify(openai.APITimeoutError(request=None))
        assert err.code == "timeout"

    def test_bad_request(self):
        import openai
        from ftre_agent_core.llm import LLMError
        from unittest.mock import MagicMock
        resp = MagicMock()
        resp.request = MagicMock()
        err = LLMError.classify(openai.BadRequestError("bad", response=resp, body=None))
        assert err.code == "bad_request"

    def test_auth_error(self):
        import openai
        from ftre_agent_core.llm import LLMError
        from unittest.mock import MagicMock
        resp = MagicMock()
        resp.request = MagicMock()
        err = LLMError.classify(openai.AuthenticationError("no key", response=resp, body=None))
        assert err.code == "auth_error"

    def test_unretryable_codes(self):
        from ftre_agent_core.llm import LLMError
        # 与旧版 UNRETRYABLE_ERROR_CODES = {"auth_error", "bad_request", "content_filter"} 一致
        assert LLMError.UNRETRYABLE_CODES == {"auth_error", "bad_request", "content_filter"}

    def test_api_error(self):
        import openai
        from ftre_agent_core.llm import LLMError
        from unittest.mock import MagicMock
        resp = MagicMock()
        resp.request = MagicMock()
        # 通用 APIError 应归入 api_error
        err = LLMError.classify(openai.APIError("generic", request=resp.request, body=None))
        assert err.code == "api_error"


# ─── 6. Tool 基类 ────────────────────────────────────────────

class TestToolBaseStillWorks:
    """Tool/ToolParameter/Injected/tool 装饰器不受影响"""

    def test_tool_creation(self):
        from ftre_agent_core.tool import Tool, ToolParameter
        t = Tool(
            name="test",
            description="测试",
            parameters=[ToolParameter(name="x", type="string", description="输入")],
            func=lambda x: x,
        )
        assert t.name == "test"
        assert t.to_openai_dict()["function"]["name"] == "test"

    def test_tool_decorator(self):
        from ftre_agent_core.tool import tool
        @tool()
        def greet(name: str) -> str:
            """打招呼"""
            return f"Hi {name}"
        assert greet.name == "greet"
        assert greet.execute(name="world") == "Hi world"

    def test_tool_is_async(self):
        from ftre_agent_core.tool import Tool
        async def async_fn(): return "async"
        t = Tool(name="async_tool", description="", parameters=[], func=async_fn)
        assert t.is_async() is True


# ─── 8. ftre 后端引用无断裂 ──────────────────────────────────

class TestFtreBackendImports:
    """ftre 后端引用的 ftre_agent_core 导出仍然可用"""

    def test_react_agent(self):
        from ftre_agent_core.agent import ReActAgent
        assert ReActAgent is not None

    def test_tool_exports(self):
        from ftre_agent_core.tool import Tool, ToolParameter, Injected, tool
        for item in [Tool, ToolParameter, Injected, tool]:
            assert item is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
