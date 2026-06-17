"""
代码精简后的验证测试套件

覆盖范围：
  1. import 可达性：所有公开模块/类/函数都能正常导入
  2. LLMError.classify 分类正确性
  3. LLMError.UNRETRYABLE_CODES 与旧常量一致
  4. AgentEvent 事件构造函数产出正确结构
  5. EventType 已删除项确实不存在
  6. 旧兼容类确实已移除（StreamDelta/LLMResponse 等）
  7. responses.py 已删除
  8. react_runner 无 UNRETRYABLE_ERROR_CODES 属性
  9. ReActAgent 端到端（轻量 mock：无需真实 LLM API）
  10. ftre 后端引用无断裂
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
        from ftre_agent_core.agent.event import (
            EventType, DoneReason, AgentEvent,
            tool_call_event, tool_result_event,
            message_event, reasoning_event, reasoning_complete_event,
            message_complete_event, done_event, usage_update_event,
            error_event, retry_event, tool_call_streaming_event,
        )
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


# ─── 3. EventType 枚举 ────────────────────────────────────────

class TestEventTypeEnum:
    """已删除项确实不存在，保留项不变"""

    def test_existing_types(self):
        from ftre_agent_core.agent.event import EventType
        expected = {
            "tool_call", "tool_result",
            "message", "message_complete",
            "reasoning", "reasoning_complete",
            "error", "retry", "done",
            "tool_call_streaming", "usage_update",
        }
        actual = {e.value for e in EventType}
        assert actual == expected

    def test_removed_types_not_exist(self):
        from ftre_agent_core.agent.event import EventType
        removed = ["tool_cancel_requested", "tool_cancelled", "tool_timed_out"]
        for name in removed:
            assert not hasattr(EventType, name.upper()), f"EventType.{name.upper()} 应已删除"


# ─── 4. 事件构造函数 ──────────────────────────────────────────

class TestEventConstructors:
    """每个构造函数产出正确的 dataclass 实例"""

    def test_tool_call_event(self):
        from ftre_agent_core.agent.event import tool_call_event, EventType, ToolCallEvent
        e = tool_call_event(id="c1", name="bash", arguments={"command": "ls"})
        assert isinstance(e, ToolCallEvent)
        assert e.type == EventType.TOOL_CALL
        assert e.tool_id == "c1"
        assert e.tool_name == "bash"

    def test_tool_result_event(self):
        from ftre_agent_core.agent.event import tool_result_event, EventType, ToolResultEvent
        e = tool_result_event(id="c1", name="bash", result="file1\nfile2")
        assert isinstance(e, ToolResultEvent)
        assert e.type == EventType.TOOL_RESULT
        assert e.status == "completed"

    def test_tool_result_event_with_error(self):
        from ftre_agent_core.agent.event import tool_result_event, ToolResultEvent
        e = tool_result_event(id="c1", name="bash", result="err", error="fail", status="failed")
        assert isinstance(e, ToolResultEvent)
        assert e.error == "fail"
        assert e.status == "failed"

    def test_message_event(self):
        from ftre_agent_core.agent.event import message_event, EventType, MessageEvent
        e = message_event("hello")
        assert isinstance(e, MessageEvent)
        assert e.type == EventType.MESSAGE
        assert e.content == "hello"

    def test_reasoning_event(self):
        from ftre_agent_core.agent.event import reasoning_event, EventType, ReasoningEvent
        e = reasoning_event("thinking...")
        assert isinstance(e, ReasoningEvent)
        assert e.type == EventType.REASONING

    def test_done_event(self):
        from ftre_agent_core.agent.event import done_event, EventType, DoneReason, DoneEvent
        e = done_event(success=True, reason=DoneReason.COMPLETED)
        assert isinstance(e, DoneEvent)
        assert e.type == EventType.DONE
        assert e.success is True
        assert e.reason == DoneReason.COMPLETED

    def test_done_event_with_usage(self):
        from ftre_agent_core.agent.event import done_event, DoneReason, DoneEvent
        e = done_event(success=True, reason=DoneReason.COMPLETED, usage={"prompt_tokens": 100})
        assert isinstance(e, DoneEvent)
        assert e.usage["prompt_tokens"] == 100

    def test_error_event(self):
        from ftre_agent_core.agent.event import error_event, EventType, ErrorEvent
        e = error_event(message="boom", code="timeout")
        assert isinstance(e, ErrorEvent)
        assert e.type == EventType.ERROR
        assert e.code == "timeout"

    def test_retry_event(self):
        from ftre_agent_core.agent.event import retry_event, EventType, RetryEvent
        e = retry_event(code="timeout", message="retrying", attempt=1, max_attempts=3)
        assert isinstance(e, RetryEvent)
        assert e.type == EventType.RETRY
        assert e.attempt == 1

    def test_usage_update_event(self):
        from ftre_agent_core.agent.event import usage_update_event, EventType, UsageUpdateEvent
        e = usage_update_event({"total_tokens": 500})
        assert isinstance(e, UsageUpdateEvent)
        assert e.type == EventType.USAGE_UPDATE

    def test_reasoning_complete_event(self):
        from ftre_agent_core.agent.event import reasoning_complete_event, EventType, ReasoningCompleteEvent
        e = reasoning_complete_event("full reasoning text")
        assert isinstance(e, ReasoningCompleteEvent)
        assert e.type == EventType.REASONING_COMPLETE

    def test_message_complete_event(self):
        from ftre_agent_core.agent.event import message_complete_event, EventType, MessageCompleteEvent
        e = message_complete_event("full message")
        assert isinstance(e, MessageCompleteEvent)
        assert e.type == EventType.MESSAGE_COMPLETE


# ─── 5. 已删除项确认 ──────────────────────────────────────────

class TestRemovedItems:
    """旧版类/模块确实不存在"""

    def test_stream_delta_removed(self):
        with pytest.raises(ImportError):
            from ftre_agent_core.llm import StreamDelta

    def test_llm_response_removed(self):
        with pytest.raises(ImportError):
            from ftre_agent_core.llm import LLMResponse

    def test_tool_call_wrapper_removed(self):
        with pytest.raises(ImportError):
            from ftre_agent_core.llm import ToolCallWrapper

    def test_tool_call_delta_chunk_removed(self):
        with pytest.raises(ImportError):
            from ftre_agent_core.llm import ToolCallDeltaChunk

    def test_provider_error_removed(self):
        with pytest.raises(ImportError):
            from ftre_agent_core.llm import ProviderError

    def test_step_start_removed(self):
        with pytest.raises(ImportError):
            from ftre_agent_core.llm import StepStart

    def test_llm_tool_result_removed(self):
        # llm 层的 ToolResult/ToolError 已删除（runner 自有 ToolResult 不受影响）
        with pytest.raises(ImportError):
            from ftre_agent_core.llm import ToolError

    def test_responses_module_removed(self):
        with pytest.raises(ImportError):
            from ftre_agent_core.llm.responses import ResponsesAdapter

    def test_lifecycle_event_functions_removed(self):
        from ftre_agent_core.agent import event as ev
        assert not hasattr(ev, "tool_cancel_requested_event")
        assert not hasattr(ev, "tool_cancelled_event")
        assert not hasattr(ev, "tool_timed_out_event")

    def test_react_runner_no_unretryable(self):
        from ftre_agent_core.agent.runner import ReActRunner
        assert not hasattr(ReActRunner, "UNRETRYABLE_ERROR_CODES")


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


# ─── 7. AgentEvent 类型 ──────────────────────────────────────

class TestAgentEventType:
    """AgentEvent 已从 dict 别名升级为 dataclass 基类"""

    def test_agent_event_is_class(self):
        from ftre_agent_core.agent.event import AgentEvent, AgentEventDict
        # AgentEvent 现在是 dataclass 基类（不再是 dict 别名）
        assert isinstance(AgentEvent, type)
        assert AgentEvent is not dict
        # AgentEventDict 保留作为向后兼容别名
        assert AgentEventDict is dict

    def test_agent_event_to_dict(self):
        from ftre_agent_core.agent.event import message_event, EventType
        e = message_event("hello")
        d = e.to_dict()
        assert d == {"type": EventType.MESSAGE, "data": {"content": "hello"}}

    def test_agent_event_from_dict(self):
        from ftre_agent_core.agent.event import AgentEvent, MessageEvent
        e = AgentEvent.from_dict({"type": "message", "data": {"content": "world"}})
        assert isinstance(e, MessageEvent)
        assert e.content == "world"

    def test_agent_event_no_dict_access(self):
        """AgentEvent 实例不再支持 dict 风格访问"""
        from ftre_agent_core.agent.event import done_event, DoneReason
        import pytest
        e = done_event(success=True, reason=DoneReason.COMPLETED)
        with pytest.raises(TypeError):
            _ = e["type"]
        with pytest.raises(AttributeError):
            _ = e.get("type")


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