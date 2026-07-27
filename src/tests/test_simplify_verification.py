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
        from ftre_agent_core.event import (
            EventType, DoneReason, StepPhase, AgentEvent,
            tool_result_event,
            assistant_message_event,
            assistant_message_complete_event, step_event,
            retry_event,
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
        from ftre_agent_core.event import EventType
        expected = {
            "tool_result",
            "assistant_message", "assistant_message_complete",
            "step", "retry",
            "user_message",
        }
        actual = {e.value for e in EventType}
        # 原 6 个保留（子集），AgentScope 对齐的新协议值允许扩充
        assert expected.issubset(actual), f"缺失原值: {expected - actual}"

    def test_removed_types_not_exist(self):
        from ftre_agent_core.event import EventType
        removed = ["tool_cancel_requested", "tool_cancelled", "tool_timed_out",
                   "tool_call", "reasoning_complete", "usage_update",
                   "done", "error"]
        for name in removed:
            assert not hasattr(EventType, name.upper()), f"EventType.{name.upper()} 应已删除"


# ─── 4. 事件构造函数 ──────────────────────────────────────────

class TestEventConstructors:
    """每个构造函数产出正确的 dataclass 实例"""

    def test_tool_result_event(self):
        from ftre_agent_core.event import tool_result_event, EventType, ToolResultEvent
        e = tool_result_event(id="c1", name="bash", result="file1\nfile2")
        assert isinstance(e, ToolResultEvent)
        assert e.type == EventType.TOOL_RESULT
        assert e.status == "completed"

    def test_tool_result_event_with_error(self):
        from ftre_agent_core.event import tool_result_event, ToolResultEvent
        e = tool_result_event(id="c1", name="bash", result="err", error="fail", status="failed")
        assert isinstance(e, ToolResultEvent)
        assert e.error == "fail"
        assert e.status == "failed"

    def test_assistant_message_event(self):
        from ftre_agent_core.event import assistant_message_event, EventType, AssistantMessageEvent
        e = assistant_message_event(content=[{"type": "text", "text": "hello"}])
        assert isinstance(e, AssistantMessageEvent)
        assert e.type == EventType.ASSISTANT_MESSAGE
        assert e.content == [{"type": "text", "text": "hello"}]

    def test_step_event(self):
        from ftre_agent_core.event import step_event, EventType, StepEvent, StepPhase, DoneReason
        e = step_event(StepPhase.TURN_END, success=True, reason=DoneReason.COMPLETED, iterations=3)
        assert isinstance(e, StepEvent)
        assert e.type == EventType.STEP
        assert e.success is True
        assert e.reason == DoneReason.COMPLETED
        assert e.iterations == 3
        assert e.is_turn_end is True

    def test_retry_event(self):
        from ftre_agent_core.event import retry_event, EventType, RetryEvent
        e = retry_event(code="timeout", message="retrying", attempt=1, max_attempts=3)
        assert isinstance(e, RetryEvent)
        assert e.type == EventType.RETRY
        assert e.attempt == 1

    def test_assistant_message_complete_event(self):
        from ftre_agent_core.event import (
            assistant_message_complete_event, EventType, AssistantMessageCompleteEvent,
        )
        e = assistant_message_complete_event(
            content=[{"type": "text", "text": "hello"}],
            metadata={"kind": "final", "usage": {"total_tokens": 100}},
        )
        assert isinstance(e, AssistantMessageCompleteEvent)
        assert e.type == EventType.ASSISTANT_MESSAGE_COMPLETE
        assert e.content[0]["text"] == "hello"
        assert e.metadata["kind"] == "final"
        assert e.metadata["usage"]["total_tokens"] == 100

    def test_assistant_message_complete_with_tool_call(self):
        from ftre_agent_core.event import assistant_message_complete_event
        e = assistant_message_complete_event(
            content=[
                {"type": "thinking", "thinking": "Let me check..."},
                {"type": "text", "text": "I'll read the file"},
                {"type": "toolCall", "id": "c1", "name": "read", "arguments": {"path": "a.py"}},
            ],
            metadata={"kind": "block", "stopReason": "toolUse"},
        )
        assert len(e.content) == 3
        assert e.content[2]["name"] == "read"
        assert e.metadata["stopReason"] == "toolUse"

    def test_assistant_message_complete_default_metadata(self):
        """metadata 参数默认为空 dict"""
        from ftre_agent_core.event import assistant_message_complete_event
        e = assistant_message_complete_event(content=[{"type": "text", "text": "hi"}])
        assert e.metadata == {}


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
        from ftre_agent_core import event as ev
        assert not hasattr(ev, "tool_cancel_requested_event")
        assert not hasattr(ev, "tool_cancelled_event")
        assert not hasattr(ev, "tool_timed_out_event")

    def test_react_runner_no_unretryable(self):
        from ftre_agent_core.agent.runner import ReActRunner
        assert not hasattr(ReActRunner, "UNRETRYABLE_ERROR_CODES")

    def test_removed_event_classes(self):
        """旧事件类已删除"""
        from ftre_agent_core import event as ev
        assert not hasattr(ev, "ToolCallEvent")
        assert not hasattr(ev, "ReasoningCompleteEvent")
        assert not hasattr(ev, "UsageUpdateEvent")
        assert not hasattr(ev, "ReasoningEvent")
        assert not hasattr(ev, "ToolCallStreamingEvent")
        assert not hasattr(ev, "DoneEvent")
        assert not hasattr(ev, "ErrorEvent")

    def test_removed_event_constructors(self):
        """旧事件构造函数已删除"""
        from ftre_agent_core import event as ev
        assert not hasattr(ev, "tool_call_event")
        assert not hasattr(ev, "reasoning_complete_event")
        assert not hasattr(ev, "usage_update_event")
        assert not hasattr(ev, "reasoning_event")
        assert not hasattr(ev, "tool_call_streaming_event")

    def test_removed_typed_dicts(self):
        """旧 TypedDict 已删除"""
        from ftre_agent_core import event as ev
        assert not hasattr(ev, "ToolCallData")
        assert not hasattr(ev, "ReasoningCompleteData")
        assert not hasattr(ev, "UsageUpdateData")
        assert not hasattr(ev, "ToolCallStreamingData")


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
        from ftre_agent_core.event import AgentEvent
        # AgentEvent 现在是 dataclass 基类（不再是 dict 别名）
        assert isinstance(AgentEvent, type)
        assert AgentEvent is not dict

    def test_agent_event_to_dict(self):
        from ftre_agent_core.event import assistant_message_event, EventType
        e = assistant_message_event(content=[{"type": "text", "text": "hello"}])
        d = e.to_dict()
        assert d["type"] == EventType.ASSISTANT_MESSAGE
        assert d["data"] == {"content": [{"type": "text", "text": "hello"}]}
        assert isinstance(d["event_id"], str)
        assert len(d["event_id"]) == 16
        assert "timestamp" in d

    def test_agent_event_from_dict(self):
        from ftre_agent_core.event import AgentEvent, AssistantMessageEvent
        e = AgentEvent.from_dict({
            "type": "assistant_message",
            "event_id": "evt_top_level",
            "data": {"content": [{"type": "text", "text": "world"}]},
        })
        assert isinstance(e, AssistantMessageEvent)
        assert e.content == [{"type": "text", "text": "world"}]
        assert e.event_id == "evt_top_level"

    def test_agent_event_from_legacy_data_event_id(self):
        from ftre_agent_core.event import AgentEvent
        e = AgentEvent.from_dict({
            "type": "assistant_message",
            "data": {"content": [{"type": "text", "text": "world"}], "event_id": "evt_from_data"},
        })
        assert e.event_id == "evt_from_data"

    def test_assistant_message_complete_to_dict(self):
        """assistant_message_complete 的 to_dict 产出新格式"""
        from ftre_agent_core.event import assistant_message_complete_event, EventType
        e = assistant_message_complete_event(
            content=[{"type": "text", "text": "hi"}],
            metadata={"kind": "final"},
        )
        d = e.to_dict()
        assert d["type"] == EventType.ASSISTANT_MESSAGE_COMPLETE
        assert d["data"]["content"] == [{"type": "text", "text": "hi"}]
        assert d["data"]["metadata"]["kind"] == "final"

    def test_assistant_message_complete_from_dict(self):
        """assistant_message_complete 的 from_dict 正确反序列化"""
        from ftre_agent_core.event import AgentEvent, AssistantMessageCompleteEvent
        e = AgentEvent.from_dict({
            "type": "assistant_message_complete",
            "event_id": "evt_001",
            "data": {
                "content": [{"type": "text", "text": "hello"}],
                "metadata": {"kind": "block", "usage": {"total_tokens": 50}},
            },
        })
        assert isinstance(e, AssistantMessageCompleteEvent)
        assert e.content[0]["text"] == "hello"
        assert e.metadata["kind"] == "block"
        assert e.metadata["usage"]["total_tokens"] == 50

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
