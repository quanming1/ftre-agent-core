"""
Agent 集成测试 - 使用真实 LLM API
"""
import pytest
from ftre_agent_core.agent import ReActAgent, EventType
from ftre_agent_core.tool import tool


API_KEY = "sk-1cdcb7b0e7fb40d49bd6b66b8666022e"
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "openai/qwen-plus"


@pytest.fixture
def simple_tools():
    """简单工具集"""
    @tool()
    def get_weather(city: str) -> str:
        """获取指定城市的天气"""
        weather_data = {
            "北京": "晴天，25°C",
            "上海": "多云，28°C",
            "广州": "小雨，30°C",
        }
        return weather_data.get(city, f"{city}：数据未知")

    @tool()
    def calculate(expression: str) -> str:
        """计算数学表达式"""
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"计算错误: {e}"

    return [get_weather, calculate]


class TestAgentBasic:
    """Agent 基础测试"""

    def test_simple_chat(self):
        """简单对话，无工具调用"""
        agent = ReActAgent(
            model=MODEL,
            api_key=API_KEY,
            api_base=API_BASE,
            system_prompt="你是一个简洁的助手，回答尽量简短。",
            tools=[],
        )

        events = list(agent.run("你好"))

        event_types = [e["type"] for e in events]
        assert EventType.MESSAGE in event_types or EventType.MESSAGE_COMPLETE in event_types
        assert EventType.DONE in event_types

    def test_tool_call(self, simple_tools):
        """测试工具调用"""
        agent = ReActAgent(
            model=MODEL,
            api_key=API_KEY,
            api_base=API_BASE,
            system_prompt="你是一个天气助手。当用户询问天气时，使用 get_weather 工具。",
            tools=simple_tools,
        )

        events = list(agent.run("北京天气怎么样？"))

        event_types = [e["type"] for e in events]
        assert EventType.TOOL_CALL in event_types
        assert EventType.TOOL_RESULT in event_types
        assert EventType.DONE in event_types

        tool_call_event = next(e for e in events if e["type"] == EventType.TOOL_CALL)
        assert tool_call_event["data"]["name"] == "get_weather"

    def test_tool_result_in_response(self, simple_tools):
        """验证工具结果被正确使用"""
        agent = ReActAgent(
            model=MODEL,
            api_key=API_KEY,
            api_base=API_BASE,
            system_prompt="你是一个天气助手。查询天气后直接告诉用户结果。",
            tools=simple_tools,
        )

        events = list(agent.run("上海今天天气如何？"))

        tool_result_event = next(
            (e for e in events if e["type"] == EventType.TOOL_RESULT),
            None
        )
        assert tool_result_event is not None
        assert "28" in tool_result_event["data"]["result"] or "多云" in tool_result_event["data"]["result"]

    def test_calculate_tool(self, simple_tools):
        """测试计算工具"""
        agent = ReActAgent(
            model=MODEL,
            api_key=API_KEY,
            api_base=API_BASE,
            system_prompt="你是一个计算助手。使用 calculate 工具进行计算。",
            tools=simple_tools,
        )

        events = list(agent.run("计算 123 * 456"))

        event_types = [e["type"] for e in events]
        assert EventType.TOOL_CALL in event_types

        tool_result_event = next(
            (e for e in events if e["type"] == EventType.TOOL_RESULT),
            None
        )
        assert tool_result_event is not None
        assert "56088" in tool_result_event["data"]["result"]


class TestAgentInterrupt:
    """Agent 中断测试"""

    def test_interrupt_before_tool(self, simple_tools):
        """测试工具执行前中断"""
        agent = ReActAgent(
            model=MODEL,
            api_key=API_KEY,
            api_base=API_BASE,
            system_prompt="你是一个天气助手。",
            tools=simple_tools,
            interrupt_before=["get_weather"],
        )

        events = list(agent.run("北京天气怎么样？"))

        event_types = [e["type"] for e in events]
        assert EventType.INTERRUPT in event_types

        interrupt_event = next(e for e in events if e["type"] == EventType.INTERRUPT)
        assert interrupt_event["data"]["tool_name"] == "get_weather"

    def test_resume_after_interrupt(self, simple_tools):
        """测试中断后恢复执行"""
        agent = ReActAgent(
            model=MODEL,
            api_key=API_KEY,
            api_base=API_BASE,
            system_prompt="你是一个天气助手。",
            tools=simple_tools,
            interrupt_before=["get_weather"],
        )

        events = list(agent.run("北京天气怎么样？"))
        assert EventType.INTERRUPT in [e["type"] for e in events]

        resume_events = list(agent.resume(approved=True))

        event_types = [e["type"] for e in resume_events]
        assert EventType.TOOL_RESULT in event_types
        assert EventType.DONE in event_types

    def test_reject_after_interrupt(self, simple_tools):
        """测试中断后拒绝执行"""
        agent = ReActAgent(
            model=MODEL,
            api_key=API_KEY,
            api_base=API_BASE,
            system_prompt="你是一个天气助手。如果工具被拒绝，告诉用户无法执行。",
            tools=simple_tools,
            interrupt_before=["get_weather"],
        )

        events = list(agent.run("北京天气怎么样？"))
        assert EventType.INTERRUPT in [e["type"] for e in events]

        resume_events = list(agent.resume(approved=False))

        event_types = [e["type"] for e in resume_events]
        assert EventType.DONE in event_types


class TestAgentMultiTurn:
    """多轮对话测试"""

    def test_conversation_context(self, simple_tools):
        """测试对话上下文保持"""
        agent = ReActAgent(
            model=MODEL,
            api_key=API_KEY,
            api_base=API_BASE,
            system_prompt="你是一个助手，记住用户说的话。",
            tools=[],
        )

        list(agent.run("我叫小明"))

        events = list(agent.run("我叫什么名字？"))

        message_events = [
            e for e in events
            if e["type"] == EventType.MESSAGE and e["data"]["content"]
        ]
        full_response = "".join(e["data"]["content"] for e in message_events)
        assert "小明" in full_response or "明" in full_response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
