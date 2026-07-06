"""
Agent 集成测试 - 使用真实 LLM API
"""
import pytest
from ftre_agent_core.agent import ReActAgent, EventType
from ftre_agent_core.tool import tool


API_KEY = "bce-v3/ALTAKSP-OaWOvCQU8el1GqZscIGgB/0cb52ddf8cabf8e8da91c707a16b4c9315ec270e"
API_BASE = "https://qianfan.baidubce.com/v2/coding"
MODEL = "openai/minimax-m2.5"


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
        assert EventType.ASSISTANT_MESSAGE_COMPLETE in event_types
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
        assert EventType.TOOL_RESULT in event_types
        assert EventType.DONE in event_types

        # tool_call 现在嵌入 assistant_message_complete 的 content[] 中
        amc = next(e for e in events if e["type"] == EventType.ASSISTANT_MESSAGE_COMPLETE)
        tool_call_blocks = [b for b in amc["data"]["content"] if b.get("type") == "toolCall"]
        assert len(tool_call_blocks) > 0
        assert tool_call_blocks[0]["name"] == "get_weather"

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
        assert EventType.TOOL_RESULT in event_types

        tool_result_event = next(
            (e for e in events if e["type"] == EventType.TOOL_RESULT),
            None
        )
        assert tool_result_event is not None
        assert "56088" in tool_result_event["data"]["result"]


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

        amc_events = [
            e for e in events
            if e["type"] == EventType.ASSISTANT_MESSAGE_COMPLETE
        ]
        # 从 content[] 中提取所有 text 块
        full_response = ""
        for e in amc_events:
            for block in e["data"]["content"]:
                if block.get("type") == "text":
                    full_response += block["text"]
        assert "小明" in full_response or "明" in full_response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
