"""
Responses API 测试 - 使用 xiamai 厂商
"""
import pytest
from ftre_agent_core.agent import ReActAgent, EventType
from ftre_agent_core.tool import tool

# xiamai 配置
API_KEY = "sk-3d6e0dc78b9b00f6c30709dd3e8d6bef55922219e64419ec7b7236051afa1638"
API_BASE = "http://ai.xiamai.top/v1"
MODEL = "openai/gpt-5.4"


def test_simple_chat():
    """简单对话测试"""
    print("=" * 50)
    print("测试 1: 简单对话")
    print("=" * 50)
    
    agent = ReActAgent(
        model=MODEL,
        api_key=API_KEY,
        api_base=API_BASE,
        api_type="responses",
        system_prompt="你是一个简洁的助手，回答尽量简短。",
        tools=[],
    )
    
    print("发送: 你好")
    for event in agent.run("你好"):
        if event["type"] == EventType.MESSAGE:
            print(event["data"]["content"], end="", flush=True)
        elif event["type"] == EventType.ERROR:
            print(f"\n错误: {event['data']}")
        elif event["type"] == EventType.DONE:
            print(f"\n完成: {event['data']}")
    print()


def test_tool_call():
    """工具调用测试"""
    print("=" * 50)
    print("测试 2: 工具调用")
    print("=" * 50)
    
    @tool()
    def get_weather(city: str) -> str:
        """获取指定城市的天气"""
        weather_data = {
            "北京": "晴天，25°C",
            "上海": "多云，28°C",
            "广州": "小雨，30°C",
        }
        return weather_data.get(city, f"{city}：数据未知")
    
    agent = ReActAgent(
        model=MODEL,
        api_key=API_KEY,
        api_base=API_BASE,
        api_type="responses",
        system_prompt="你是一个天气助手。当用户询问天气时，使用 get_weather 工具。",
        tools=[get_weather],
    )
    
    print("发送: 北京天气怎么样？")
    for event in agent.run("北京天气怎么样？"):
        event_type = event["type"]
        if event_type == EventType.MESSAGE:
            print(event["data"]["content"], end="", flush=True)
        elif event_type == EventType.TOOL_CALL:
            print(f"\n[工具调用] {event['data']['name']}({event['data']['arguments']})")
        elif event_type == EventType.TOOL_RESULT:
            print(f"[工具结果] {event['data']['result']}")
        elif event_type == EventType.ERROR:
            print(f"\n错误: {event['data']}")
        elif event_type == EventType.DONE:
            print(f"\n完成: {event['data']}")
    print()


def test_multi_turn():
    """多轮对话测试"""
    print("=" * 50)
    print("测试 3: 多轮对话")
    print("=" * 50)
    
    agent = ReActAgent(
        model=MODEL,
        api_key=API_KEY,
        api_base=API_BASE,
        api_type="responses",
        system_prompt="你是一个助手，记住用户说的话。",
        tools=[],
    )
    
    print("发送: 我叫小明，今年25岁")
    for event in agent.run("我叫小明，今年25岁"):
        if event["type"] == EventType.MESSAGE:
            print(event["data"]["content"], end="", flush=True)
    print()
    
    print("\n发送: 我叫什么名字？多少岁？")
    for event in agent.run("我叫什么名字？多少岁？"):
        if event["type"] == EventType.MESSAGE:
            print(event["data"]["content"], end="", flush=True)
        elif event["type"] == EventType.ERROR:
            print(f"\n错误: {event['data']}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Responses API 测试 - xiamai 厂商")
    print("=" * 50 + "\n")
    
    try:
        test_simple_chat()
    except Exception as e:
        print(f"测试 1 失败: {e}")
    
    print()
    
    try:
        test_tool_call()
    except Exception as e:
        print(f"测试 2 失败: {e}")
    
    print()
    
    try:
        test_multi_turn()
    except Exception as e:
        print(f"测试 3 失败: {e}")
