"""
并发工具调用测试

测试场景：LLM 一次返回多个 tool_calls，框架并行执行它们。
"""
import time
import json
from ftre_agent_core.agent import ReActAgent, EventType
from ftre_agent_core.tool import tool

API_KEY = "sk-REDACTED"
API_BASE = "https://llm-gateway.REDACTED.example.com/v1"
MODEL = "openai/DeepSeek-V3.2"


@tool()
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    time.sleep(1)  # 模拟网络延迟
    weather = {"北京": "晴天 25°C", "上海": "多云 28°C", "广州": "小雨 30°C"}
    return weather.get(city, f"{city}: 未知")


@tool()
def get_population(city: str) -> str:
    """获取指定城市的人口"""
    time.sleep(1)  # 模拟网络延迟
    pop = {"北京": "2171万", "上海": "2489万", "广州": "1881万"}
    return pop.get(city, f"{city}: 未知")


@tool()
def get_gdp(city: str) -> str:
    """获取指定城市的GDP"""
    time.sleep(1)  # 模拟网络延迟
    gdp = {"北京": "4.16万亿", "上海": "4.72万亿", "广州": "2.88万亿"}
    return gdp.get(city, f"{city}: 未知")


def main():
    agent = ReActAgent(
        model=MODEL,
        api_key=API_KEY,
        api_base=API_BASE,
        system_prompt="你是一个城市信息助手。当用户询问城市信息时，同时调用多个工具获取天气、人口和GDP数据。一次性调用所有需要的工具。",
        tools=[get_weather, get_population, get_gdp],
    )

    print("=== 并发工具调用测试 ===\n")
    print("提问: 告诉我北京的天气、人口和GDP\n")

    start = time.perf_counter()
    tool_calls = []
    tool_results = []

    for event in agent.run("告诉我北京的天气、人口和GDP"):
        etype = event["type"].value
        data = event.get("data", {})

        if etype == "assistant_message":
            print(data.get("content", ""), end="", flush=True)
        elif etype == "assistant_message_complete":
            for b in data.get("content", []):
                if b.get("type") == "toolCall":
                    tool_calls.append(b)
                    print(f"  [TOOL_CALL] {b['name']}({b['arguments']})")
        elif etype == "tool_result":
            tool_results.append(data)
            print(f"  [TOOL_RESULT] {data['name']} → {data['result']}")
        elif etype == "done":
            print(f"\n  [DONE] success={data.get('success')}")

    elapsed = time.perf_counter() - start

    print(f"\n--- 结果统计 ---")
    print(f"总耗时: {elapsed:.2f}s")
    print(f"工具调用数: {len(tool_calls)}")
    print(f"工具结果数: {len(tool_results)}")

    # 如果 3 个工具并行执行（每个 sleep 1s），总耗时应该 ~1s 而不是 ~3s
    if len(tool_calls) >= 2:
        tool_time = elapsed - 2  # 减去大约 2 次 LLM 调用时间
        if tool_time < 2:
            print(f"✅ 工具并行执行（工具部分耗时 < 2s）")
        else:
            print(f"⚠️  工具可能串行执行（工具部分耗时 > 2s）")


if __name__ == "__main__":
    main()
