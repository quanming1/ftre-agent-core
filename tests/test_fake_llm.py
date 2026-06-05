"""
Runner _step 测试
运行：python tests/test_fake_llm.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ftre_agent_core.agent import ReActAgent, EventType
from ftre_agent_core.llm.completion import StreamDelta, LLMResponse, ToolCallWrapper, ToolCallDeltaChunk
from ftre_agent_core.tool import tool


def make_agent(tools=None):
    return ReActAgent(
        model="fake",
        api_key="fake",
        system_prompt="你是测试助手",
        tools=tools or [],
    )


# ============================================================
# 场景 1：纯文本，正常结束
# ============================================================

def test_text():
    print("\n【场景1】纯文本输出，正常结束")

    def fake_stream(messages, tools=None):
        yield StreamDelta(content="你好")
        yield StreamDelta(content="，我是假模型")
        yield StreamDelta(content="。")

    agent = make_agent()
    agent.runner.llm.stream = fake_stream
    events = list(agent.run("hi"))
    types = [e["type"].value for e in events]
    full_text = "".join(e["data"]["content"] for e in events if e["type"] == EventType.MESSAGE)

    print(f"  事件序列 : {types}")
    print(f"  完整文本 : {full_text!r}")
    assert full_text == "你好，我是假模型。"
    assert "message_complete" in types
    assert "done" in types
    print("  PASS")


# ============================================================
# 场景 2：带 reasoning + 多个工具调用 + 第二轮 reasoning + 最终回答
# 完全按照真实 deepseek-reasoner 输出抄写
# ============================================================

def test_reasoning_and_multi_tool():
    print("\n【场景2】reasoning + 多工具调用 + 第二轮 reasoning + 最终回答")

    @tool(description="查询城市当前天气")
    def get_weather(city: str) -> str:
        return f"{city}：晴，26°C，东南风3级"

    @tool(description="查询城市的空气质量指数")
    def get_aqi(city: str) -> str:
        return f"{city} AQI：42（优）"

    call_count = 0

    def fake_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # ---- 第一轮：reasoning → 简短 message → 两个并行工具调用 ----
            # reasoning
            for token in ["用户", "想", "了解", "北京的", "天气", "和", "空气质量", "。",
                          "我需要", "调用", "两个", "工具", "：", "get_weather", " 和 ", "get_aqi", "，",
                          "可以", "同时", "调用", "。"]:
                yield StreamDelta(reasoning=token)

            # message
            for token in ["好的", "，", "我来", "查询", "一下", "北京", "今天的", "天气", "和", "空气质量", "情况", "！"]:
                yield StreamDelta(content=token)

            # tool_call 流式 delta（两个并行）
            yield StreamDelta(tool_calls=[ToolCallDeltaChunk(index=0, id="call_00_weather", name="get_weather")])
            for arg_piece in ["{", '"', "city", '"', ": ", '"', "北京", '"', "}"]:
                yield StreamDelta(tool_calls=[ToolCallDeltaChunk(index=0, arguments_delta=arg_piece)])

            yield StreamDelta(tool_calls=[ToolCallDeltaChunk(index=1, id="call_01_aqi", name="get_aqi")])
            for arg_piece in ["{", '"', "city", '"', ": ", '"', "北京", '"', "}"]:
                yield StreamDelta(tool_calls=[ToolCallDeltaChunk(index=1, arguments_delta=arg_piece)])

            yield StreamDelta(usage={"prompt_tokens": 352, "completion_tokens": 124, "total_tokens": 476})

            # LLMResponse 触发工具执行
            yield LLMResponse(
                content="好的，我来查询一下北京今天的天气和空气质量情况！",
                reasoning="用户想了解北京的天气和空气质量。我需要调用两个工具：get_weather 和 get_aqi，可以同时调用。",
                tool_calls=[
                    ToolCallWrapper({"id": "call_00_weather", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "北京"}'}}),
                    ToolCallWrapper({"id": "call_01_aqi",     "type": "function", "function": {"name": "get_aqi",     "arguments": '{"city": "北京"}'}}),
                ],
                usage={"prompt_tokens": 352, "completion_tokens": 124, "total_tokens": 476},
            )

        elif call_count == 2:
            # ---- 第二轮：拿到工具结果后，reasoning → 最终回答 ----
            for token in ["查询", "结果", "已经", "返回", "了。",
                          "北京", "今天", "晴，26°C，东南风3级，",
                          "AQI 42（优）。", "我来", "给", "用户", "一个", "完整的", "回答", "。"]:
                yield StreamDelta(reasoning=token)

            final = (
                "查询结果如下，北京的天气和空气质量都挺不错的！\n\n"
                "## 北京今日天气\n"
                "| 项目 | 详情 |\n|------|------|\n"
                "| 天气状况 | 晴 |\n| 气温 | 26°C |\n| 风向风力 | 东南风3级 |\n\n"
                "## 空气质量\n"
                "| 项目 | 详情 |\n|------|------|\n"
                "| AQI指数 | 42（优）|\n\n"
                "今天北京天气晴朗舒适，空气质量也非常棒，适合外出活动！"
            )
            for token in final.split("，"):
                yield StreamDelta(content=token + ("，" if token != final.split("，")[-1] else ""))

            yield StreamDelta(usage={"prompt_tokens": 516, "completion_tokens": 195, "total_tokens": 711})

    agent = make_agent(tools=[get_weather, get_aqi])
    agent.runner.llm.stream = fake_stream
    events = list(agent.run("北京今天天气和空气质量怎么样？"))
    types = [e["type"].value for e in events]

    tool_results = [e for e in events if e["type"] == EventType.TOOL_RESULT]
    full_text = "".join(e["data"]["content"] for e in events if e["type"] == EventType.MESSAGE)
    reasoning_chunks = [e for e in events if e["type"] == EventType.REASONING]

    print(f"  事件序列     : {types}")
    print(f"  工具调用结果 : {[r['data']['result'] for r in tool_results]}")
    print(f"  reasoning 轮数: {len([e for e in events if e['type'] == EventType.REASONING_COMPLETE])}")
    print(f"  最终文本长度 : {len(full_text)}")
    print(f"  最终文本     : {full_text[:60]!r}...")

    assert len(tool_results) == 2, f"期望2个工具结果，实际{len(tool_results)}"
    assert len(reasoning_chunks) > 0, "没有 reasoning 事件"
    assert "done" in types
    assert events[-1]["data"]["success"] is True
    print("  PASS")


if __name__ == "__main__":
    test_text()
    test_reasoning_and_multi_tool()
    print("\n全部场景通过。")
