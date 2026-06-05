"""
用真实模型跑一次多轮工具调用 + reasoning，把原始事件流打印出来，
方便把真实输出抄进 fake_stream。

运行：python tests/probe_real_model.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ftre_agent_core.agent import ReActAgent, EventType
from ftre_agent_core.tool import tool

API_KEY  = "sk-REDACTED"
API_BASE = "https://api.deepseek.com/v1"
MODEL    = "openai/deepseek-reasoner"   # 支持 reasoning_content


@tool(description="查询城市当前天气")
def get_weather(city: str) -> str:
    """city: 城市名"""
    return f"{city}：晴，26°C，东南风3级"


@tool(description="查询城市的空气质量指数")
def get_aqi(city: str) -> str:
    """city: 城市名"""
    return f"{city} AQI：42（优）"


agent = ReActAgent(
    model=MODEL,
    api_key=API_KEY,
    api_base=API_BASE,
    system_prompt="你是一个天气助手，尽量使用工具获取真实数据再回答。",
    tools=[get_weather, get_aqi],
    max_iterations=10,
)

print("开始跑真模型，prompt: 北京今天天气和空气质量怎么样？\n")

for ev in agent.run("北京今天天气和空气质量怎么样？"):
    t = ev["type"].value
    data = ev.get("data", {})

    if t == "message":
        print(f"[{t}] {data['content']!r}")
    elif t == "reasoning":
        print(f"[{t}] {data['content']!r}")
    elif t == "message_complete":
        print(f"[{t}] len={len(data.get('content',''))}")
    elif t == "reasoning_complete":
        print(f"[{t}] len={len(data.get('content',''))}")
    elif t == "tool_call_streaming":
        print(f"[{t}] {data}")
    elif t == "tool_result":
        print(f"[{t}] id={data['id']} name={data['name']} result={data['result']!r}")
    elif t == "tool_call":
        print(f"[{t}] {data}")
    elif t == "usage_update":
        print(f"[{t}] {data}")
    elif t == "done":
        print(f"[{t}] success={data.get('success')} reason={data.get('reason')}")
    else:
        print(f"[{t}] {data}")
