"""测试并行工具的事件派发顺序：谁先完成谁先 yield"""
import time
from ftre_agent_core.agent import ReActAgent, EventType
from ftre_agent_core.tool import tool

API_KEY = "sk-HIYFHsm6Oyx1MotZXpxtXOMfDGj6azzPKw3GPQX4RxASrAZH"
API_BASE = "https://llm-gateway.mlamp.cn/v1"
MODEL = "openai/DeepSeek-V3.2"


@tool()
def slow_a(x: str) -> str:
    """工具A（慢，5秒）"""
    time.sleep(5)
    return "A done"


@tool()
def fast_b(x: str) -> str:
    """工具B（快，1秒）"""
    time.sleep(1)
    return "B done"


@tool()
def medium_c(x: str) -> str:
    """工具C（中，3秒）"""
    time.sleep(3)
    return "C done"


agent = ReActAgent(
    model=MODEL,
    api_key=API_KEY,
    api_base=API_BASE,
    system_prompt="直接调用所有三个工具，参数都传 test。不要使用think工具。",
    tools=[slow_a, fast_b, medium_c],
)

start = time.perf_counter()
print("=== 并行工具事件派发顺序测试 ===")
print("slow_a=5s, fast_b=1s, medium_c=3s\n")

for event in agent.run("调用 slow_a fast_b medium_c，参数都是 test"):
    etype = event["type"].value
    t = time.perf_counter() - start
    data = event.get("data", {})

    if etype == "assistant_message_complete":
        for b in data.get("content", []):
            if b.get("type") == "toolCall":
                print(f"  [{t:.1f}s] TOOL_CALL: {b['name']}")
    elif etype == "tool_result":
        print(f"  [{t:.1f}s] TOOL_RESULT: {data['name']} -> {data['result']}")
    elif etype == "done":
        print(f"  [{t:.1f}s] DONE")
