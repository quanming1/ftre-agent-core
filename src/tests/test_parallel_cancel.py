"""测试并行工具执行中取消"""
import time
import threading
from ftre_agent_core.agent import ReActAgent, EventType
from ftre_agent_core.agent.runner import RunStatus
from ftre_agent_core.tool import tool

API_KEY = "sk-REDACTED"
API_BASE = "https://llm-gateway.REDACTED.example.com/v1"
MODEL = "openai/DeepSeek-V3.2"


@tool()
def slow_a(x: str) -> str:
    """工具A（10秒）"""
    time.sleep(10)
    return "A done"


@tool()
def slow_b(x: str) -> str:
    """工具B（10秒）"""
    time.sleep(10)
    return "B done"


@tool()
def slow_c(x: str) -> str:
    """工具C（10秒）"""
    time.sleep(10)
    return "C done"


agent = ReActAgent(
    model=MODEL,
    api_key=API_KEY,
    api_base=API_BASE,
    system_prompt="直接调用所有三个工具，参数都传 test。不要使用think工具。",
    tools=[slow_a, slow_b, slow_c],
)

events = []
tool_calls_seen = threading.Event()
start = time.perf_counter()


def consume():
    for event in agent.run("调用 slow_a slow_b slow_c，参数都是 test"):
        t = time.perf_counter() - start
        events.append(event)
        etype = event["type"].value
        data = event.get("data", {})

        if etype == "tool_call":
            print(f"  [{t:.1f}s] TOOL_CALL: {data['name']}")
            tool_calls_seen.set()
        elif etype == "tool_result":
            print(f"  [{t:.1f}s] TOOL_RESULT: {data['name']} status={data.get('status')} result={data.get('result')}")
        elif etype == "message":
            print(data.get("content", ""), end="", flush=True)
        elif etype == "done":
            print(f"  [{t:.1f}s] DONE: success={data.get('success')} reason={data.get('reason')}")


print("=== 并行工具执行中取消测试 ===")
print("3 个工具各 sleep 10s，2s 后取消\n")

thread = threading.Thread(target=consume)
thread.start()

# 等工具开始执行后 2 秒取消
tool_calls_seen.wait(timeout=30)
time.sleep(2)

print(f"\n  --- [{time.perf_counter()-start:.1f}s] 发送取消信号 ---\n")
cancel_start = time.perf_counter()
agent.cancel_sync()
cancel_done = time.perf_counter()

thread.join(timeout=10)

# 统计
print(f"\n--- 结果 ---")
print(f"取消延迟: {(cancel_done - cancel_start)*1000:.1f}ms")
print(f"总事件数: {len(events)}")
print(f"TOOL_CALL 数: {sum(1 for e in events if e['type'] == EventType.TOOL_CALL)}")
print(f"TOOL_RESULT 数: {sum(1 for e in events if e['type'] == EventType.TOOL_RESULT)}")
tool_results = [e for e in events if e["type"] == EventType.TOOL_RESULT]
for tr in tool_results:
    d = tr["data"]
    print(f"  {d['name']}: status={d.get('status')}")
done_events = [e for e in events if e["type"] == EventType.DONE]
if done_events:
    print(f"DONE reason: {done_events[0]['data'].get('reason')}")
print(f"最终状态: {agent.state.status.value}")
