"""
取消执行 - 工具执行中取消测试

测试场景：
1. 定义一个耗时 10 秒的工具
2. Agent 调用该工具时，在工具执行中途发送取消
3. 验证取消延迟、事件流和最终状态
"""
import time
import threading
import json
from ftre_agent_core.agent import ReActAgent, EventType
from ftre_agent_core.agent.runner import RunStatus
from ftre_agent_core.tool import tool

API_KEY = "sk-HIYFHsm6Oyx1MotZXpxtXOMfDGj6azzPKw3GPQX4RxASrAZH"
API_BASE = "https://llm-gateway.mlamp.cn/v1"
MODEL = "openai/DeepSeek-V3.2"


@tool()
def long_running_task(task_name: str) -> str:
    """执行一个耗时很长的任务（模拟 10 秒）"""
    print(f"\n  [工具开始] {task_name}，预计 10 秒...")
    for i in range(100):
        time.sleep(0.1)  # 总共 10 秒
        if i % 10 == 0:
            print(f"  [工具进度] {i}%", flush=True)
    return f"{task_name} 完成！"


def test_cancel_during_tool():
    agent = ReActAgent(
        model=MODEL,
        api_key=API_KEY,
        api_base=API_BASE,
        system_prompt="你是一个助手。当用户要求执行任务时，使用 long_running_task 工具。",
        tools=[long_running_task],
    )

    events = []
    tool_call_seen = threading.Event()

    def consume():
        for event in agent.run("请执行一个名为'数据分析'的任务"):
            events.append(event)
            etype = event["type"].value
            data = event.get("data", {})

            if etype == "message":
                print(data.get("content", ""), end="", flush=True)
            elif etype == "assistant_message_complete":
                for b in data.get("content", []):
                    if b.get("type") == "toolCall":
                        print(f"\n  [TOOL_CALL] {b['name']}({b['arguments']})")
                        tool_call_seen.set()
            elif etype == "tool_result":
                print(f"  [TOOL_RESULT] status={data.get('status')}, result={data.get('result')[:50]}...")
            elif etype == "step":
                phase = data.get("phase")
                if phase == "turn_end":
                    print(f"\n  [STEP] success={data.get('success')}, reason={data.get('reason')}")

    # 启动消费线程
    thread = threading.Thread(target=consume)
    thread.start()

    # 等待工具开始执行
    print("等待工具被调用...")
    tool_call_seen.wait(timeout=30)

    # 工具执行 2 秒后取消
    time.sleep(2)
    print(f"\n\n--- 工具执行中，发送取消信号 ---")
    cancel_start = time.perf_counter()
    agent.cancel_sync()
    cancel_done = time.perf_counter()

    thread.join(timeout=10)

    # 统计
    latency_ms = (cancel_done - cancel_start) * 1000
    # tool_call 现在嵌入 assistant_message_complete 的 content[]
    amc_with_tools = [e for e in events if e["type"] == EventType.ASSISTANT_MESSAGE_COMPLETE
                      and any(b.get("type") == "toolCall" for b in e["data"]["content"])]
    tool_results = [e for e in events if e["type"] == EventType.TOOL_RESULT]
    done_events = [e for e in events if e["type"] == EventType.STEP
                   and e["data"].get("phase") == "turn_end"]

    print(f"\n--- 工具取消测试结果 ---")
    print(f"取消延迟: {latency_ms:.1f}ms")
    print(f"总事件数: {len(events)}")
    print(f"含 toolCall 的 AMC 事件: {len(amc_with_tools)}")
    print(f"TOOL_RESULT 事件: {len(tool_results)}")
    if tool_results:
        print(f"  status: {tool_results[0]['data'].get('status')}")
    print(f"STEP turn_end 事件: {len(done_events) > 0}")
    if done_events:
        print(f"  reason: {done_events[0]['data'].get('reason')}")
    print(f"最终状态: {agent.state.status.value}")
    print(f"状态正确: {agent.state.status == RunStatus.CANCELLED}")


if __name__ == "__main__":
    test_cancel_during_tool()
