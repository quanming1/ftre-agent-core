"""测试工具执行报错的处理"""
import time
from ftre_agent_core.agent import ReActAgent, EventType
from ftre_agent_core.tool import tool

API_KEY = "sk-REDACTED"
API_BASE = "https://llm-gateway.REDACTED.example.com/v1"
MODEL = "openai/DeepSeek-V3.2"


@tool()
def divide(a: int, b: int) -> str:
    """除法计算"""
    return str(a / b)


@tool()
def read_file(path: str) -> str:
    """读取文件内容"""
    with open(path, "r") as f:
        return f.read()


@tool()
def crash_tool(msg: str) -> str:
    """一个一定会崩溃的工具"""
    raise RuntimeError(f"工具内部崩溃: {msg}")


agent = ReActAgent(
    model=MODEL,
    api_key=API_KEY,
    api_base=API_BASE,
    system_prompt="你是一个助手。使用提供的工具完成任务。如果工具报错，告诉用户错误信息。",
    tools=[divide, read_file, crash_tool],
)

print("=== 工具执行报错测试 ===\n")

# 测试1: 除零错误
print("--- 测试1: 除零错误 ---")
for event in agent.run("计算 10 除以 0"):
    etype = event["type"].value
    data = event.get("data", {})
    if etype == "tool_call":
        print(f"  [TOOL_CALL] {data['name']}({data['arguments']})")
    elif etype == "tool_result":
        print(f"  [TOOL_RESULT] status={data.get('status')}")
        print(f"    result: {data.get('result')}")
        print(f"    error: {data.get('error')}")
    elif etype == "message":
        print(data.get("content", ""), end="", flush=True)
    elif etype == "done":
        print(f"\n  [DONE] success={data.get('success')}")

# 清空 memory 开始新对话
agent.memory.clear()

# 测试2: 文件不存在
print("\n--- 测试2: 文件不存在 ---")
for event in agent.run("读取文件 /nonexistent/path.txt"):
    etype = event["type"].value
    data = event.get("data", {})
    if etype == "tool_call":
        print(f"  [TOOL_CALL] {data['name']}({data['arguments']})")
    elif etype == "tool_result":
        print(f"  [TOOL_RESULT] status={data.get('status')}")
        print(f"    result: {data.get('result')[:100]}")
        print(f"    error: {data.get('error', '')[:100] if data.get('error') else None}")
    elif etype == "message":
        print(data.get("content", ""), end="", flush=True)
    elif etype == "done":
        print(f"\n  [DONE] success={data.get('success')}")

agent.memory.clear()

# 测试3: 工具内部 raise
print("\n--- 测试3: 工具内部 raise RuntimeError ---")
for event in agent.run("调用 crash_tool，参数 msg 为 'boom'"):
    etype = event["type"].value
    data = event.get("data", {})
    if etype == "tool_call":
        print(f"  [TOOL_CALL] {data['name']}({data['arguments']})")
    elif etype == "tool_result":
        print(f"  [TOOL_RESULT] status={data.get('status')}")
        print(f"    result: {data.get('result')}")
        print(f"    error: {data.get('error')}")
    elif etype == "message":
        print(data.get("content", ""), end="", flush=True)
    elif etype == "done":
        print(f"\n  [DONE] success={data.get('success')}")
