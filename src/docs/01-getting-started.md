# 快速开始

## 安装

```bash
pip install -e ".[dev]"
```

## 创建一个简单的对话 Agent

```python
from ftre_agent_core.agent import ReActAgent

agent = ReActAgent(
    model="openai/gpt-4",
    api_key="sk-your-key",
    api_base="https://api.openai.com/v1",
    system_prompt="你是一个简洁的助手。",
    tools=[],
)

for event in agent.run("你好"):
    if event["type"].value == "message":
        print(event["data"]["content"], end="", flush=True)
print()
```

## 添加工具

```python
from ftre_agent_core.tool import tool

@tool()
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    return f"{city}：晴天，25°C"

@tool()
def calculate(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

agent = ReActAgent(
    model="openai/gpt-4",
    api_key="sk-your-key",
    system_prompt="你是一个多功能助手。",
    tools=[get_weather, calculate],
)
```

## 处理事件流

```python
from ftre_agent_core.agent import EventType

for event in agent.run("北京天气怎么样？"):
    match event["type"]:
        case EventType.MESSAGE:
            print(event["data"]["content"], end="")
        case EventType.REASONING:
            print(f"[思考] {event['data']['content']}", end="")
        case EventType.TOOL_CALL:
            data = event["data"]
            print(f"\n🔧 调用: {data['name']}({data['arguments']})")
        case EventType.TOOL_RESULT:
            print(f"📋 结果: {event['data']['result']}")
        case EventType.DONE:
            print(f"\n✅ 完成")
```

## 多轮对话

```python
list(agent.run("我叫小明"))
for event in agent.run("我叫什么名字？"):
    if event["type"].value == "message":
        print(event["data"]["content"], end="")
```

## 使用不同的 LLM 供应商

```python
# 任何兼容 OpenAI 协议的端点
agent = ReActAgent(
    model="openai/your-model",
    api_key="your-key",
    api_base="https://your-endpoint.com/v1",
    ...
)
```

## 下一步

- [核心概念](./02-core-concepts.md) — ReAct 循环和事件系统
- [工具系统](./03-tools.md) — 深入工具定义
