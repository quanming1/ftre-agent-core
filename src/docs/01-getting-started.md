# 快速开始

本指南帮助你在 5 分钟内创建第一个 Agent。

## 安装

```bash
pip install -e ".[dev]"
```

## 创建一个简单的对话 Agent

最简单的 Agent 只需要模型配置：

```python
from ftre_agent_core.agent import ReActAgent

agent = ReActAgent(
    model="openai/gpt-4",
    api_key="sk-your-key",
    api_base="https://api.openai.com/v1",  # 可选，自定义端点
    system_prompt="你是一个简洁的助手。",
    tools=[],
)

# 运行对话
for event in agent.run("你好"):
    if event["type"].value == "message":
        print(event["data"]["content"], end="", flush=True)
print()
```

`agent.run()` 返回一个事件生成器，你可以逐个消费事件实现流式输出。

## 添加工具

Agent 的核心能力是调用工具。用 `@tool()` 装饰器定义工具：

```python
from ftre_agent_core.tool import tool

@tool()
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    weather_data = {
        "北京": "晴天，25°C",
        "上海": "多云，28°C",
    }
    return weather_data.get(city, f"{city}：数据未知")

@tool()
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"
```

把工具传给 Agent：

```python
agent = ReActAgent(
    model="openai/gpt-4",
    api_key="sk-your-key",
    system_prompt="你是一个多功能助手，可以查天气和做计算。",
    tools=[get_weather, calculate],
)
```

## 处理事件流

`agent.run()` 产生的事件类型：

```python
from ftre_agent_core.agent import EventType

for event in agent.run("北京天气怎么样？然后帮我算 123*456"):
    match event["type"]:
        case EventType.MESSAGE:
            # 流式文本片段
            print(event["data"]["content"], end="")
        case EventType.TOOL_CALL:
            # Agent 决定调用工具
            data = event["data"]
            print(f"\n🔧 调用工具: {data['name']}({data['arguments']})")
        case EventType.TOOL_RESULT:
            # 工具返回结果
            data = event["data"]
            print(f"📋 结果: {data['result']}")
        case EventType.DONE:
            # 执行完成
            print(f"\n✅ 完成 (成功: {event['data']['success']})")
```

## 多轮对话

Agent 自动维护对话上下文：

```python
# 第一轮
list(agent.run("我叫小明"))

# 第二轮 —— Agent 记得你的名字
for event in agent.run("我叫什么名字？"):
    if event["type"].value == "message":
        print(event["data"]["content"], end="")
# 输出: 你叫小明
```

## 使用不同的 LLM 供应商

框架通过 LiteLLM 支持多种供应商，只需调整 `model`、`api_key`、`api_base`：

```python
# 阿里云通义千问
agent = ReActAgent(
    model="openai/qwen-plus",
    api_key="sk-xxx",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ...
)

# 百度千帆
agent = ReActAgent(
    model="openai/ernie-4.0",
    api_key="bce-v3/your-key",
    api_base="https://qianfan.baidubce.com/v2",
    ...
)

# 自部署 / 兼容 OpenAI 的任何端点
agent = ReActAgent(
    model="openai/your-model",
    api_key="your-key",
    api_base="http://localhost:8000/v1",
    ...
)
```

## 下一步

- [核心概念](./02-core-concepts.md) — 理解 ReAct 循环和事件系统
- [工具系统](./03-tools.md) — 深入工具定义和高级用法
