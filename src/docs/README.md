# ftre-agent-core 文档

欢迎使用 ftre-agent-core —— 一个轻量级 Python Agent 框架，实现 ReAct（推理 + 行动）循环，支持流式输出、工具调用和状态快照。

## 快速导航

| 文档 | 说明 |
|------|------|
| [快速开始](./01-getting-started.md) | 5 分钟上手，创建你的第一个 Agent |
| [核心概念](./02-core-concepts.md) | 理解 ReAct 循环、事件流、状态管理 |
| [工具系统](./03-tools.md) | 定义工具、装饰器、参数推断、依赖注入 |
| [Memory](./05-memory.md) | 对话消息管理 |
| [LLM 适配](./06-llm-adapters.md) | 多供应商支持、Completions / Responses 协议 |
| [中间件](./07-middleware.md) | Tool 中间件、执行前后钩子 |
| [取消机制](./09-cancellation.md) | 用户取消、超时、资源清理 |
| [API 参考](./10-api-reference.md) | 完整类和方法索引 |

## 安装

```bash
pip install -e ".[dev]"
```

需要 Python >= 3.11。

## 最小示例

```python
from ftre_agent_core.agent import ReActAgent
from ftre_agent_core.tool import tool

@tool()
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    return f"{city}：晴天，25°C"

agent = ReActAgent(
    model="openai/gpt-4",
    api_key="sk-xxx",
    system_prompt="你是一个天气助手。",
    tools=[get_weather],
)

for event in agent.run("北京天气怎么样？"):
    if event["type"].value == "message":
        print(event["data"]["content"], end="")
```

## 项目结构

```
src/ftre_agent_core/
├── agent/              # Agent 核心（ReAct 循环、事件、Runner）
│   ├── runner/         # 执行引擎（LLM 调用、工具处理）
│   └── event.py        # 事件类型定义
├── tool/               # 工具系统（定义、注册、装饰器）
├── tool_system/        # 底层执行基础设施（取消、资源、结果）
├── memory/             # 消息管理
├── prompt/             # 提示词管理（模板、渲染）
└── threading.py        # 全局线程池
```
