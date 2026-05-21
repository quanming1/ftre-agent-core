# ftre-agent-core 文档

一个轻量级 Python Agent 框架，实现 ReAct（推理 + 行动）循环，支持流式输出、工具调用和取消。

## 快速导航

| 文档 | 说明 |
|------|------|
| [快速开始](./01-getting-started.md) | 5 分钟上手，创建你的第一个 Agent |
| [核心概念](./02-core-concepts.md) | 理解 ReAct 循环、事件流、状态管理 |
| [工具系统](./03-tools.md) | 定义工具、装饰器、参数推断、依赖注入 |
| [Memory](./05-memory.md) | 对话消息管理 |
| [LLM 适配](./06-llm-adapters.md) | 多供应商支持、Completions / Responses 协议 |
| [中间件](./07-middleware.md) | Tool 中间件、执行前后钩子 |
| [事件流](./08-events.md) | AgentEvent 完整规范、产出时机、顺序保证 |
| [取消机制](./09-cancellation.md) | 用户取消、并行工具取消 |
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
├── memory.py           # 消息管理
├── threading.py        # 全局线程池
├── llm/                # LLM 调用（流式、多协议）
│   ├── completion.py   # Completions API + 类型 + LLMHandler
│   ├── responses.py    # Responses API
│   └── utils.py        # 日志
├── tool/               # 工具系统
│   ├── base.py         # Tool + @tool 装饰器 + Injected
│   ├── registry.py     # ToolRegistry + 中间件
│   └── cancellation.py # CancellationToken
└── agent/              # Agent 核心
    ├── react.py        # ReActAgent
    ├── event.py        # 事件类型
    └── runner/         # 执行引擎
        ├── react_runner.py  # ReActRunner + RunState
        └── tool_handler.py  # ToolHandler
```
