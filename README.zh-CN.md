# ftre-agent-core

[English](README.md) | 中文

一个从 [ftre](https://github.com/quanming1/ftre) 抽取的轻量、无状态 Python Agent 运行时库。
宿主负责配置、持久化、插件和生命周期；本包拥有 ReAct 执行循环，并提供类型化的
消息/事件、Hook 规范、工具抽象、权限引擎和追踪能力。

## 设计原则

- **轻内核**：不持有 Channel、Session、Plugin 注册表或进程级可变状态。
- **流式优先**：`ReActAgent.run()` 是异步生成器，逐个产出 `AgentStreamEvent`。
- **协议唯一 Owner**：`StreamChunk` 与 OpenAI 兼容适配器由 `ftre-llm` 提供。
- **Hook 由宿主注入**：Core 只依赖异步 `HookDispatcher` 协议，作用域和生命周期由宿主决定。
- **可取消**：取消信号会传播到 LLM 和并行工具任务。

## 架构

```text
ReActAgent
└─ ReActRunner
   ├─ ReasoningExecutor  ── 消费 ftre-llm 的 StreamChunk 流
   ├─ ActingExecutor     ── ToolHandler ── ToolRegistry
   ├─ MessageContext      ── 调用方持有的 AgentState.context
   ├─ PermissionEngine
   ├─ HookDispatcher      ── 宿主注入的异步 Hook
   └─ Tracer              ── 可选导出器
```

Hook 规范包括 `tool/before`、`tool/after`、`llm/stream`、`llm/error`、
`agent/before-reasoning` 和 `agent/stop-decision`。Core 只定义类型化契约；监听器的注册
与生命周期作用域由宿主负责。

> 归属说明：`StreamChunk` 协议与 OpenAI 兼容适配器由 `ftre-llm` 唯一拥有；Core 的
> ReActAgent/ReActRunner 只消费 `ftre-llm` 的流协议。执行循环的长期归属（Core Runner
> 与 ftre Agent Runtime turn_executor 的收敛）见主仓库 TODO。

## 安装

```bash
pip install ftre-agent-core
# or for local development
pip install -e ".[dev]"
```

需要 Python 3.12。需要 OpenAI 兼容协议适配器或 Hook 规范引用的 `StreamChunk`
payload 类型时，请单独安装 `ftre-llm`。

## 快速开始

```python
import asyncio
import os

from ftre_agent_core.agent import ReActAgent
from ftre_agent_core.tool import ToolRegistry, tool


@tool(description="Add two integers")
def add_numbers(a: int, b: int) -> str:
    return str(a + b)


async def main() -> None:
    registry = ToolRegistry()
    registry.register(add_numbers)
    agent = ReActAgent(
        model="gpt-4.1-mini",
        api_key=os.environ["OPENAI_API_KEY"],
        system_prompt="You are a concise assistant.",
        tool_registry=registry,
        max_iterations=10,
    )
    async for event in agent.run("Calculate 7 + 5."):
        if event.type == "TEXT_BLOCK_DELTA":
            print(event.delta, end="", flush=True)
        elif event.type == "REPLY_END":
            print(f"\nfinished: {event.finished_reason}")


asyncio.run(main())
```

宿主已有统一 LLM Service 时，可在构造 `ReActAgent` 时传 `llm=` 注入实现
`stream(messages, tools)` 与 `cancel()` 的适配器；运行中不允许更换。

## 持久状态与追踪

`AgentState.context` 保存类型化 `Msg` 对象并可用 Pydantic 序列化。用 `MessageContext`
追加消息或将调用方持有的 context 转换为 provider messages。`Tracer` 默认关闭；需要
agent → LLM/tool 运行树时配置 `InMemoryTraceExporter` 或 `JsonlTraceExporter`。导出器
失败只记录日志，不中断执行。

## 目录结构

```text
src/ftre_agent_core/
├─ agent/                # ReAct 状态机与执行器（ReActAgent/ReActRunner）
├─ llm/                  # 基于 ftre-llm 的适配层、注册表、归一化
├─ tool/                # Tool、@tool、Injected、ToolRegistry、取消
├─ message/             # Msg 与 ContentBlock 模型及转换器
├─ permission/          # allow/deny/ask 引擎
├─ state/               # 可序列化 AgentState
├─ event/               # 流式 AgentStreamEvent 模型
├─ hooks.py             # 类型化宿主 Hook 契约
└─ tracing.py           # 可选追踪树与导出器
```

## License

MIT

## 相关项目

- [ftre](https://github.com/quanming1/ftre) — 基于这些契约的宿主应用
