# ftre-agent-core

English | [中文](README.zh-CN.md)

A small, stateless Python agent runtime library extracted from [ftre](https://github.com/quanming1/ftre).
The host owns configuration, persistence, plugins and lifecycle; this package owns the ReAct
execution loop, typed messages/events, hook contracts, tool abstractions, permissions and tracing.

## Design principles

- **Core is stateless**: no Channel, Session, Plugin registry or process-wide mutable state.
- **Streaming first**: `ReActAgent.run()` is an async generator of typed `AgentStreamEvent` values.
- **One protocol owner**: `ftre-llm` owns `StreamChunk` and the OpenAI-compatible adapters.
- **Host-injected hooks**: Core accepts an async `HookDispatcher`; the host decides scope and lifecycle.
- **Interruptible**: cancellation propagates through the LLM and parallel tool tasks.

## Architecture

```text
ReActAgent
└─ ReActRunner
   ├─ ReasoningExecutor  ── consumes ftre-llm StreamChunk streams
   ├─ ActingExecutor     ── ToolHandler ── ToolRegistry
   ├─ MessageContext      ── caller-owned AgentState.context
   ├─ PermissionEngine
   ├─ HookDispatcher      ── host-provided async hooks
   └─ Tracer              ── optional exporters
```

Hook specifications are `tool/before`, `tool/after`, `llm/stream`, `llm/error`,
`agent/before-reasoning` and `agent/stop-decision`. Core defines the typed contract; the host
registers listeners and supplies lifecycle scope.

> 归属说明：`StreamChunk` 协议与 OpenAI 兼容适配器由 `ftre-llm` 唯一拥有；Core 的
> ReActAgent/ReActRunner 只消费 `ftre-llm` 的流协议。执行循环的长期归属（Core Runner
> 与 ftre Agent Runtime turn_executor 的收敛）见主仓库 TODO。

## Installation

```bash
pip install ftre-agent-core
# or for local development
pip install -e ".[dev]"
```

Requires Python 3.12. Install `ftre-llm` separately when you need OpenAI-compatible
protocol adapters or the `StreamChunk` payload types referenced by the hook specs.

## Quick start

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

For a host-owned LLM service, pass `llm=` to `ReActAgent` at construction time. The object must
implement `stream(messages, tools)` and `cancel()`; the runner rejects replacement while a run is
in flight.

## Persistent state and tracing

`AgentState.context` contains typed `Msg` objects and can be serialized with Pydantic. Use
`MessageContext` to add messages or convert the caller-owned context to provider messages.
`Tracer` is disabled by default; configure `InMemoryTraceExporter` or `JsonlTraceExporter` when
you need an agent → LLM/tool run tree. Exporter failures are logged and never break execution.

## Repository layout

```text
src/ftre_agent_core/
├─ agent/                # ReAct state machine and executors (ReActAgent/ReActRunner)
├─ llm/                  # adapter seam over ftre-llm, registry, normalization
├─ tool/                # Tool, @tool, Injected, ToolRegistry, cancellation
├─ message/             # Msg and ContentBlock models/converters
├─ permission/          # allow/deny/ask engine
├─ state/               # serializable AgentState
├─ event/               # streaming AgentStreamEvent models
├─ hooks.py             # typed host Hook contracts
└─ tracing.py           # optional trace tree and exporters
```

## License

MIT

## Related projects

- [ftre](https://github.com/quanming1/ftre) — the host application built on these contracts
