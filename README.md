# ftre-agent-core

English | [中文](README.zh-CN.md)

A lightweight Python Agent framework, extracted from the [ftre](https://github.com/quanming1/ftre) project as the core runtime.

## Background

ftre is a local-first AI coding assistant, similar to Cursor / Windsurf, but fully open-source and self-hostable. While developing ftre, we found that the core Agent capabilities (ReAct loop, tool system, LLM adaptation, message management) are generic and shouldn't be coupled with ftre's business logic.

So we extracted this part into `ftre-agent-core`.

## Design Principles

**1. Don't reinvent wheels, just glue**

LLM calls use the OpenAI SDK — no custom HTTP wrappers. Tool definitions use JSON Schema — no invented DSL. We let users write code in familiar ways.

**2. Streaming first**

All Agent execution is streaming, yielding events step by step via a Generator. Frontends can display reasoning, tool calls, and intermediate results in real time.

**3. Interruptible**

Supports runtime cancellation via `CancellationToken`. Tools can be interrupted at any time during execution. Thread-safe cancel signals coordinate between the main loop and tool executors.

**4. Protocol adaptation**

Different LLM vendors have different API protocols (OpenAI completions vs responses). `ftre-agent-core` adapts at the lower layer, exposing a unified OpenAI SDK interface upstream.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      ReActAgent                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │                   ReActRunner                     │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │  │
│  │  │ LLMHandler  │  │ ToolHandler │  │  Memory   │  │  │
│  │  │ (streaming) │  │ (execution) │  │ (messages) │  │  │
│  │  └─────────────┘  └─────────────┘  └───────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                           │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
     ┌─────────────┐ ┌─────────────┐ ┌──────────────┐
     │  LLM Layer  │ │ Tool Layer  │ │ Cancellation │
     │ (protocol)  │ │ (registry)  │ │   (signal)   │
     └─────────────┘ └─────────────┘ └──────────────┘
```

### Core Modules

| Module | Role |
|------|------|
| `agent/` | Agent abstraction and ReAct implementation |
| `agent/runner/` | Execution engine: LLM calls, tool execution, event dispatch |
| `tool/` | Tool definition, registration, middleware, dependency injection |
| `llm/` | LLM client adapters (completions / responses protocol) |
| `memory.py` | Message history management, token counting |
| `threading.py` | Global thread pool (grouped by purpose) |
| `tracing.py` | Agent / LLM / Tool tree-shaped tracing and exporters |

## Comparison with LangChain / AutoGen

| | ftre-agent-core | LangChain | AutoGen |
|---|---|---|---|
| Focus | Lightweight runtime | All-in-one toolkit | Multi-agent collaboration |
| Dependencies | Only openai, httpx | Deep dependency tree | Depends on openai |
| Streaming | Native | Requires extra config | Supported |
| Tool definition | `@tool` decorator | Multiple approaches | Function annotations |
| Learning curve | Low | High | Medium |

We don't aim to be feature-complete — we just do Agent execution well.

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/quanming1/ftre-agent-core.git

# Local development (editable mode)
git clone https://github.com/quanming1/ftre-agent-core.git
pip install -e ./ftre-agent-core
```

## Quick Start

```python
from ftre_agent_core.agent import ReActAgent
from ftre_agent_core.tool import tool
from ftre_agent_core.llm import create_client

# Define tools
@tool(description="Read file contents")
def read_file(path: str) -> str:
    """path: file path"""
    return open(path).read()

@tool(description="Write file")
def write_file(path: str, content: str) -> str:
    """path: file path, content: file content"""
    open(path, 'w').write(content)
    return f"Written to {path}"

# Create client
client = create_client(
    api_key="sk-xxx",
    base_url="https://api.openai.com/v1",
)

# Create Agent
agent = ReActAgent(
    client=client,
    model="gpt-4",
    system_prompt="You are a file processing assistant",
    tools=[read_file, write_file],
    max_iterations=20,
)

# Stream execution
for event in agent.stream("Read config.json and format it"):
    print(event.type, event.data)
```

## Local Development

```bash
git clone https://github.com/quanming1/ftre-agent-core.git
cd ftre-agent-core
pip install -e .
```

In editable mode, changes to `ftre-agent-core` code take effect immediately without reinstalling.

## Tracing

Tracing is disabled by default. When an exporter is explicitly configured, each execution generates an
`agent -> llm/tool` run tree recording inputs, outputs, duration, status, errors, usage,
`finish_reason`, and provider response metadata.

```python
from ftre_agent_core import JsonlTraceExporter, Tracer
from ftre_agent_core.agent import ReActAgent

tracer = Tracer([JsonlTraceExporter(".ftre/traces.jsonl")])
agent = ReActAgent(
    model="gpt-4.1",
    api_key="sk-xxx",
    tracer=tracer,
)

async for event in agent.run(
    "Complete the task",
    runtime_context={
        "trace_name": "session-turn",
        "trace_tags": ["desktop"],
        "trace_metadata": {"session_id": "sess_123"},
    },
):
    print(event.type)
```

For testing or embedded usage, use `InMemoryTraceExporter.get_trace(trace_id)` to read the full
run tree. Exporter exceptions only log and never interrupt the Agent. Traces contain full messages
and tool inputs/outputs — handle access control and sensitive information appropriately when
enabling persistent exporters.

## Roadmap

- [x] ReAct Agent core loop
- [x] Tool system (definition, registration, middleware)
- [x] LLM protocol adaptation (completions / responses)
- [x] Streaming event output
- [x] Runtime cancellation (CancellationToken)
- [x] Agent / LLM / Tool tree-shaped tracing
- [ ] Checkpoint snapshot and restore
- [ ] Multi-agent collaboration
- [ ] More LLM adapters (Anthropic native, Gemini)
- [ ] Visual debugging tools

## License

MIT

## Related Projects

- [ftre](https://github.com/quanming1/ftre) - AI coding assistant built on this framework
