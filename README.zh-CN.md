# ftre-agent-core

[English](README.md) | 中文

一个轻量级 Python Agent 框架，从 [ftre](https://github.com/quanming1/ftre) 项目中抽取的核心运行时。

## 背景

ftre 是一个本地运行的 AI 编程助手，类似 Cursor / Windsurf，但完全开源、可自托管。在开发 ftre 的过程中，我们发现 Agent 的核心能力（ReAct 循环、工具系统、LLM 适配、消息管理）是通用的，不应该和 ftre 的业务逻辑耦合在一起。

于是我们把这部分抽取出来，形成了 `ftre-agent-core`。

## 设计理念

**1. 不造轮子，只做胶水**

LLM 调用用 OpenAI SDK，不自己封装 HTTP。工具定义用 JSON Schema，不发明 DSL。尽量让用户用熟悉的方式写代码。

**2. 流式优先**

所有 Agent 执行都是流式的，通过 Generator 逐步 yield 事件。前端可以实时展示思考过程、工具调用、中间结果。

**3. 可中断**

支持运行时取消（`CancellationToken`），工具执行中可随时中断。通过线程安全的取消信号，主循环和工具执行器协同响应用户取消请求。

**4. 协议适配**

不同 LLM 厂商的 API 协议不一样（OpenAI completions vs responses），`ftre-agent-core` 在底层做适配，上层统一用 OpenAI SDK 的接口。

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                      ReActAgent                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │                   ReActRunner                     │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │  │
│  │  │ LLMHandler  │  │ ToolHandler │  │  Memory   │  │  │
│  │  │  (流式调用)  │  │  (工具执行)  │  │ (消息管理) │  │  │
│  │  └─────────────┘  └─────────────┘  └───────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌──────────────┐
    │   LLM 层    │ │  Tool 层    │ │ Cancellation │
    │ (协议适配)   │ │ (注册/执行)  │ │  (取消信号)   │
    └─────────────┘ └─────────────┘ └──────────────┘
```

### 核心模块

| 模块 | 职责 |
|------|------|
| `agent/` | Agent 抽象和 ReAct 实现 |
| `agent/runner/` | 执行引擎：LLM 调用、工具执行、事件分发 |
| `tool/` | 工具定义、注册、中间件、依赖注入 |
| `llm/` | LLM 客户端适配（completions / responses 协议） |
| `memory.py` | 消息历史管理、token 计数 |
| `threading.py` | 全局线程池（按用途分组） |
| `tracing.py` | Agent / LLM / Tool 树状追踪与 exporter |

## 与 LangChain / AutoGen 的区别

| | ftre-agent-core | LangChain | AutoGen |
|---|---|---|---|
| 定位 | 轻量运行时 | 全家桶 | 多 Agent 协作 |
| 依赖 | 只依赖 openai, httpx | 依赖树很深 | 依赖 openai |
| 流式 | 原生支持 | 需要额外配置 | 支持 |
| 工具定义 | `@tool` 装饰器 | 多种方式 | 函数注解 |
| 学习成本 | 低 | 高 | 中 |

我们不追求功能全面，只做好 Agent 执行这一件事。

## 安装

```bash
# 从 GitHub 安装
pip install git+https://github.com/quanming1/ftre-agent-core.git

# 本地开发（editable 模式，改代码立即生效）
git clone https://github.com/quanming1/ftre-agent-core.git
pip install -e ./ftre-agent-core
```

## 快速开始

```python
from ftre_agent_core.agent import ReActAgent
from ftre_agent_core.tool import tool
from ftre_agent_core.llm import create_client

# 定义工具
@tool(description="读取文件内容")
def read_file(path: str) -> str:
    """path: 文件路径"""
    return open(path).read()

@tool(description="写入文件")
def write_file(path: str, content: str) -> str:
    """path: 文件路径, content: 文件内容"""
    open(path, 'w').write(content)
    return f"已写入 {path}"

# 创建客户端
client = create_client(
    api_key="sk-xxx",
    base_url="https://api.openai.com/v1",
)

# 创建 Agent
agent = ReActAgent(
    client=client,
    model="gpt-4",
    system_prompt="你是一个文件处理助手",
    tools=[read_file, write_file],
    max_iterations=20,
)

# 流式执行
for event in agent.stream("读取 config.json 并格式化"):
    print(event.type, event.data)
```

## 本地开发

```bash
# 克隆
git clone https://github.com/quanming1/ftre-agent-core.git
cd ftre-agent-core

# editable 安装
pip install -e .

# 在其他项目中引用（假设在同级目录）
pip install -e ../ftre-agent-core
```

editable 模式下，修改 `ftre-agent-core` 的代码会立即生效，不需要重新安装。

## Tracing

Tracing 默认关闭。显式配置 exporter 后，每次执行会生成一棵
`agent -> llm/tool` run 树，记录输入输出、耗时、状态、错误、usage、
`finish_reason` 和 provider 返回的响应元数据。

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
    "完成任务",
    runtime_context={
        "trace_name": "session-turn",
        "trace_tags": ["desktop"],
        "trace_metadata": {"session_id": "sess_123"},
    },
):
    print(event.type)
```

测试或嵌入式调用可使用 `InMemoryTraceExporter.get_trace(trace_id)` 读取完整
run 树。Exporter 异常只写日志，不会中断 Agent。Trace 会包含完整消息和工具
输入输出，启用持久化 exporter 时应按部署环境处理访问控制和敏感信息。

## 路线图

- [x] ReAct Agent 基础循环
- [x] 工具系统（定义、注册、中间件）
- [x] LLM 协议适配（completions / responses）
- [x] 流式事件输出
- [x] 运行时取消（CancellationToken）
- [x] Agent / LLM / Tool 树状 tracing
- [ ] Checkpoint 快照与恢复
- [ ] 多 Agent 协作
- [ ] 更多 LLM 适配器（Anthropic native、Gemini）
- [ ] 可视化调试工具

## License

MIT

## 相关项目

- [ftre](https://github.com/quanming1/ftre) - 基于此框架构建的 AI 编程助手
