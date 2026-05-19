# 核心概念

## ReAct 循环

ftre-agent-core 的核心是 **ReAct**（Reasoning + Acting）循环：

```
用户输入
    ↓
┌─────────────────────────────┐
│  1. 思考 (Reasoning)         │  LLM 分析问题，决定下一步
│  2. 行动 (Acting)            │  调用工具获取信息
│  3. 观察 (Observation)       │  处理工具返回结果
│  4. 重复                     │  直到能给出最终答案
└─────────────────────────────┘
    ↓
最终回复
```

每次循环称为一次 **迭代（iteration）**。Agent 最多执行 `max_iterations` 次迭代（默认 10），防止无限循环。

## 架构分层

```
ReActAgent          ← 用户接口层（创建、配置、调用）
    │
ReActRunner         ← 执行引擎（编排循环）
    │
    ├── LLMHandler      ← LLM 调用（流式、多协议）
    └── ToolHandler     ← 工具执行（线程池、取消）
```

- **ReActAgent** 是用户直接操作的对象，负责配置和对外 API
- **ReActRunner** 是内部执行引擎，编排 LLM 调用和工具执行
- **Handler** 层各司其职，互不耦合

## 事件驱动

所有执行结果通过 **事件流** 返回，而不是一次性返回完整结果。这使得：

- 流式输出成为可能（逐字显示）
- 调用方可以实时感知执行进度
- 取消可以在任意时刻发生

### 事件类型

| 事件 | 说明 | 触发时机 |
|------|------|----------|
| `MESSAGE` | 文本片段 | LLM 流式输出每个 chunk |
| `MESSAGE_COMPLETE` | 完整文本 | 一次 LLM 回复结束 |
| `TOOL_CALL` | 工具调用 | LLM 决定调用工具 |
| `TOOL_RESULT` | 工具结果 | 工具执行完成 |
| `TOOL_CALL_STREAMING` | 工具调用流式 | tool_call 参数逐步到达 |
| `ERROR` | 错误 | LLM 调用失败 |
| `RETRY` | 重试 | 自动重试中 |
| `DONE` | 完成 | 整个执行结束 |
| `USAGE_UPDATE` | 用量更新 | Token 使用统计 |

### 事件结构

每个事件是一个字典：

```python
{
    "type": EventType.TOOL_CALL,
    "data": {
        "id": "call_abc123",
        "name": "get_weather",
        "arguments": {"city": "北京"}
    }
}
```

## 执行流程详解

一次完整的 `agent.run("查北京天气")` 执行流程：

```
1. run("查北京天气")
   ├── 添加用户消息到 Memory
   └── 进入 _loop()

2. _loop() → _step() [第 1 次迭代]
   ├── 从 Memory 获取完整消息列表
   ├── LLMHandler.stream(messages, tools)
   │   ├── yield StreamDelta(tool_calls=[...])  → TOOL_CALL_STREAMING 事件
   │   └── yield LLMResponse(tool_calls=[...])  → 完整 tool_call
   ├── ToolHandler.execute("get_weather", {"city": "北京"})
   │   ├── yield TOOL_CALL 事件
   │   ├── 在线程池中执行工具
   │   └── yield TOOL_RESULT 事件
   └── 工具结果写入 Memory

3. _loop() → _step() [第 2 次迭代]
   ├── LLMHandler.stream(messages, tools)
   │   └── yield StreamDelta(content="北京今天...")  → MESSAGE 事件
   └── yield MESSAGE_COMPLETE 事件

4. 循环结束
   └── yield DONE 事件
```

## 状态管理

Agent 的运行状态由 `RunState` 管理：

```python
class RunStatus(Enum):
    IDLE         # 空闲，未开始
    RUNNING      # 运行中
    COMPLETED    # 已完成
    ERROR        # 出错
    CANCELLED    # 用户取消
```

状态转换：

```
IDLE → RUNNING → COMPLETED
                → ERROR
                → CANCELLED
```

## Memory 系统

Memory 管理当前循环的消息列表，对 LLM 来说就是 `messages`：

```python
[
    {"role": "system", "content": "你是一个助手..."},
    {"role": "user", "content": "查北京天气"},
    {"role": "assistant", "content": None, "tool_calls": [...]},
    {"role": "tool", "content": "晴天，25°C", "tool_call_id": "call_123"},
    {"role": "assistant", "content": "北京今天晴天，25°C。"},
]
```

Memory 负责：
- 维护消息列表
- Token 使用统计

## 下一步

- [工具系统](./03-tools.md) — 定义和使用工具
- [Memory](./05-memory.md) — 消息管理
