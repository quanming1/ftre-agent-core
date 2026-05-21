# 核心概念

## ReAct 循环

核心是 **ReAct**（Reasoning + Acting）循环：

```
用户输入 → [LLM 思考 → 调用工具 → 观察结果] × N → 最终回复
```

每次循环称为一次迭代。`max_iterations` 默认 `None`（无限循环，直到 LLM 不再调用工具或被取消）。

## 架构

```
ReActAgent          ← 用户接口
    │
ReActRunner         ← 执行引擎
    │
    ├── LLMHandler      ← LLM 流式调用
    └── ToolHandler     ← 工具执行（并行、取消）
```

## 事件驱动

所有执行结果通过事件流返回。详见 [事件流文档](./08-events.md)。

| 事件 | 说明 |
|------|------|
| `MESSAGE` | 流式文本增量 |
| `MESSAGE_COMPLETE` | 一段文本的完整值 |
| `REASONING` | 推理过程增量（DeepSeek R1 等） |
| `TOOL_CALL` | 工具开始执行 |
| `TOOL_RESULT` | 工具执行完成 |
| `TOOL_CALL_STREAMING` | 工具调用参数流式增量 |
| `TOOL_CANCEL_REQUESTED` | 工具取消请求已发出 |
| `TOOL_CANCELLED` | 工具已确认取消 |
| `TOOL_TIMED_OUT` | 工具执行超时 |
| `ERROR` | LLM 调用失败 |
| `RETRY` | LLM 调用失败正在重试 |
| `DONE` | ReAct 循环结束（终止信号） |
| `USAGE_UPDATE` | Token 用量更新 |

事件结构：

```python
{"type": EventType.TOOL_CALL, "data": {"id": "call_123", "name": "get_weather", "arguments": {"city": "北京"}}}
```

## 执行流程

```
run("查北京天气")
  ├── 添加用户消息到 Memory
  └── _loop()
       └── _step()  (每次迭代)
            ├── LLM 流式调用
            │   ├── StreamDelta(content=...) → MESSAGE 事件
            │   ├── StreamDelta(reasoning=...) → REASONING 事件
            │   ├── StreamDelta(tool_calls=...) → TOOL_CALL_STREAMING 事件
            │   ├── StreamDelta(usage=...) → USAGE_UPDATE 事件
            │   └── LLMResponse(tool_calls=...) → 进入工具执行
            │       ├── USAGE_UPDATE 事件
            │       └── MESSAGE_COMPLETE 事件（如果有前置文本）
            ├── _handle_tool_calls(response)
            │   ├── 解析 JSON 参数
            │   │   └── 解析失败 → TOOL_RESULT(error="[PARSE_ERROR]...")
            │   └── ToolHandler.execute(...)
            │       ├── TOOL_CALL 事件（每个工具）
            │       ├── 线程池并行执行
            │       └── TOOL_RESULT 事件（每个工具完成时）
            └── 下一次迭代（或 DONE）
```

**终止条件**（任一满足即停止循环）：
- LLM 回复纯文本（不调用工具）→ DONE(completed)
- 达到 max_iterations → DONE(max_iterations)
- LLM 调用失败 → ERROR + DONE(error)
- 用户取消 → DONE(cancelled)

## 状态

```python
class RunStatus(Enum):
    IDLE        # 未开始
    RUNNING     # 运行中
    COMPLETED   # 完成
    ERROR       # 出错
    CANCELLED   # 取消
```

## 下一步

- [工具系统](./03-tools.md)
- [Memory](./05-memory.md)
