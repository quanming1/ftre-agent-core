# 核心概念

## ReAct 循环

核心是 **ReAct**（Reasoning + Acting）循环：

```
用户输入 → [LLM 思考 → 调用工具 → 观察结果] × N → 最终回复
```

每次循环称为一次迭代。Agent 最多执行 `max_iterations` 次（默认 10）。

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

所有执行结果通过事件流返回：

| 事件 | 说明 |
|------|------|
| `MESSAGE` | 流式文本片段 |
| `MESSAGE_COMPLETE` | 完整文本 |
| `REASONING` | 推理过程（DeepSeek R1 等） |
| `TOOL_CALL` | 工具调用 |
| `TOOL_RESULT` | 工具结果 |
| `TOOL_CALL_STREAMING` | 工具调用参数流式 |
| `ERROR` | 错误 |
| `RETRY` | 重试 |
| `DONE` | 完成 |
| `USAGE_UPDATE` | 用量更新 |

事件结构：

```python
{"type": EventType.TOOL_CALL, "data": {"id": "call_123", "name": "get_weather", "arguments": {"city": "北京"}}}
```

## 执行流程

```
run("查北京天气")
  ├── 添加用户消息到 Memory
  └── _loop()
       └── _step()
            ├── LLM 流式调用
            │   ├── StreamDelta(content=...) → MESSAGE 事件
            │   └── LLMResponse(tool_calls=...) → 进入工具执行
            ├── ToolHandler.execute(...)
            │   ├── TOOL_CALL 事件
            │   ├── 线程池执行工具
            │   └── TOOL_RESULT 事件
            └── 下一次迭代...
```

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
