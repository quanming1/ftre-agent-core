# Tool Metadata 传递机制

> Tool 执行结果支持携带 metadata，用于从 ToolContext 向 tool_result_event 传递上下文信息

## 核心数据结构

### ToolResultData (event 类型)
```python
class ToolResultData(TypedDict, total=False):
    id: str
    name: str
    result: str
    error: str | None
    status: str
    error_code: str | None
    metadata: dict[str, Any]  # 新增
```

### ToolResult (handler dataclass)
```python
@dataclass
class ToolResult:
    call_id: str
    name: str
    result: str
    error: str | None = None
    status: str = "completed"
    metadata: dict = field(default_factory=dict)  # 新增
```

## 业务流程

### Metadata 复制链路
tool_handler:execute_cancellable → _build_result → ToolContext.metadata → ToolResult.metadata → tool_result_event → ToolResultData.metadata

## 核心文件

| 文件 | 职责 |
|------|------|
| `src/ftre_agent_core/agent/event.py` | ToolResultData 定义、tool_result_event() 函数 |
| `src/ftre_agent_core/agent/runner/handler/tool_handler.py` | ToolResult dataclass、metadata 复制逻辑 |

## 关键实现

- `_build_result()` 中: `metadata=dict(ctx.metadata)` 将 context 的 metadata 复制到 ToolResult
- `tool_result_event()` 支持 metadata 参数，条件加入 data
- `execute_and_emit()` 从 result 中提取 metadata 传递给 event

## 使用场景

- 工具中间件需要在工具结果中传递额外上下文
- 工具执行链路追踪
- 工具调用审计信息携带
