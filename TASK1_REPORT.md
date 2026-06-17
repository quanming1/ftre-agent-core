# TASK1 Report: Event Dict → Class 改造

## 改了什么

将 `AgentEvent` 从 `dict` 别名升级为 `@dataclass` 基类，定义 11 个事件子类，
同时保持所有现有公共 API 向后兼容。

### 核心变化

1. **`AgentEvent` 基类** (`@dataclass`)
   - `type: EventType` 字段（`init=False`，由子类在 `__post_init__` 中设置）
   - `to_dict() -> dict`：序列化为 `{"type": "...", "data": {...}}`，与旧格式 100% 兼容
   - `from_dict(d: dict) -> AgentEvent`：从 dict 反序列化（委托给 `_from_type` 工厂）
   - `_data_dict() -> dict`：子类覆盖，返回 data 段
   - `__getitem__` / `get()` / `__contains__`：向后兼容 dict 风格访问

2. **11 个事件子类**
   - `ToolCallEvent`: tool_id, tool_name, arguments
   - `ToolResultEvent`: tool_id, tool_name, result, error, status, error_code, metadata
   - `MessageEvent`: content
   - `MessageCompleteEvent`: content
   - `ReasoningEvent`: content
   - `ReasoningCompleteEvent`: content
   - `DoneEvent`: success, reason, usage
   - `ErrorEvent`: message, code
   - `RetryEvent`: code, message, attempt, max_attempts
   - `ToolCallStreamingEvent`: tool_calls
   - `UsageUpdateEvent`: usage

   每个子类的 `__post_init__` 使用 `object.__setattr__` 设置 `type` 字段。
   `_data_dict()` 将新字段名映射回原始 dict key（如 `tool_id` → `id`）。

3. **`_from_type(t, data)` 分发工厂**：根据 type 字符串分派到对应子类

### 保留不变

- 所有 11 个构造函数签名不变（`tool_call_event()`, `tool_result_event()` 等）
- `to_dict()` 输出与旧 `{"type": "...", "data": {...}}` 格式 100% 兼容
- `TypedDict` 类型（`ToolCallData`, `ToolResultData` 等）全部保留
- `AgentEventDict = dict` 别名保留
- 所有现有 import 路径不变
- 消费者仍可使用 `event["type"]` / `event["data"]` / `event.get("data", {})` 访问

### 向后兼容机制

通过 `__getitem__` 和 `get()` 方法，新 dataclass 实例仍然支持 dict 风格的
`event["type"]` / `event["data"]` 访问模式，所有现有消费者代码无需修改。

## 测试结果

- **test_simplify_verification.py**: 49/49 passed ✅
  - 包含新增 4 个事件类测试（`test_agent_event_is_class`, `test_agent_event_to_dict`,
    `test_agent_event_from_dict`, `test_agent_event_dict_backward_compat`）
- **test_retry.py**: 2/2 passed ✅
- **test_tool_base.py**: 5/5 passed ✅
- **test_tool_decorator.py**: 7/7 passed ✅
- **test_tool_registry.py**: 9/9 passed ✅
- 总计：**72/72 passed**（不含需要真实 LLM API 的集成测试和已有 bug 的取消/速度测试）

其余 26 个失败均为预存问题（async generator 在同步线程中的迭代错误、
`cancel_sync` 方法不存在），与本次改动无关。

## 改动文件列表

| 文件 | 改动类型 |
|------|----------|
| `src/ftre_agent_core/agent/event.py` | 重写：从 dict 升级为 @dataclass 体系 |
| `src/ftre_agent_core/agent/__init__.py` | 更新：导出新 dataclass 类型 + `AgentEventDict` |
| `src/tests/test_simplify_verification.py` | 更新：`test_agent_event_is_dict` → 4 个新测试 |

## Commit Hash

`cbceffa`（改动前 HEAD，改动尚未 commit）
