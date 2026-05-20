# API 参考

## ReActAgent

```python
from ftre_agent_core.agent import ReActAgent, EventType
```

### 构造函数

```python
ReActAgent(
    model: str,                      # LiteLLM 模型名
    api_key: str,                    # API 密钥
    api_base: str | None = None,     # 自定义端点
    api_type: str = "completions",   # "completions" 或 "responses"
    system_prompt: str = "",         # 系统提示词
    tools: list[Tool] = None,        # 工具列表
    max_iterations: int = 10,        # 最大迭代次数
    memory: MemoryManager = None,    # 自定义 Memory
)
```

### 方法

| 方法 | 说明 |
|------|------|
| `run(message)` | 运行 ReAct 循环，返回事件 Generator |
| `cancel()` | 异步取消 |
| `cancel_sync()` | 同步取消 |
| `cancel_nowait()` | 仅发信号 |
| `add_tool(tool)` | 动态添加工具 |
| `remove_tool(name)` | 动态移除工具 |

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `state` | `RunState` | 运行状态 |
| `tools` | `ToolRegistry` | 工具注册表 |
| `memory` | `MemoryManager` | 消息管理 |
| `system_prompt` | `str` | 系统提示词 |

---

## EventType

| 值 | 说明 |
|----|------|
| `MESSAGE` | 流式文本 |
| `MESSAGE_COMPLETE` | 完整文本 |
| `REASONING` | 推理过程 |
| `TOOL_CALL` | 工具调用 |
| `TOOL_RESULT` | 工具结果 |
| `TOOL_CALL_STREAMING` | 工具调用流式 |
| `ERROR` | 错误 |
| `RETRY` | 重试 |
| `DONE` | 完成 |
| `USAGE_UPDATE` | 用量 |

---

## Tool

```python
from ftre_agent_core.tool import Tool, ToolParameter, tool, Injected
```

### @tool 装饰器

```python
@tool(name=None, description=None, parameters=None)
def my_tool(...) -> str: ...
```

### Tool 类

| 方法 | 说明 |
|------|------|
| `execute(**kwargs)` | 执行 |
| `to_openai_dict()` | 转 OpenAI 格式 |
| `is_async()` | 是否异步 |

### ToolRegistry

| 方法 | 说明 |
|------|------|
| `register(tool)` | 注册 |
| `unregister(name)` | 注销 |
| `get(name)` | 获取 |
| `has(name)` | 是否存在 |
| `execute(name, **kwargs)` | 执行（含注入） |
| `to_openai_tools()` | 导出全部 |
| `add_middleware(mw)` | 添加中间件 |
| `provide(key, provider)` | 注册注入源 |

---

## MemoryManager

```python
from ftre_agent_core.memory import MemoryManager
```

| 方法 | 说明 |
|------|------|
| `add_user(content)` | 添加用户消息 |
| `add_assistant(content)` | 添加助手消息 |
| `add_tool_result(tool_call_id, content)` | 添加工具结果 |
| `add_raw(message)` | 添加原始消息 |
| `get_messages()` | 获取完整列表（含 system） |
| `clear()` | 清空 |

---

## CancellationToken

```python
from ftre_agent_core.tool import CancellationToken, ToolCancelledError
```

| 方法 | 说明 |
|------|------|
| `cancel(reason)` | 触发取消 |
| `is_cancelled()` | 是否已取消 |
| `raise_if_cancelled()` | 已取消则抛异常 |
| `on_cancel(callback)` | 注册回调 |
| `wait(timeout)` | 等待取消 |

---

## ToolMiddleware

```python
from ftre_agent_core.tool import ToolMiddleware, ToolContext
```

```python
class MyMiddleware(ToolMiddleware):
    def before(self, context: ToolContext) -> ToolContext: ...
    def after(self, context: ToolContext, result) -> result: ...
```

---

## RunState

```python
from ftre_agent_core.agent.runner import RunState, RunStatus
```

| 属性/方法 | 说明 |
|-----------|------|
| `status` | 当前状态 |
| `is_running` | 是否运行中 |
| `is_done` | 是否已结束 |
| `is_cancelled` | 是否已取消 |
