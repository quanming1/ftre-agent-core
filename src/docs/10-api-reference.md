# API 参考

## Agent

### ReActAgent

```python
from ftre_agent_core.agent import ReActAgent, EventType
```

#### 构造函数

```python
ReActAgent(
    model: str,                          # LiteLLM 模型名（如 "openai/gpt-4"）
    api_key: str,                        # API 密钥
    api_base: str | None = None,         # 自定义端点
    api_type: str = "completions",       # "completions" 或 "responses"
    system_prompt: str = None,           # 系统提示词
    tools: list[Tool] = None,            # 工具列表
    max_iterations: int = 10,            # 最大迭代次数
    memory: MemoryProtocol = None,       # 自定义 Memory
)
```

#### 方法

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `run(message)` | `Generator[AgentEvent]` | 运行 ReAct 循环 |
| `cancel()` | `coroutine` | 异步取消（等待善后） |
| `cancel_sync()` | `None` | 同步取消（阻塞） |
| `cancel_nowait()` | `None` | 仅发信号 |
| `add_tool(tool)` | `None` | 动态添加工具 |
| `remove_tool(name)` | `None` | 动态移除工具 |

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `state` | `RunState` | 当前运行状态 |
| `tools` | `ToolRegistry` | 工具注册表 |
| `memory` | `MemoryProtocol` | 记忆管理器 |
| `system_prompt` | `str` | 系统提示词（可读写） |

---

### EventType

```python
from ftre_agent_core.agent import EventType
```

| 值 | 说明 |
|----|------|
| `EventType.MESSAGE` | 流式文本片段 |
| `EventType.MESSAGE_COMPLETE` | 完整文本 |
| `EventType.TOOL_CALL` | 工具调用 |
| `EventType.TOOL_RESULT` | 工具结果 |
| `EventType.TOOL_CALL_STREAMING` | 工具调用流式 |
| `EventType.TOOL_CANCEL_REQUESTED` | 工具取消请求 |
| `EventType.TOOL_CANCELLED` | 工具已取消 |
| `EventType.TOOL_TIMED_OUT` | 工具超时 |
| `EventType.ERROR` | 错误 |
| `EventType.RETRY` | 重试 |
| `EventType.DONE` | 完成 |
| `EventType.USAGE_UPDATE` | 用量更新 |

---

## Tool

### tool 装饰器

```python
from ftre_agent_core.tool import tool, ToolParameter
```

```python
@tool(
    name: str = None,                    # 工具名（默认用函数名）
    description: str = None,             # 描述（默认用 docstring）
    parameters: list[ToolParameter] = None,  # 参数（默认自动推断）
)
def my_tool(...) -> str:
    ...
```

### Tool 类

```python
from ftre_agent_core.tool import Tool, ToolParameter
```

#### 构造函数

```python
Tool(
    name: str = None,
    description: str = None,
    parameters: list[ToolParameter] = None,
    func: Callable = None,
)
```

#### 方法

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `execute(**kwargs)` | `str` | 同步执行 |
| `execute_async(**kwargs)` | `coroutine[str]` | 异步执行 |
| `to_openai_dict()` | `dict` | 转为 OpenAI 格式 |
| `is_async()` | `bool` | 是否异步工具 |

### ToolParameter

```python
@dataclass
class ToolParameter:
    name: str              # 参数名
    type: str              # JSON Schema 类型
    description: str       # 参数描述
    required: bool = True  # 是否必需
    enum: list = None      # 可选值列表
```

### ToolRegistry

```python
from ftre_agent_core.tool import ToolRegistry
```

| 方法 | 说明 |
|------|------|
| `register(tool)` | 注册工具 |
| `unregister(name)` | 注销工具 |
| `get(name)` | 获取工具 |
| `has(name)` | 检查是否存在 |
| `execute(name, **kwargs)` | 执行工具（含注入） |
| `execute_async(name, **kwargs)` | 异步执行 |
| `to_openai_tools()` | 导出所有工具为 OpenAI 格式 |
| `add_middleware(mw)` | 添加中间件 |
| `remove_middleware(mw)` | 移除中间件 |
| `provide(key, provider)` | 注册注入源 |

### Injected

```python
from ftre_agent_core.tool.inject import Injected

# 用作参数默认值
def my_tool(query: str, ctx=Injected("context")) -> str:
    ...
```

---

## Memory

### MemoryManager

```python
from ftre_agent_core.memory import MemoryManager
```

#### 构造函数

```python
MemoryManager(options: dict = None)
# options:
#   "system_prompt": str
#   "max_messages": int (默认 100)
```

#### 方法

| 方法 | 说明 |
|------|------|
| `add_user(content)` | 添加用户消息 |
| `add_assistant(content)` | 添加助手消息 |
| `add_tool_result(tool_call_id, content)` | 添加工具结果 |
| `add_raw(message)` | 添加原始消息 |
| `get_messages()` | 获取完整消息列表（含 system） |
| `clear()` | 清空消息 |

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `messages` | `list[dict]` | 消息列表（不含 system） |
| `system_prompt` | `str` | 系统提示词 |

---

## Tool System（底层）

### CancellationToken

```python
from ftre_agent_core.tool_system import CancellationToken, ToolCancelledError
```

| 方法 | 说明 |
|------|------|
| `cancel(reason)` | 触发取消 |
| `is_cancelled()` | 检查是否已取消 |
| `raise_if_cancelled()` | 已取消则抛异常 |
| `on_cancel(callback)` | 注册回调（返回 unregister 函数） |
| `wait(timeout)` | 阻塞等待取消 |

### ToolExecutionHandle

```python
from ftre_agent_core.tool_system import ToolExecutionHandle, ToolExecutionStatus
```

| 方法 | 说明 |
|------|------|
| `transition_to(status)` | 状态转换 |
| `request_cancel(reason)` | 请求取消 |
| `finish(result)` | 标记完成 |
| `snapshot()` | 获取当前快照 |

### ResourceRegistry

```python
from ftre_agent_core.tool_system import ResourceRegistry
```

| 方法 | 说明 |
|------|------|
| `register(resource)` | 注册资源 |
| `unregister(resource)` | 注销资源 |
| `cancel_all(reason)` | 取消所有资源 |
| `cleanup_all()` | 清理所有资源 |

---

## Middleware

### ToolMiddleware

```python
from ftre_agent_core.tool.middleware import ToolMiddleware, ToolContext
```

```python
class MyMiddleware(ToolMiddleware):
    def before(self, context: ToolContext) -> ToolContext:
        ...
    
    def after(self, context: ToolContext, result) -> result:
        ...
```

### ToolContext

```python
@dataclass
class ToolContext:
    call_id: str
    name: str
    arguments: dict[str, Any]
    metadata: dict[str, Any]
    cancel_token: CancellationToken
    resources: ResourceRegistry
```

| 方法 | 说明 |
|------|------|
| `skip(result="")` | 短路执行 |
| `skipped` | 是否已短路 |
| `skip_result` | 短路结果 |

---

## Threading

### ThreadPoolRegistry

```python
from ftre_agent_core.threading import thread_pool
```

| 池 | 用途 | 线程数 |
|----|------|--------|
| `thread_pool.chat` | Agent 循环 | 16 |
| `thread_pool.io` | 文件 I/O | 16 |
| `thread_pool.background` | 后台任务 | 8 |
| `thread_pool.tool` | 工具执行 | 24 |

```python
# 使用
future = thread_pool.tool.submit(fn, arg1, arg2)

# 关闭
thread_pool.shutdown()
```

---

## Prompt

### PromptManager

```python
from ftre_agent_core.prompt import prompts
```

| 方法 | 说明 |
|------|------|
| `register(name, text)` | 注册提示词 |
| `load_file(path)` | 从文件加载 |
| `load_dir(path)` | 批量加载目录 |
| `get(name)` | 获取原始文本 |
| `render(name, **vars)` | 渲染（变量替换） |
| `has(name)` | 是否存在 |
| `unregister(name)` | 移除 |

变量语法：`{{ variable_name }}`
