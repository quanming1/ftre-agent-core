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

### 示例

```python
import threading
from ftre_agent_core.agent import ReActAgent, EventType
from ftre_agent_core.tool import tool

@tool()
def search(query: str) -> str:
    """搜索信息"""
    return f"关于 {query} 的搜索结果..."

agent = ReActAgent(
    model="openai/gpt-4",
    api_key="sk-xxx",
    api_base="https://api.openai.com/v1",
    system_prompt="你是一个搜索助手。",
    tools=[search],
)

# 基本使用
for event in agent.run("搜索 Python 最新版本"):
    if event["type"] == EventType.MESSAGE:
        print(event["data"]["content"], end="")
    elif event["type"] == EventType.TOOL_CALL:
        print(f"\n调用工具: {event['data']['name']}")
    elif event["type"] == EventType.TOOL_RESULT:
        print(f"工具结果: {event['data']['result']}")

# 多轮对话
list(agent.run("记住我叫小明"))
for event in agent.run("我叫什么？"):
    ...

# 取消（在另一个线程中）
def run_agent():
    for event in agent.run("写一篇很长的文章"):
        print(event["type"].value)

thread = threading.Thread(target=run_agent)
thread.start()
import time; time.sleep(2)
agent.cancel_sync()  # 2 秒后取消
thread.join()
```

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

### 示例：完整事件处理

```python
for event in agent.run("查北京天气"):
    match event["type"]:
        case EventType.REASONING:
            # DeepSeek R1 等模型的推理过程
            print(f"💭 {event['data']['content']}", end="")

        case EventType.MESSAGE:
            print(event["data"]["content"], end="")

        case EventType.MESSAGE_COMPLETE:
            print()  # 换行

        case EventType.TOOL_CALL:
            d = event["data"]
            print(f"🔧 {d['name']}({d['arguments']})")

        case EventType.TOOL_RESULT:
            d = event["data"]
            if d.get("error"):
                print(f"❌ {d['name']} 失败: {d['error']}")
            else:
                print(f"✅ {d['name']}: {d['result']}")

        case EventType.USAGE_UPDATE:
            u = event["data"]["usage"]
            print(f"📊 tokens: {u.prompt_tokens}+{u.completion_tokens}")

        case EventType.ERROR:
            print(f"⚠️ {event['data']['code']}: {event['data']['message']}")

        case EventType.DONE:
            d = event["data"]
            print(f"\n{'✅' if d['success'] else '❌'} {d['reason'].value}")
```

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

### 示例

```python
from ftre_agent_core.tool import tool, Tool, ToolParameter, Injected

# 方式1：装饰器
@tool()
def get_weather(city: str) -> str:
    """获取天气"""
    return f"{city}: 晴天 25°C"

# 方式2：手动指定参数
@tool(
    name="calc",
    description="计算表达式",
    parameters=[ToolParameter(name="expr", type="string", description="数学表达式")]
)
def calculate(expr: str) -> str:
    return str(eval(expr))

# 方式3：继承
class FileTool(Tool):
    name = "read_file"
    description = "读取文件"
    parameters = [ToolParameter(name="path", type="string", description="文件路径")]

    def _run(self, path: str) -> str:
        return open(path).read()

# 依赖注入
@tool()
def context_search(query: str, workspace=Injected("workspace")) -> str:
    """带上下文的搜索"""
    return search_in(workspace, query)

agent.tools.provide("workspace", lambda: "/current/project")

# 动态管理
agent.add_tool(get_weather)
agent.remove_tool("get_weather")
print(agent.tools.names)  # ['calc', 'read_file', ...]
```

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

### 示例

```python
from ftre_agent_core.memory import MemoryManager

memory = MemoryManager({"system_prompt": "你是代码助手"})

# 手动构建对话
memory.add_user("写一个排序函数")
memory.add_assistant("def sort(arr): return sorted(arr)")
memory.add_user("加上类型注解")

print(memory.get_messages())
# [
#   {"role": "system", "content": "你是代码助手"},
#   {"role": "user", "content": "写一个排序函数"},
#   {"role": "assistant", "content": "def sort(arr): return sorted(arr)"},
#   {"role": "user", "content": "加上类型注解"},
# ]

# 传给 Agent 使用已有上下文
agent = ReActAgent(memory=memory, model=..., api_key=...)
for event in agent.run("再加上文档字符串"):
    ...
```

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

### 示例：在工具中支持取消

```python
import time
from ftre_agent_core.tool import tool, CancellationToken, ToolCancelledError

@tool()
def long_task(steps: int) -> str:
    """一个支持取消的长任务"""
    for i in range(steps):
        time.sleep(1)
        # 工具内部无法直接拿到 token，
        # 但框架会在外部轮询取消并中断执行
    return f"完成 {steps} 步"

# 框架层面：工具在子线程执行，主线程每 50ms 检查取消
# 用户调 agent.cancel_sync() 后，工具会在下一个 50ms 窗口被标记为 cancelled
```

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

### 示例

```python
import time
from ftre_agent_core.tool import ToolMiddleware, ToolContext

class TimingMiddleware(ToolMiddleware):
    """记录每个工具的执行时间"""

    def before(self, context: ToolContext) -> ToolContext:
        context.metadata["start"] = time.perf_counter()
        return context

    def after(self, context: ToolContext, result):
        elapsed = time.perf_counter() - context.metadata["start"]
        print(f"[{context.name}] {elapsed:.2f}s")
        return result

class PermissionMiddleware(ToolMiddleware):
    """权限控制"""

    def __init__(self, blocked: set[str]):
        self.blocked = blocked

    def before(self, context: ToolContext) -> ToolContext:
        if context.name in self.blocked:
            context.skip(result=f"[权限不足] {context.name} 被禁止")
        return context

agent.tools.add_middleware(TimingMiddleware())
agent.tools.add_middleware(PermissionMiddleware({"delete_file", "drop_table"}))
```

---

## RunState

```python
from ftre_agent_core.agent.runner import RunState, RunStatus
```

| 属性/方法 | 说明 |
|-----------|------|
| `status` | 当前状态 |
| `iteration` | 当前迭代次数 |
| `is_running` | 是否运行中 |
| `is_done` | 是否已结束 |
| `is_cancelled` | 是否已取消 |
| `error` | 错误信息（ERROR 状态时） |

### 示例

```python
agent = ReActAgent(...)

# 检查状态
print(agent.state.status)  # RunStatus.IDLE

# 运行后
list(agent.run("你好"))
print(agent.state.status)      # RunStatus.COMPLETED
print(agent.state.iteration)   # 1

# 取消后
agent.cancel_sync()
print(agent.state.status)  # RunStatus.CANCELLED
```
