# 取消机制

ftre-agent-core 提供完整的取消支持，让用户可以随时中止 Agent 执行，同时保证资源正确清理。

## 取消方式

### 异步取消（推荐）

```python
import asyncio

# 在 async 上下文中
await agent.cancel()
# 返回时，所有善后工作已完成
```

### 同步取消

```python
# 在同步代码中（阻塞等待善后完成）
agent.cancel_sync()
```

### 仅发信号（不等待）

```python
# 只发取消信号，不等待善后
agent.cancel_nowait()
# 善后由 generator 消费链路自然完成
```

## 取消流程

```
cancel() 调用
    ↓
1. state.cancel()          ← 设置取消标志位
2. llm.cancel()            ← 硬关 HTTP 连接
    ├── 设置 _cancelled 标志
    └── close_stream()     ← 关闭活跃的流式连接
    ↓
3. 各检查点检测到取消
    ├── state.check_cancel() 抛出 CancelledError
    └── 或 adapter 检查 is_cancelled 退出循环
    ↓
4. _loop() 捕获 CancelledError
    ├── 已有内容写入 Memory
    ├── 未执行的 tool_calls 补上 "[用户取消，未执行]"
    └── yield DONE(reason=CANCELLED) 事件
    ↓
5. _done_event.set()       ← 通知等待者善后完成
```

## CancellationToken

底层取消机制基于 `CancellationToken`，线程安全：

```python
from ftre_agent_core.tool_system import CancellationToken, ToolCancelledError

token = CancellationToken()

# 注册取消回调
unregister = token.on_cancel(lambda reason: print(f"被取消: {reason}"))

# 在工具中检查取消
def my_long_running_tool(cancel_token: CancellationToken):
    for i in range(1000):
        cancel_token.raise_if_cancelled()  # 抛出 ToolCancelledError
        do_work(i)

# 取消
token.cancel(reason="用户取消")

# 检查状态
token.is_cancelled()  # True
token.reason          # "用户取消"
```

### 在工具中使用

工具可以通过中间件的 `ToolContext` 获取 cancel_token：

```python
class MyCancellableMiddleware(ToolMiddleware):
    def before(self, context: ToolContext) -> ToolContext:
        # context.cancel_token 可用于传递给工具内部
        context.metadata["cancel_token"] = context.cancel_token
        return context
```

## ToolExecutionHandle

每次工具执行都有一个 `ToolExecutionHandle` 跟踪状态：

```python
from ftre_agent_core.tool_system import ToolExecutionHandle, ToolExecutionStatus

handle = ToolExecutionHandle(call_id="call_123", name="search")

# 状态转换
handle.transition_to(ToolExecutionStatus.RUNNING)
handle.transition_to(ToolExecutionStatus.COMPLETED)

# 请求取消
handle.request_cancel(reason="用户取消")
# 这会：
# 1. 设置状态为 CANCELLING
# 2. 触发 cancel_token
# 3. 取消所有注册的资源
```

### 状态机

```
PENDING → RUNNING → COMPLETED
                  → FAILED
                  → CANCELLED
                  → TIMED_OUT
         CANCELLING ↗
```

终态（COMPLETED / FAILED / CANCELLED / TIMED_OUT）一旦到达，不可再转换。

## 资源管理

工具执行中可能创建子进程、打开文件等资源。`ResourceRegistry` 统一管理：

```python
from ftre_agent_core.tool_system import ResourceRegistry, ProcessResource

resources = ResourceRegistry()

# 注册子进程
process = subprocess.Popen(["long-running-command"])
resources.register(ProcessResource(process))

# 取消时自动清理所有资源
resources.cancel_all(reason="用户取消")

# 或手动清理
resources.cleanup_all()
```

### 自定义资源

实现 `ManagedResource` 协议：

```python
from ftre_agent_core.tool_system.resources import ManagedResource

class DatabaseConnection(ManagedResource):
    def __init__(self, conn):
        self.conn = conn
    
    def cancel(self, reason: str) -> None:
        self.conn.cancel_query()
    
    def cleanup(self) -> None:
        self.conn.close()
```

## 工具执行中的取消

ToolHandler 在子线程中执行工具，主线程每 100ms 轮询取消信号：

```
主线程                          子线程
  │                               │
  ├── submit(tool_func)           │
  │                               ├── 开始执行工具
  ├── poll (100ms)                │
  │   └── 未取消，继续等          │
  ├── poll (100ms)                │
  │   └── 检测到取消！            │
  │       ├── cancel_token.cancel()
  │       └── 等待子线程结束      ├── raise_if_cancelled()
  │                               └── ToolCancelledError!
  ├── 收到结果（cancelled）
  └── 返回取消结果
```

## 取消后的善后

取消发生时，框架保证：

1. **已产生的内容保留** — 已流式输出的文本写入 Memory
2. **未执行的工具补齐** — 未执行的 tool_calls 补上结果（"[用户取消，未执行]"），保持消息格式完整
3. **资源正确释放** — 子进程终止、连接关闭
4. **状态一致** — RunState 设为 CANCELLED，done_event 触发

## 超时

工具执行超时也通过取消机制处理：

```python
# ToolHandler 内部
handle.transition_to(ToolExecutionStatus.TIMED_OUT)
handle.cancel_token.cancel("timed_out")
```

超时产生 `TOOL_TIMED_OUT` 事件，工具结果标记为 `status="timed_out"`。

## 下一步

- [API 参考](./10-api-reference.md) — 完整的类和方法索引
