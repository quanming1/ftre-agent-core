# 取消机制

## 取消方式

```python
# 异步
await agent.cancel()

# 同步（阻塞等待善后）
agent.cancel_sync()

# 仅发信号（不等待）
agent.cancel_nowait()
```

## 取消流程

```
cancel() 调用
  ↓
1. state.cancel()       ← 设取消标志
2. llm.cancel()         ← 硬关 HTTP 连接
  ↓
3. 流式循环检测到取消 → CancelledError
  ↓
4. _loop() 捕获 → yield DONE(CANCELLED)
```

取消信号设置毫秒级。LLM HTTP 连接强制关闭后立即中断流式输出。工具执行中的取消延迟 ≤ 50ms（轮询间隔）。

## LLM 调用中取消

HTTP 连接被强制关闭，`for chunk in response` 立即退出。不需要等下一个 chunk。

## 工具执行中取消

ToolHandler 主线程每 50ms 轮询取消信号：

```
主线程                    子线程
  │                         │
  ├── submit(tool)          ├── 执行工具...
  ├── poll (50ms)           │
  ├── 检测到取消！          │
  │   └── future.cancel()  │
  └── yield tool_result(cancelled)
```

## 并行工具取消

多个工具并行执行时取消，所有未完成的工具都会收到取消结果：

```python
# 3 个工具并行，取消后：
TOOL_RESULT: tool_a status=cancelled
TOOL_RESULT: tool_b status=cancelled
TOOL_RESULT: tool_c status=cancelled
DONE: reason=CANCELLED
```

## CancellationToken

底层取消信号，线程安全：

```python
from ftre_agent_core.tool import CancellationToken

token = CancellationToken()
token.cancel(reason="用户取消")
token.is_cancelled()        # True
token.raise_if_cancelled()  # 抛 ToolCancelledError

# 注册回调
unregister = token.on_cancel(lambda reason: cleanup())
```

## 下一步

- [API 参考](./10-api-reference.md)
