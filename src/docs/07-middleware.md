# 中间件

中间件让你在工具执行前后插入自定义逻辑，而不需要修改 Agent 核心代码。

## 执行模型

```
before 链（按注册顺序）→ 工具执行（或短路）→ after 链（逆序，洋葱模型）
```

```
请求进入 →  Middleware A.before
                → Middleware B.before
                    → 工具执行
                ← Middleware B.after
            ← Middleware A.after
        → 返回结果
```

## 定义中间件

继承 `ToolMiddleware`，覆写需要的钩子：

```python
from ftre_agent_core.tool.middleware import ToolMiddleware, ToolContext

class LoggingMiddleware(ToolMiddleware):
    """记录所有工具调用"""
    
    def before(self, context: ToolContext) -> ToolContext:
        print(f"[LOG] 调用 {context.name}({context.arguments})")
        context.metadata["start_time"] = time.time()
        return context
    
    def after(self, context: ToolContext, result) -> any:
        elapsed = time.time() - context.metadata["start_time"]
        print(f"[LOG] {context.name} 完成，耗时 {elapsed:.2f}s")
        return result
```

## 注册中间件

```python
# 通过 ToolRegistry 注册
agent.tools.add_middleware(LoggingMiddleware())
agent.tools.add_middleware(CacheMiddleware())

# 移除
agent.tools.remove_middleware(middleware_instance)

# 查看已注册的中间件
print(agent.tools.middlewares)
```

## ToolContext

`ToolContext` 贯穿整个调用生命周期：

```python
@dataclass
class ToolContext:
    call_id: str                    # 工具调用 ID
    name: str                       # 工具名称
    arguments: dict[str, Any]       # 调用参数
    metadata: dict[str, Any]        # 中间件间通信
    cancel_token: CancellationToken # 取消令牌
    resources: ResourceRegistry     # 资源注册表
```

中间件之间通过 `metadata` 传递数据：

```python
class AuthMiddleware(ToolMiddleware):
    def before(self, context: ToolContext) -> ToolContext:
        context.metadata["user_id"] = get_current_user()
        return context

class AuditMiddleware(ToolMiddleware):
    def after(self, context: ToolContext, result):
        user_id = context.metadata.get("user_id")
        log_audit(user_id, context.name, result)
        return result
```

## 短路执行

`before` 钩子可以跳过实际工具执行，直接返回结果：

```python
class CacheMiddleware(ToolMiddleware):
    def __init__(self):
        self.cache = {}
    
    def before(self, context: ToolContext) -> ToolContext:
        cache_key = f"{context.name}:{json.dumps(context.arguments, sort_keys=True)}"
        if cache_key in self.cache:
            # 命中缓存，跳过实际执行
            context.skip(result=self.cache[cache_key])
        else:
            context.metadata["cache_key"] = cache_key
        return context
    
    def after(self, context: ToolContext, result):
        # 缓存结果
        cache_key = context.metadata.get("cache_key")
        if cache_key and not context.skipped:
            self.cache[cache_key] = result.result
        return result
```

## 修改参数

`before` 可以修改工具的调用参数：

```python
class SanitizeMiddleware(ToolMiddleware):
    def before(self, context: ToolContext) -> ToolContext:
        # 过滤危险字符
        if "path" in context.arguments:
            context.arguments["path"] = sanitize_path(context.arguments["path"])
        return context
```

## 修改结果

`after` 可以修改或替换工具结果：

```python
class TruncateMiddleware(ToolMiddleware):
    """截断过长的工具输出"""
    
    def __init__(self, max_length: int = 5000):
        self.max_length = max_length
    
    def after(self, context: ToolContext, result):
        if len(result.result) > self.max_length:
            result.result = result.result[:self.max_length] + "\n...[已截断]"
        return result
```

## 错误处理

- `before` 抛出异常 → 工具不执行，异常向上传播
- 工具执行抛出 `CancelledError` → 不经过 `after`，直接传播
- `after` 抛出异常 → 异常向上传播

## 实用中间件示例

### 权限控制

```python
class PermissionMiddleware(ToolMiddleware):
    def __init__(self, allowed_tools: set[str]):
        self.allowed = allowed_tools
    
    def before(self, context: ToolContext) -> ToolContext:
        if context.name not in self.allowed:
            context.skip(result=f"[权限不足] 不允许执行 {context.name}")
        return context
```

### 超时控制

```python
class TimeoutMiddleware(ToolMiddleware):
    def __init__(self, timeout_seconds: float = 30):
        self.timeout = timeout_seconds
    
    def before(self, context: ToolContext) -> ToolContext:
        context.metadata["timeout"] = self.timeout
        return context
```

### 重试

```python
class RetryMiddleware(ToolMiddleware):
    def after(self, context: ToolContext, result):
        if result.error and "timeout" in result.error:
            # 标记需要重试
            context.metadata["should_retry"] = True
        return result
```

## 下一步

- [取消机制](./09-cancellation.md) — 取消和资源管理
