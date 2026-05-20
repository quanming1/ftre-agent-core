# 中间件

在工具执行前后插入自定义逻辑。

## 执行模型

```
before 链（注册顺序）→ 工具执行（或短路）→ after 链（逆序）
```

## 定义中间件

```python
from ftre_agent_core.tool import ToolMiddleware, ToolContext

class LoggingMiddleware(ToolMiddleware):
    def before(self, context: ToolContext) -> ToolContext:
        print(f"调用 {context.name}({context.arguments})")
        return context

    def after(self, context: ToolContext, result):
        print(f"{context.name} 完成")
        return result
```

## 注册

```python
agent.tools.add_middleware(LoggingMiddleware())
agent.tools.remove_middleware(mw)
```

## ToolContext

```python
@dataclass
class ToolContext:
    call_id: str
    name: str
    arguments: dict[str, Any]
    metadata: dict[str, Any]       # 中间件间通信
    cancel_token: CancellationToken
```

## 短路执行

```python
class CacheMiddleware(ToolMiddleware):
    def before(self, context: ToolContext) -> ToolContext:
        cached = self.cache.get(context.name, context.arguments)
        if cached:
            context.skip(result=cached)  # 跳过实际执行
        return context
```

## 修改参数

```python
class SanitizeMiddleware(ToolMiddleware):
    def before(self, context: ToolContext) -> ToolContext:
        context.arguments["path"] = sanitize(context.arguments.get("path", ""))
        return context
```

## 下一步

- [取消机制](./09-cancellation.md)
