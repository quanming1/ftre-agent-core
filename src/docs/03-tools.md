# 工具系统

## 定义工具

### 装饰器（推荐）

```python
from ftre_agent_core.tool import tool

@tool()
def search_files(query: str, max_results: int = 10) -> str:
    """在工作区中搜索文件内容"""
    return f"找到 {max_results} 个结果"
```

自动从函数签名推断参数名、类型、是否必需。

### 自定义名称和描述

```python
@tool(name="grep", description="正则搜索文件内容")
def search_files(pattern: str, path: str = ".") -> str:
    ...
```

### 继承 Tool 类

```python
from ftre_agent_core.tool import Tool, ToolParameter

class DatabaseQuery(Tool):
    name = "query_db"
    description = "执行数据库查询"
    parameters = [
        ToolParameter(name="sql", type="string", description="SQL 语句"),
    ]

    def _run(self, sql: str) -> str:
        return execute_sql(sql)
```

## 参数类型映射

| Python | JSON Schema |
|--------|-------------|
| `str` | `string` |
| `int`, `float` | `number` |
| `bool` | `boolean` |
| `list` | `array` |
| `dict` | `object` |

## 依赖注入

不暴露给 LLM 的参数，用 `Injected` 标记：

```python
from ftre_agent_core.tool import tool, Injected

@tool()
def workspace_search(query: str, ctx=Injected("context")) -> str:
    """搜索工作区"""
    # ctx 由框架注入，不出现在 LLM 的工具参数中
    return do_search(query, ctx)

# 注册注入源
agent.tools.provide("context", lambda: get_current_context())
```

## 工具注册表

```python
agent.tools.register(my_tool)
agent.tools.unregister("my_tool")
agent.tools.get("my_tool")
agent.tools.has("my_tool")
agent.tools.to_openai_tools()  # 导出 OpenAI 格式
```

## 下一步

- [中间件](./07-middleware.md) — 工具执行前后钩子
