# 工具系统

工具是 Agent 与外部世界交互的桥梁。LLM 通过工具获取实时信息、执行操作、调用 API。

## 定义工具

### 方式一：装饰器（推荐）

最简单的方式，自动从函数签名推断参数：

```python
from ftre_agent_core.tool import tool

@tool()
def search_files(query: str, max_results: int = 10) -> str:
    """在工作区中搜索文件内容"""
    # 实现搜索逻辑
    return f"找到 {max_results} 个匹配 '{query}' 的结果"
```

框架会自动：
- 用函数名作为工具名（`search_files`）
- 用 docstring 作为工具描述
- 从类型注解推断参数类型
- 有默认值的参数标记为可选

### 自定义名称和描述

```python
@tool(name="grep", description="使用正则表达式搜索文件内容")
def search_files(pattern: str, path: str = ".") -> str:
    """内部实现"""
    ...
```

### 手动指定参数

当自动推断不够精确时：

```python
from ftre_agent_core.tool import tool, ToolParameter

@tool(
    name="create_file",
    description="创建新文件",
    parameters=[
        ToolParameter(name="path", type="string", description="文件路径"),
        ToolParameter(name="content", type="string", description="文件内容"),
        ToolParameter(
            name="mode",
            type="string",
            description="写入模式",
            required=False,
            enum=["overwrite", "append"]
        ),
    ]
)
def create_file(path: str, content: str, mode: str = "overwrite") -> str:
    ...
```

### 方式二：继承 Tool 类

适合复杂工具或需要状态的场景：

```python
from ftre_agent_core.tool import Tool, ToolParameter

class DatabaseQuery(Tool):
    name = "query_db"
    description = "执行数据库查询"
    parameters = [
        ToolParameter(name="sql", type="string", description="SQL 语句"),
        ToolParameter(name="limit", type="number", description="最大行数", required=False),
    ]

    def __init__(self, connection_string: str):
        super().__init__()
        self.conn = connect(connection_string)

    def _run(self, sql: str, limit: int = 100) -> str:
        results = self.conn.execute(sql, limit=limit)
        return format_results(results)
```

## 参数类型映射

Python 类型自动映射为 JSON Schema 类型：

| Python 类型 | JSON Schema 类型 |
|-------------|-----------------|
| `str` | `string` |
| `int` | `number` |
| `float` | `number` |
| `bool` | `boolean` |
| `list` | `array` |
| `dict` | `object` |

## 异步工具

工具可以是异步函数：

```python
@tool()
async def fetch_url(url: str) -> str:
    """获取 URL 内容"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()
```

框架自动检测并正确处理异步工具。

## 依赖注入

有些参数不应该暴露给 LLM（如内部状态、上下文对象），使用 `Injected` 标记：

```python
from ftre_agent_core.tool.inject import Injected

@tool()
def workspace_search(query: str, memory=Injected("memory")) -> str:
    """搜索工作区，使用对话上下文优化结果"""
    # memory 由框架自动注入，不会出现在 LLM 的工具参数中
    context = memory  # 可以访问对话历史
    return do_search(query, context)
```

注册注入源：

```python
agent.tools.provide("memory", lambda: agent.memory.messages)
agent.tools.provide("workspace", lambda: current_workspace)
```

注入的参数：
- 不会出现在发送给 LLM 的 function parameters 中
- 每次工具执行时调用 provider 获取最新值
- 如果调用方显式传了该参数，注入不会覆盖

## 工具注册表

`ToolRegistry` 管理所有工具：

```python
# 通过 Agent 访问
registry = agent.tools

# 动态添加/移除工具
agent.add_tool(my_tool)
agent.remove_tool("my_tool")

# 查询
registry.has("get_weather")  # True/False
registry.get("get_weather")  # Tool 对象或 None

# 手动执行（绕过 Agent 循环）
result = registry.execute("get_weather", city="北京")

# 导出为 OpenAI 格式
openai_tools = registry.to_openai_tools()
```

## 内置工具

框架自动注册一个内置工具 `think`：

```python
# LLM 可以调用 think 进行内部推理
# mode="think": 深度分析，决定下一步行动
# mode="reflect": 自我审查，确认结果正确
```

这个工具不执行任何外部操作，只是给 LLM 一个"思考空间"，帮助它做出更好的决策。

## OpenAI 格式输出

每个工具可以转换为 OpenAI function calling 格式：

```python
weather_tool.to_openai_dict()
# {
#     "type": "function",
#     "function": {
#         "name": "get_weather",
#         "description": "获取指定城市的天气",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string", "description": "参数 city"}
#             },
#             "required": ["city"]
#         }
#     }
# }
```

## 下一步

- [中间件](./07-middleware.md) — 工具执行前后的钩子
