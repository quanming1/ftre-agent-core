"""
Tool 依赖注入标记

用法：在 tool 函数参数中声明需要注入的依赖，执行时由 ToolRegistry 自动填充。

    from ftre_agent_core.tool.inject import Injected

    @tool(name="workspace_search", ...)
    def workspace_search(query: str, memory=Injected("memory")) -> str:
        # memory 会在执行时自动注入，不会暴露给 LLM
        ...

注入源通过 ToolRegistry.inject(key, provider) 注册：

    registry.inject("memory", lambda: agent.memory.messages)

provider 是一个 callable，每次 tool 执行时调用取值。
"""


class Injected:
    """
    注入标记

    作为 tool 函数参数的默认值，标记该参数需要从 ToolRegistry 注入。
    参数不会出现在 OpenAI function parameters 中。

    Args:
        key: 注入键名，与 ToolRegistry.inject(key, provider) 的 key 对应
    """

    def __init__(self, key: str):
        self.key = key

    def __repr__(self) -> str:
        return f"Injected({self.key!r})"
