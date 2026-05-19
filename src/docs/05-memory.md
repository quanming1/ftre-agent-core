# Memory 系统

Memory 管理当前 ReAct 循环中的消息列表，是 LLM 理解上下文的基础。

## MemoryManager

框架内置的 `MemoryManager` 管理单次执行中的消息和 token 统计：

```python
from ftre_agent_core.memory import MemoryManager

memory = MemoryManager({
    "system_prompt": "你是一个助手。",
    "max_messages": 100,
})
```

## 消息操作

```python
# 添加消息
memory.add_user("你好")
memory.add_assistant("你好！有什么可以帮你的？")
memory.add_tool_result(tool_call_id="call_123", content="执行成功")

# 添加原始 OpenAI 格式消息
memory.add_raw({
    "role": "assistant",
    "content": None,
    "tool_calls": [{"id": "call_123", "type": "function", ...}]
})

# 获取消息列表（含 system prompt，用于发送给 LLM）
messages = memory.get_messages()

# 获取消息列表（不含 system）
raw_messages = memory.messages
```

## 消息格式

发送给 LLM 的消息列表结构：

```python
[
    {"role": "system", "content": "你是一个助手..."},
    {"role": "user", "content": "查北京天气"},
    {"role": "assistant", "content": None, "tool_calls": [...]},
    {"role": "tool", "content": "晴天，25°C", "tool_call_id": "call_123"},
    {"role": "assistant", "content": "北京今天晴天，25°C。"},
]
```

## Token 统计

```python
# 累计 token 使用（由 Runner 自动调用）
memory.token.add(usage)

# 查看统计
print(memory.token.prompt_tokens)
print(memory.token.completion_tokens)
print(memory.token.total_tokens)
print(memory.token.request_count)
```

## 生命周期

MemoryManager 的生命周期与单次 `agent.run()` 调用对齐：

```python
agent = ReActAgent(...)

# 每次 run() 时，消息会累积到 memory 中
list(agent.run("你好"))          # memory: [user, assistant]
list(agent.run("查天气"))        # memory: [user, assistant, user, assistant+tool_calls, tool, assistant]

# 如果需要清空重新开始
agent.memory.clear()
```

## 自定义 Memory

你可以传入自己的 MemoryManager 实例：

```python
memory = MemoryManager({"system_prompt": "自定义提示词"})

agent = ReActAgent(
    model="openai/gpt-4",
    api_key="sk-xxx",
    memory=memory,
    tools=[...],
)
```

也可以继承 `MemoryManager` 扩展行为：

```python
class MyMemory(MemoryManager):
    def after_loop(self):
        """循环结束后的自定义逻辑"""
        print(f"本次循环使用了 {self.token.total_tokens} tokens")
```

## 下一步

- [LLM 适配](./06-llm-adapters.md) — 多供应商和协议支持
- [取消机制](./09-cancellation.md) — 运行中取消和资源清理
