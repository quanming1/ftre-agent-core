# Memory

管理当前 ReAct 循环中的消息列表。

## 基本用法

```python
from ftre_agent_core.memory import MemoryManager

# 通过 Agent 自动创建
agent = ReActAgent(system_prompt="你是助手", ...)

# 或手动传入
memory = MemoryManager({"system_prompt": "你是助手"})
agent = ReActAgent(memory=memory, ...)
```

## 消息操作

```python
memory.add_user("你好")
memory.add_assistant("你好！")
memory.add_tool_result(tool_call_id="call_123", content="结果")
memory.add_raw({"role": "assistant", "content": None, "tool_calls": [...]})

# 获取完整消息列表（含 system）
messages = memory.get_messages()

# 清空
memory.clear()
```

## 消息格式

```python
[
    {"role": "system", "content": "你是助手"},
    {"role": "user", "content": "查天气"},
    {"role": "assistant", "content": None, "tool_calls": [...]},
    {"role": "tool", "content": "晴天", "tool_call_id": "call_123"},
    {"role": "assistant", "content": "今天晴天。"},
]
```

## Token 用量

通过事件流中的 `USAGE_UPDATE` 事件获取，每次 LLM 调用后触发：

```python
for event in agent.run("..."):
    if event["type"] == EventType.USAGE_UPDATE:
        print(event["data"]["usage"])
```

## 下一步

- [LLM 适配](./06-llm-adapters.md)
