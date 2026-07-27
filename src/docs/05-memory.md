# Memory

`MemoryManager` 管理发给 provider 的消息列表。

```python
from ftre_agent_core.memory import MemoryManager

memory = MemoryManager({"system_prompt": "你是助手"})
memory.add_user("你好")
memory.add_assistant("你好！")
memory.add_tool_result(tool_call_id="call_123", content="结果")
memory.add_raw({"role": "assistant", "content": "", "tool_calls": []})

messages = memory.get_messages()
memory.clear()
```

Token 用量由 `MODEL_CALL_END` 的 `input_tokens` / `output_tokens` 提供，
并由 `Msg.append_event()` 聚合到 `Msg.usage`。

Memory 是运行时 provider 消息容器；长期存储应保存 `Msg` 快照。
