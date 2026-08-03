# Agent state and message context

`AgentState` owns the serializable conversation context. `MessageContext` is a
stateless helper that reads and updates that caller-owned list.

```python
from ftre_agent_core import AgentState, MessageContext
from ftre_agent_core.message import TextBlock, ToolCallBlock

state = AgentState()
MessageContext.add_user(state.context, "你好")
MessageContext.append_reply_blocks(
    state.context,
    reply_id="reply_123",
    blocks=[
        TextBlock(text="我来调用 echo。"),
        ToolCallBlock(
            id="call_123",
            name="echo",
            arguments={"text": "你好"},
        ),
    ],
)
MessageContext.add_tool_result(
    state.context,
    reply_id="reply_123",
    tool_call_id="call_123",
    name="echo",
    content="结果",
)

messages = MessageContext.get_messages(state.context, system_prompt="你是助手")
MessageContext.clear(state.context)
```

`AgentState.model_dump(mode="json")` produces persistable data and
`AgentState.model_validate(data)` restores typed `Msg` objects.

Token usage is supplied by `MODEL_CALL_END` and aggregated by
`Msg.append_event()` into `Msg.token`:

- `token.usage`: total usage of all LLM calls in the current reply;
- `token.last_call_usage`: usage reported by the final successful LLM call.
