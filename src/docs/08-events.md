# 流式事件（AgentStreamEvent）

`ReActAgent.run()` 产出 Pydantic 事件对象。所有事件继承 `EventBase`，公共字段为：

```python
{
    "id": "16位十六进制 ID",
    "created_at": "ISO 8601",
    "metadata": {},
    "type": "TEXT_BLOCK_DELTA",
    # 事件专属字段直接位于顶层
}
```

不存在 `data` 包裹层，也不存在聚合完成事件。Event 是实时传输/trace 模型；
消息快照由 `Msg` 表示。

## 生命周期

- `REPLY_START(session_id, reply_id, name, role)`
- `REPLY_END(session_id, reply_id, finished_reason, error)`
- `MODEL_CALL_START(reply_id, model_name)`
- `MODEL_CALL_END(reply_id, prompt_tokens, completion_tokens, total_tokens, finished_reason)`
- `EXCEED_MAX_ITERS(reply_id, name)`
- `retry(reply_id, code, message, attempt, max_attempts)`
- `CUSTOM(name, value)`

## 内容块

文本、思考和二进制数据均采用 start/delta/end 三段式：

- `TEXT_BLOCK_START` / `TEXT_BLOCK_DELTA` / `TEXT_BLOCK_END`
- `THINKING_BLOCK_START` / `THINKING_BLOCK_DELTA` / `THINKING_BLOCK_END`
- `DATA_BLOCK_START` / `DATA_BLOCK_DELTA` / `DATA_BLOCK_END`
- `HINT_BLOCK` 是一次性事件

同一内容块由 `block_id` 关联，同一回复由 `reply_id` 关联。

## 工具

- `TOOL_CALL_START(reply_id, tool_call_id, tool_call_name)`
- `TOOL_CALL_DELTA(reply_id, tool_call_id, delta)`
- `TOOL_CALL_END(reply_id, tool_call_id)`
- `TOOL_RESULT_START(reply_id, tool_call_id, tool_call_name)`
- `TOOL_RESULT_TEXT_DELTA(reply_id, tool_call_id, delta)`
- `TOOL_RESULT_DATA_DELTA(reply_id, tool_call_id, block_id, media_type, data, url)`
- `TOOL_RESULT_END(reply_id, tool_call_id, state, metadata)`

## 聚合为 Msg

```python
from ftre_agent_core.message import AssistantMsg

message = None
async for event in agent.run(prompt):
    if event.type == "REPLY_START":
        message = AssistantMsg(
            name=event.name,
            content=[],
            id=event.reply_id,
            created_at=event.created_at,
        )
    if message is not None and getattr(event, "reply_id", None) == message.id:
        message.append_event(event)
    if event.type == "REPLY_END":
        persist(message)  # 一条 Msg，而不是一批 Event
```

`Msg.append_event()` 会聚合正文、思考、工具参数、工具结果、usage 和结束状态。
