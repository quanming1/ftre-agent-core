# API 参考

## ReActAgent

```python
ReActAgent(
    model: str,
    api_key: str,
    api_base: str | None = None,
    api_type: str = "completions",
    system_prompt: str = "",
    tool_registry: ToolRegistry | None = None,
    max_iterations: int | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str = "",
    memory: MemoryManager | None = None,
    max_retries: int = 5,
    retry_delay: float = 3.0,
    tracer: Tracer | None = None,
    hook_manager: FtreCoreHookManager | None = None,
)
```

### run

```python
async def run(
    message,
    runtime_context: dict | None = None,
) -> AsyncGenerator[AgentStreamEvent, None]
```

### cancel_nowait

请求取消当前运行。最终回复使用 `REPLY_END` 表达完成、错误或中断状态。

## Event

- `EventBase`: `id`, `created_at`, `metadata`
- `AgentStreamEvent`: 所有事件类的联合类型
- `EventType`: 当前事件名称枚举

事件字段采用扁平 Pydantic 模型。事件清单见 [08-events.md](08-events.md)。

## Msg

```python
Msg(
    name: str,
    content: list[ContentBlock],
    role: Literal["user", "assistant", "system"],
    id: str,
    metadata: dict,
    created_at: str,
    usage: Usage | None,
    finished_at: str | None,
    finished_reason: ReplyFinishedReason | None,
    structured_output: dict | None,
    error: dict | None,
)
```

工厂函数：

- `UserMsg(name, content, **kwargs)`
- `AssistantMsg(name, content="", **kwargs)`
- `SystemMsg(name, content, **kwargs)`

`Msg.append_event(event)` 将同一 `reply_id` 的实时事件聚合为消息快照。

## ContentBlock

- `TextBlock`
- `ThinkingBlock`
- `DataBlock`（`Base64Source` / `URLSource`）
- `HintBlock`
- `ToolCallBlock`
- `ToolResultBlock`

## 转换

- `to_openai_part(block)`
- `from_openai_part(part)`
- `to_openai_message(blocks, role=None)`
- `from_openai_message(message)`

## Tool

工具通过 `ToolRegistry` 注册，由 `ReActAgent` 在工具调用事件后执行。工具可以返回
字符串、事件对象，或 `(result, metadata)` 元组。
