# 技术设计：Responses API Support

> **架构概要：** 在 LLMHandler 中增加 `_stream_responses()` 方法处理 Responses 协议，根据 `api_type` 分发到对应实现。流式事件格式和 Tool Calling 格式需要单独适配，但对上层 Runner 暴露统一的 `StreamDelta` / `LLMResponse` 接口。

## 涉及文件

| 操作 | 文件路径 | 职责 |
|------|----------|------|
| 修改 | `src/ftre_agent_core/agent/base.py` | 构造参数增加 `api_type` |
| 修改 | `src/ftre_agent_core/agent/react.py` | 构造参数增加 `api_type` |
| 修改 | `src/ftre_agent_core/agent/runner/react_runner.py` | 传递 `api_type` 给 LLMHandler |
| 修改 | `src/ftre_agent_core/agent/runner/handler/llm_handler.py` | 核心改动：支持两种协议 |

## 现有代码意图分析

### LLMHandler.stream()

**当前意图**：封装 `litellm.completion()` 流式调用，输出统一的 `StreamDelta` / `LLMResponse`。

**隐式约束**：
- 上层 Runner 依赖 `StreamDelta.content` 和 `StreamDelta.tool_calls` 的格式
- `ToolCallAccumulator` 假设 tool_call 有 `index`、`id`、`function.name`、`function.arguments` 属性
- 取消机制通过 `self._cancelled` 标志位实现

**改动影响**：
- 需要保证 Responses 模式输出的 `StreamDelta` / `LLMResponse` 格式与 Completions 模式完全一致
- Responses 的 tool_call 格式不同，需要适配为统一结构

## 架构决策

### 决策 1：方法拆分

将 `stream()` 拆分为三个方法：
- `stream()` — 公开接口，根据 `api_type` 分发
- `_stream_completion()` — Completions 协议实现（现有逻辑）
- `_stream_responses()` — Responses 协议实现（新增）

理由：逻辑隔离清晰，便于独立测试和维护。

### 决策 2：Responses Tool Call 适配

Responses 的 tool_call 结构与 Completions 不同，需要转换为统一格式：

```python
# Responses 原始格式
item.type == "function_call"
item.name, item.arguments, item.call_id

# 转换为统一的 _ToolCallWrapper 格式
_ToolCallWrapper({
    "id": item.call_id,
    "type": "function",
    "function": {"name": item.name, "arguments": item.arguments}
})
```

### 决策 3：Responses 流式事件处理

Responses 流式事件类型：
- `response.output_text.delta` → 提取 `event.delta` 作为文本增量
- `response.function_call_arguments.delta` → 累积工具参数
- `response.completed` → 完整响应，提取 usage

## 接口设计

### Agent 构造参数

```python
class Agent(ABC):
    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str | None = None,
        api_type: str = "completions",  # 新增
        system_prompt: str = "你是一个有帮助的助手。",
        tools: list[Tool] = None,
        memory: MemoryProtocol | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.api_type = api_type  # 新增
        ...
```

### LLMHandler

```python
class LLMHandler:
    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str | None = None,
        api_type: str = "completions"  # 新增
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.api_type = api_type
        self._cancelled = False

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None
    ) -> Generator[StreamDelta | LLMResponse, None, None]:
        """根据 api_type 分发到对应实现"""
        if self.api_type == "responses":
            yield from self._stream_responses(messages, tools)
        else:
            yield from self._stream_completion(messages, tools)

    def _stream_completion(self, messages, tools):
        """现有 litellm.completion() 逻辑"""
        ...

    def _stream_responses(self, messages, tools):
        """新增 litellm.responses() 逻辑"""
        ...
```

### _stream_responses 实现要点

```python
def _stream_responses(
    self,
    messages: list[dict],
    tools: list[dict] | None = None
) -> Generator[StreamDelta | LLMResponse, None, None]:
    self._cancelled = False
    _dump_llm_input(messages, tools, self.model)

    # responses() 的 input 直接兼容 messages 格式
    response = litellm.responses(
        model=self.model,
        input=messages,
        tools=self._convert_tools_for_responses(tools) if tools else None,
        api_key=self.api_key,
        api_base=self.api_base,
        stream=True,
    )

    content_buffer: list[str] = []
    tool_calls_buffer: list[dict] = []  # 收集 function_call items
    usage = None

    for event in response:
        if self._cancelled:
            break

        event_type = getattr(event, "type", None)

        if event_type == "response.output_text.delta":
            delta_text = getattr(event, "delta", "")
            if delta_text:
                content_buffer.append(delta_text)
                yield StreamDelta(content=delta_text)

        elif event_type == "response.function_call_arguments.delta":
            # 累积工具参数（需要按 call_id 分组）
            ...

        elif event_type == "response.completed":
            # 完整响应，提取 tool_calls 和 usage
            completed_response = getattr(event, "response", None)
            if completed_response:
                usage = getattr(completed_response, "usage", None)
                # 遍历 output 找 function_call
                for item in getattr(completed_response, "output", []):
                    if getattr(item, "type", None) == "function_call":
                        tool_calls_buffer.append({
                            "id": item.call_id,
                            "type": "function",
                            "function": {
                                "name": item.name,
                                "arguments": item.arguments
                            }
                        })

    # 最终输出
    if tool_calls_buffer:
        yield LLMResponse(
            content="".join(content_buffer) if content_buffer else None,
            tool_calls=[_ToolCallWrapper(tc) for tc in tool_calls_buffer],
            usage=usage,
        )
    else:
        if usage:
            yield StreamDelta(usage=usage)
```

### 工具定义格式转换

Responses API 的工具定义格式略有不同，需要转换：

```python
def _convert_tools_for_responses(self, tools: list[dict]) -> list[dict]:
    """将 completion 格式的 tools 转换为 responses 格式"""
    # 如果 LiteLLM 内部已做转换，可能不需要这一步
    # 待实际测试确认
    return tools
```

## 与现有逻辑的关系

```
ReActAgent(api_type="responses")
    │
    └── ReActRunner
            │
            └── LLMHandler(api_type="responses")
                    │
                    ├── stream()
                    │     └── api_type == "responses"?
                    │           ├── Yes → _stream_responses()
                    │           └── No  → _stream_completion()
                    │
                    └── 输出统一的 StreamDelta / LLMResponse
                              ↓
                        ReActRunner（无感知差异）
```

## 测试要点

1. **基础对话**：`api_type="responses"` 下纯文本对话正常
2. **工具调用**：`api_type="responses"` 下工具调用、结果回传正常
3. **流式输出**：文本增量实时 yield
4. **取消机制**：软取消在两种模式下行为一致
5. **向下兼容**：不传 `api_type` 时默认 `"completions"`，行为不变
