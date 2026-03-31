# 任务清单：Responses API Support

> **目标：** 让 ftre-agent-core 支持 `api_type="responses"` 的 LLM 厂商
> **技术栈：** LiteLLM responses API

---

### Task 1: Agent 构造参数增加 api_type

**文件：**
- 修改: `src/ftre_agent_core/agent/base.py`
- 修改: `src/ftre_agent_core/agent/react.py`

- [ ] **Step 1: 修改 Agent 基类**

```python
# src/ftre_agent_core/agent/base.py

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

- [ ] **Step 2: 修改 ReActAgent**

```python
# src/ftre_agent_core/agent/react.py

def __init__(
    self,
    model: str,
    api_key: str,
    api_base: str | None = None,
    api_type: str = "completions",  # 新增
    system_prompt: str = None,
    tools: list[Tool] = None,
    max_iterations: int = 10,
    interrupt_before: list[str] = None,
    interrupt_all: bool = False,
    memory=None,
):
    ...
    super().__init__(
        model=model,
        api_key=api_key,
        api_base=api_base,
        api_type=api_type,  # 新增
        system_prompt=system_prompt or default_prompt,
        tools=tools,
        memory=memory,
    )
```

---

### Task 2: ReActRunner 传递 api_type

**文件：**
- 修改: `src/ftre_agent_core/agent/runner/react_runner.py`

- [ ] **Step 1: 修改 LLMHandler 实例化**

```python
# 第 63 行附近
self.llm = LLMHandler(
    agent.model,
    agent.api_key,
    agent.api_base,
    agent.api_type  # 新增
)
```

---

### Task 3: LLMHandler 支持 api_type 分发

**文件：**
- 修改: `src/ftre_agent_core/agent/runner/handler/llm_handler.py`

- [ ] **Step 1: 修改构造函数**

```python
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
    self.api_type = api_type  # 新增
    self._cancelled = False
```

- [ ] **Step 2: 重命名现有 stream() 为 _stream_completion()**

将现有 `stream()` 方法完整移动到 `_stream_completion()`，保持逻辑不变。

- [ ] **Step 3: 新建 stream() 分发方法**

```python
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
```

---

### Task 4: 实现 _stream_responses()

**文件：**
- 修改: `src/ftre_agent_core/agent/runner/handler/llm_handler.py`

- [ ] **Step 1: 添加 _stream_responses 方法**

```python
def _stream_responses(
    self,
    messages: list[dict],
    tools: list[dict] | None = None
) -> Generator[StreamDelta | LLMResponse, None, None]:
    """Responses API 流式调用"""
    self._cancelled = False
    _dump_llm_input(messages, tools, self.model)

    response = litellm.responses(
        model=self.model,
        input=messages,  # 直接兼容 messages 格式
        tools=tools if tools else None,
        api_key=self.api_key,
        api_base=self.api_base,
        stream=True,
    )

    content_buffer: list[str] = []
    tool_calls_buffer: list[dict] = []
    usage = None

    try:
        for event in response:
            if self._cancelled:
                break

            event_type = getattr(event, "type", None)

            # 文本增量
            if event_type == "response.output_text.delta":
                delta_text = getattr(event, "delta", "")
                if delta_text:
                    content_buffer.append(delta_text)
                    yield StreamDelta(content=delta_text)

            # 完成事件
            elif event_type == "response.completed":
                completed_response = getattr(event, "response", None)
                if completed_response:
                    usage = getattr(completed_response, "usage", None)
                    # 遍历 output 找 function_call
                    for item in getattr(completed_response, "output", []):
                        if getattr(item, "type", None) == "function_call":
                            tool_calls_buffer.append({
                                "id": getattr(item, "call_id", ""),
                                "type": "function",
                                "function": {
                                    "name": getattr(item, "name", ""),
                                    "arguments": getattr(item, "arguments", "")
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
    finally:
        pass
```

---

### Task 5: 更新迁移文档

**文件：**
- 修改: `docs/litellm-migration.md`

- [ ] **Step 1: 添加 api_type 参数说明**

在「参数说明」表格中添加：

```markdown
| `api_type` | `str` | 协议类型，`"completions"`（默认）或 `"responses"` |
```

- [ ] **Step 2: 添加 Responses API 使用示例**

```markdown
### Responses API 厂商

某些厂商使用 Responses 协议：

```python
agent = ReActAgent(
    model="openai/gpt-5.4",
    api_key="sk-xxx",
    api_base="http://ai.xiamai.top/v1",
    api_type="responses",  # 指定 Responses 协议
    ...
)
```
```

---

### Task 6: 集成测试

**文件：**
- 修改: `src/tests/test_agent_integration.py`

- [ ] **Step 1: 添加 Responses 模式测试（如有可用的 Responses 厂商）**

```python
class TestAgentResponses:
    """Responses API 测试"""

    @pytest.mark.skip(reason="需要 Responses 协议厂商")
    def test_responses_simple_chat(self):
        """Responses 模式简单对话"""
        agent = ReActAgent(
            model="openai/gpt-5.4",
            api_key="sk-xxx",
            api_base="http://ai.xiamai.top/v1",
            api_type="responses",
            system_prompt="你是一个简洁的助手",
            tools=[],
        )
        events = list(agent.run("你好"))
        event_types = [e["type"] for e in events]
        assert EventType.DONE in event_types
```

---

### Task 7: 验证

- [ ] **Step 1: 运行现有测试确保向下兼容**

```bash
py -m pytest src/tests/test_agent_integration.py -v
```

预期: 所有测试通过（默认 api_type="completions"）

- [ ] **Step 2: 提交**

```bash
git add -A
git commit -m "feat: support api_type=responses for LiteLLM Responses API

- Add api_type parameter to Agent/ReActAgent (default: completions)
- LLMHandler dispatches to _stream_completion or _stream_responses
- Adapt responses stream events and tool_call format
- Update migration docs"
```
