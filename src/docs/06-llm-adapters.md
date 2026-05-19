# LLM 适配

ftre-agent-core 通过 [LiteLLM](https://github.com/BerriAI/litellm) 支持多种 LLM 供应商，并提供两种协议适配器。

## 供应商配置

### 模型命名格式

使用 LiteLLM 的 `provider/model` 格式：

```python
# OpenAI
model = "openai/gpt-4"

# 自定义端点（兼容 OpenAI 协议的任何服务）
model = "openai/your-model-name"
api_base = "https://your-endpoint.com/v1"
```

### 常见供应商示例

```python
# OpenAI 官方
agent = ReActAgent(
    model="openai/gpt-4",
    api_key="sk-xxx",
)

# 阿里云通义千问
agent = ReActAgent(
    model="openai/qwen-plus",
    api_key="sk-xxx",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 百度千帆
agent = ReActAgent(
    model="openai/ernie-4.0",
    api_key="bce-v3/your-access-key/your-secret-key",
    api_base="https://qianfan.baidubce.com/v2",
)

# 本地部署（vLLM / Ollama / LMStudio）
agent = ReActAgent(
    model="openai/llama-3",
    api_key="not-needed",
    api_base="http://localhost:8000/v1",
)
```

## 协议类型

框架支持两种 LLM 调用协议：

### Completions API（默认）

标准的 Chat Completions 协议，兼容性最好：

```python
agent = ReActAgent(
    model="openai/gpt-4",
    api_key="sk-xxx",
    api_type="completions",  # 默认值，可省略
)
```

底层调用 `litellm.completion()`，支持：
- 流式输出（`stream=True`）
- Function Calling（tool_calls）
- Usage 统计（`stream_options={"include_usage": True}`）

### Responses API

OpenAI 新版 Responses 协议：

```python
agent = ReActAgent(
    model="openai/gpt-4",
    api_key="sk-xxx",
    api_type="responses",
)
```

底层调用 `litellm.responses()`，特点：
- 工具格式不同（扁平结构 vs 嵌套结构）
- 支持 `previous_response_id` 上下文关联
- 消息格式转换（`role=tool` → `type=function_call_output`）

## 适配器架构

```
LLMHandler
    │
    ├── CompletionAdapter   (api_type="completions")
    │   └── litellm.completion()
    │
    └── ResponsesAdapter    (api_type="responses")
        └── litellm.responses()
```

两种适配器输出统一的类型：

```python
# 流式文本片段
StreamDelta(content="你好", tool_calls=None, usage=None)

# 完整的工具调用响应
LLMResponse(
    content=None,
    tool_calls=[ToolCallWrapper(...)],
    usage={"prompt_tokens": 100, "completion_tokens": 50}
)
```

## 流式输出

LLM 的流式输出通过 Generator 传递：

```python
for delta in llm_handler.stream(messages, tools):
    if isinstance(delta, StreamDelta):
        if delta.content:
            # 文本片段
            print(delta.content, end="")
        if delta.tool_calls:
            # 工具调用参数逐步到达
            for tc in delta.tool_calls:
                print(f"工具 {tc.index}: name={tc.name}, args+={tc.arguments_delta}")
    elif isinstance(delta, LLMResponse):
        # 完整响应（包含所有 tool_calls）
        for tc in delta.tool_calls:
            print(f"完整调用: {tc.name}({tc.arguments})")
```

## Tool Call 累积器

流式 tool_call 的参数是分片到达的，`ToolCallAccumulator` 负责累积：

```
chunk 1: {index: 0, id: "call_123", function: {name: "get_"}}
chunk 2: {index: 0, function: {name: "weather"}}
chunk 3: {index: 0, function: {arguments: '{"ci'}}
chunk 4: {index: 0, function: {arguments: 'ty": "北京"}'}}
    ↓ 累积后
完整调用: {id: "call_123", function: {name: "get_weather", arguments: '{"city": "北京"}'}}
```

累积器还处理一些供应商的异常行为：
- **index 复用**：某些供应商在同一个 index 上发送多个不同的 tool_call
- **自动拆分**：检测到 ID 变化、参数已完整、名称变化时，自动分配新 slot

## 错误分类

LLM 调用失败时，错误被分类处理：

```python
class LLMError:
    code: str       # 错误码
    message: str    # 错误信息
    retryable: bool # 是否可重试
```

| 错误码 | 说明 | 可重试 |
|--------|------|--------|
| `rate_limit` | 请求频率超限 | ✅ |
| `timeout` | 请求超时 | ✅ |
| `server_error` | 服务端错误 | ✅ |
| `auth_error` | 认证失败 | ❌ |
| `bad_request` | 请求无效 | ❌ |
| `unknown` | 未知错误 | ❌ |

可重试的错误会自动重试（带退避），并产生 `RETRY` 事件。

## 取消 LLM 调用

```python
llm_handler.cancel()
```

取消时：
1. 设置 `_cancelled` 标志位
2. 调用 `adapter.close_stream()` 硬关 HTTP 连接
3. 适配器在下次 yield 前检查标志位，立即退出

## 下一步

- [中间件](./07-middleware.md) — 工具执行的前后钩子
- [取消机制](./09-cancellation.md) — 完整的取消策略
