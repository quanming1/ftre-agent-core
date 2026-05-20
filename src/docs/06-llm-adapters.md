# LLM 适配

通过 LiteLLM 支持多种供应商，提供两种协议适配器。

## 供应商配置

使用 `openai/model-name` 格式 + 自定义 `api_base`：

```python
# OpenAI
ReActAgent(model="openai/gpt-4", api_key="sk-xxx")

# 任何 OpenAI 兼容端点
ReActAgent(model="openai/deepseek-v3", api_key="sk-xxx", api_base="https://your-gateway.com/v1")

# 本地部署
ReActAgent(model="openai/llama-3", api_key="not-needed", api_base="http://localhost:8000/v1")
```

## 协议类型

### Completions API（默认）

```python
ReActAgent(api_type="completions", ...)  # 默认，可省略
```

### Responses API

```python
ReActAgent(api_type="responses", ...)
```

## 流式输出

LLM 输出通过 Generator 逐步传递：

- `StreamDelta(content=...)` → 文本片段
- `StreamDelta(reasoning=...)` → 推理过程（DeepSeek R1）
- `StreamDelta(tool_calls=...)` → 工具调用参数增量
- `LLMResponse(tool_calls=...)` → 完整工具调用（流结束时）

## 错误分类

| 错误码 | 说明 | 
|--------|------|
| `rate_limit` | 频率超限 |
| `timeout` | 超时 |
| `network` | 网络失败 |
| `auth_error` | 认证失败 |
| `bad_request` | 请求无效 |
| `content_filter` | 内容审核 |

## 取消

`LLMHandler.cancel()` 硬关 HTTP 连接，毫秒级中断流式调用。

## 日志

每次 LLM 调用的输入和原始 chunk 输出记录到 `data/logs/llm/{日期}/{时间}.log`。

## 下一步

- [取消机制](./09-cancellation.md)
