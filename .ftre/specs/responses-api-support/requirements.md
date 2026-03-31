# Responses API Support

> **目标：** 让 ftre-agent-core 支持 `api_type="responses"` 的 LLM 厂商

## 简介

部分 LLM 厂商（如某些 OpenAI 代理）使用 Responses 协议而非标准的 Completions 协议。当前 `LLMHandler` 只支持 `litellm.completion()`，需要增加 `litellm.responses()` 调用方式。

典型配置：
```json
{
  "xiamai": {
    "api_key": "sk-xxx",
    "base_url": "http://ai.xiamai.top/v1",
    "api_type": "responses",
    "models": { "gpt-5.4": "gpt-5.4" }
  }
}
```

## 术语表

- **Completions API**: 传统的 LLM 调用协议，使用 `messages` 列表
- **Responses API**: OpenAI 最新协议，使用 `input` 参数（兼容 messages 格式）
- **api_type**: 区分协议类型的参数，值为 `"completions"` 或 `"responses"`

## 需求

### 需求 1：Agent 构造参数增加 api_type

**用户故事：** 作为使用者，我希望在创建 Agent 时指定 api_type，以便使用 Responses 协议的厂商。

#### 验收标准
1. `Agent` 和 `ReActAgent` 构造函数新增 `api_type: str = "completions"` 参数
2. 默认值 `"completions"` 保证向下兼容，现有代码无需修改
3. WHEN `api_type="responses"` THEN Agent 使用 `litellm.responses()` 调用 LLM

### 需求 2：LLMHandler 支持 Responses 协议

**用户故事：** 作为开发者，我希望 LLMHandler 能根据 api_type 选择正确的调用方式。

#### 验收标准
1. `LLMHandler.__init__()` 新增 `api_type` 参数
2. WHEN `api_type="completions"` THEN 调用 `litellm.completion()`（现有逻辑）
3. WHEN `api_type="responses"` THEN 调用 `litellm.responses(input=messages, ...)`
4. 两种模式的输出格式统一为 `StreamDelta` / `LLMResponse`

### 需求 3：Responses 流式事件适配

**用户故事：** 作为开发者，我希望 Responses 模式的流式输出与 Completions 模式行为一致。

#### 验收标准
1. 流式文本：`event.type == "response.output_text.delta"` 时，yield `StreamDelta(content=event.delta)`
2. 完成事件：`event.type == "response.completed"` 时，处理完整响应
3. 取消机制：软取消（标志位检查）在两种模式下行为一致

### 需求 4：Responses Tool Calling 适配

**用户故事：** 作为使用者，我希望 Responses 模式下的工具调用与 Completions 模式行为一致。

#### 验收标准
1. 检测工具调用：遍历 `response.output` 找 `item.type == "function_call"`
2. 提取参数：`item.name`、`item.arguments`、`item.call_id`
3. 工具结果回传：追加 `{"type": "function_call_output", "call_id": ..., "output": ...}` 到 input
4. 上层 Runner 无感知差异，统一收到 `LLMResponse(tool_calls=[...])`

## 协议差异对照

| 特性 | completion() | responses() |
|------|-------------|-------------|
| 消息参数 | `messages=[...]` | `input=[...]`（兼容 messages 格式） |
| 流式文本 | `chunk.choices[0].delta.content` | `event.delta`（when `event.type == "response.output_text.delta"`） |
| 工具调用检测 | `response.choices[0].message.tool_calls` | `response.output` 遍历找 `item.type == "function_call"` |
| 工具结果回传 | messages 追加 `role: "tool"` | input 追加 `type: "function_call_output"` |

## 边界情况

- **api_type 无效值**：非 `"completions"` / `"responses"` 时，抛出 `ValueError`
- **厂商不支持 responses**：LiteLLM 会自动桥接转换，对上层透明
- **流式事件类型未知**：忽略未识别的 event.type，只处理已知类型
