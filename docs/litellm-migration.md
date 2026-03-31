# LiteLLM 迁移指南

本文档说明 ftre-agent-core 从 OpenAI SDK 迁移到 LiteLLM 的变更内容。

## 概述

我们将 LLM 调用层从「OpenAI SDK + 自研协议适配器」全面切换为 [LiteLLM](https://github.com/BerriAI/litellm)。LiteLLM 是一个成熟的开源库，内置 100+ LLM 后端的协议适配，简化了多模型支持。

## Breaking Changes

### Agent 构造参数变更

**Before:**
```python
from ftre_agent_core.llm import create_client
from ftre_agent_core.agent import ReActAgent

client = create_client(api_key="sk-xxx", base_url="https://...")
agent = ReActAgent(
    client=client,
    model="qwen-plus",
    system_prompt="你是一个助手",
    tools=[...],
)
```

**After:**
```python
from ftre_agent_core.agent import ReActAgent

agent = ReActAgent(
    model="openai/qwen-plus",          # LiteLLM 格式: provider/model
    api_key="sk-xxx",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    system_prompt="你是一个助手",
    tools=[...],
)
```

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | `str` | LiteLLM 模型格式，如 `openai/gpt-4`、`anthropic/claude-3-opus`、`openai/qwen-plus` |
| `api_key` | `str` | API 密钥 |
| `api_base` | `str \| None` | 自定义 API 端点（可选），用于兼容 OpenAI 协议的第三方服务 |
| `api_type` | `str` | 协议类型，`"completions"`（默认）或 `"responses"` |

### 模型命名格式

LiteLLM 使用 `provider/model` 格式指定模型：

```python
# OpenAI
model="openai/gpt-4"
model="openai/gpt-3.5-turbo"

# Anthropic
model="anthropic/claude-3-opus-20240229"
model="anthropic/claude-3-sonnet-20240229"

# Azure OpenAI
model="azure/my-deployment-name"

# 阿里云 DashScope（通过 OpenAI 兼容协议）
model="openai/qwen-plus"
api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"

# 本地 Ollama
model="ollama/llama2"
api_base="http://localhost:11434"
```

完整支持列表见 [LiteLLM 文档](https://docs.litellm.ai/docs/providers)。

### Responses API 厂商

某些厂商使用 Responses 协议而非 Completions 协议：

```python
agent = ReActAgent(
    model="openai/gpt-5.4",
    api_key="sk-xxx",
    api_base="http://ai.xiamai.top/v1",
    api_type="responses",  # 指定 Responses 协议
    system_prompt="你是一个助手",
    tools=[...],
)
```

典型的 Responses 协议厂商配置：
```json
{
  "xiamai": {
    "api_key": "sk-xxx",
    "base_url": "http://ai.xiamai.top/v1",
    "api_type": "responses"
  }
}
```

## 已移除的 API

以下 API 已被移除，不再可用：

```python
# 已移除
from ftre_agent_core.llm import create_client
from ftre_agent_core.llm import register_adapter
from ftre_agent_core.llm import ADAPTER_REGISTRY
```

## 依赖变更

`pyproject.toml` 依赖从 `openai` 改为 `litellm`：

```toml
# Before
dependencies = [
    "openai>=1.0.0",
    "httpx",
]

# After
dependencies = [
    "litellm>=1.0.0",
]
```

安装新依赖：

```bash
pip install -e .
# 或
pip install litellm
```

## 内部变更（对使用者透明）

以下变更不影响外部 API，但记录在此供参考：

1. **删除 `llm/` 模块**：整个协议适配器架构被移除，包括 `BaseProtocolAdapter`、`ResponsesAdapter`、`FakeResponse` 等
2. **LLMHandler 重构**：内部调用从 `client.chat.completions.create()` 改为 `litellm.completion()`
3. **取消机制变更**：从硬取消（强关 HTTP 连接）改为软取消（标志位检查），取消信号在下一个 chunk 到达时生效

## 迁移检查清单

- [ ] 更新 Agent 构造方式，移除 `client` 参数
- [ ] 添加 `model`、`api_key`、`api_base` 参数
- [ ] 将模型名改为 LiteLLM 格式（`provider/model`）
- [ ] 移除对 `ftre_agent_core.llm` 的所有 import
- [ ] 运行 `pip install litellm` 安装新依赖
