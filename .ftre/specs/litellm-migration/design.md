# 技术设计：LiteLLM Migration

> **架构概要：** 删除整个 `llm/` 模块，用 LiteLLM 的 `litellm.completion()` 替代自研适配器。Agent 构造参数从 `client` 改为 `model/api_key/api_base`，LLMHandler 内部直接调用 LiteLLM。取消机制从硬取消（关 socket）改为软取消（标志位检查）。

## 涉及文件

| 操作 | 文件路径 | 职责 |
|------|----------|------|
| 删除 | `src/ftre_agent_core/llm/` | 整个目录删除（adapters/、base.py、types.py、registry.py、__init__.py） |
| 修改 | `src/ftre_agent_core/agent/base.py` | 构造参数 client → model/api_key/api_base |
| 修改 | `src/ftre_agent_core/agent/react.py` | 构造参数 client → model/api_key/api_base |
| 修改 | `src/ftre_agent_core/agent/runner/react_runner.py` | LLMHandler 实例化参数变更 |
| 修改 | `src/ftre_agent_core/agent/runner/handler/llm_handler.py` | 使用 litellm.completion()，软取消，异常适配 |
| 修改 | `src/ftre_agent_core/agent/__init__.py` | 删除 llm 相关导出（如有） |
| 修改 | `pyproject.toml` | 依赖 openai → litellm |
| 修改 | `src/tests/test_agent_integration.py` | 删除 create_client，直接传参 |

## 现有代码意图分析

### `llm/` 模块

**当前意图**：提供协议适配层，让上层代码（LLMHandler）不感知底层 API 协议差异。`create_client()` 根据 `api_type` 返回 OpenAI SDK 或自研适配器，两者都暴露 `client.chat.completions.create()` 鸭子接口。

**隐式约束**：
- 适配器返回的对象必须有 `chat.completions.create()` 方法
- 流式响应的 chunk 格式必须兼容 OpenAI SDK（有 `choices[0].delta`）
- 适配器需要实现 `cancel_stream()` 方法供取消使用

**为什么删除是安全的**：LiteLLM 已内置 100+ 后端适配，返回格式与 OpenAI SDK 兼容，不需要自研适配层。

### `LLMHandler`

**当前意图**：封装 LLM 流式调用，对上层屏蔽 chunk 拼接、tool_calls 累积细节。

**隐式约束**：
- `stream()` 返回 Generator，yield `StreamDelta` 或 `LLMResponse`
- `ToolCallAccumulator` 处理流式 tool_call 拼接，包括检测 index 复用异常
- `cancel()` 需要能中断正在进行的流式读取

**改动影响**：
- 调用方式变更，但输出格式（`StreamDelta`/`LLMResponse`）保持不变
- 取消机制变为软取消，延迟可接受

### `Agent` / `ReActAgent`

**当前意图**：`client` 作为依赖注入，Agent 不关心 client 如何创建。

**改动影响**：从注入 client 改为注入配置（model/api_key/api_base），LLMHandler 内部创建调用。上层使用方式简化。

## 架构决策

### 决策 1：LiteLLM 调用方式

选择 `litellm.completion()` 函数式调用，而非 `litellm.Router` 或其他封装。

理由：
- 简单直接，一次调用传入所有参数
- 不需要维护全局状态或配置
- 与现有代码改动最小

### 决策 2：软取消实现

选择「标志位 + 生成器内检查」，而非尝试强关 LiteLLM 内部连接。

理由：
- LiteLLM 不暴露底层 httpx response，无法强关
- 软取消实现简单，延迟可接受
- 生成器内每次 yield 前检查，取消信号在下一个 chunk 到达时生效

### 决策 3：异常映射

LiteLLM 抛出的异常类型与 OpenAI SDK 不同，需要重新映射：

| LiteLLM 异常 | 映射到 code |
|--------------|-------------|
| `litellm.RateLimitError` | `rate_limit` |
| `litellm.Timeout` | `timeout` |
| `litellm.APIConnectionError` | `network` |
| `litellm.ContentPolicyViolationError` | `content_filter` |
| `litellm.APIError` | `api_error` |
| 其他 `Exception` | `unknown` |

## 接口设计

### Agent 构造参数

```python
class Agent(ABC):
    def __init__(
        self,
        model: str,                    # LiteLLM 格式，如 "openai/gpt-4"
        api_key: str,                  # API 密钥
        api_base: str | None = None,   # 自定义端点（可选）
        system_prompt: str = "你是一个有帮助的助手。",
        tools: list[Tool] = None,
        memory: MemoryProtocol | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        # ... 其余不变
```

### LLMHandler

```python
class LLMHandler:
    def __init__(self, model: str, api_key: str, api_base: str | None = None):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self._cancelled = False  # 取消标志位

    def cancel(self) -> None:
        """设置取消标志位（软取消）"""
        self._cancelled = True

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None
    ) -> Generator[StreamDelta | LLMResponse, None, None]:
        """流式调用 LLM（使用 LiteLLM）"""
        self._cancelled = False  # 重置标志位
        
        response = litellm.completion(
            model=self.model,
            messages=messages,
            tools=tools,
            api_key=self.api_key,
            api_base=self.api_base,
            stream=True,
            stream_options={"include_usage": True},
        )
        
        # ... 消费流式响应，每次 yield 前检查 self._cancelled
```

### LLMError.classify() 适配

```python
import litellm

@staticmethod
def classify(e: Exception) -> "LLMError":
    """根据异常类型分类错误（适配 LiteLLM）"""
    if isinstance(e, litellm.RateLimitError):
        return LLMError(message=f"请求频率超限: {e}", code="rate_limit")
    if isinstance(e, litellm.Timeout):
        return LLMError(message=f"请求超时: {e}", code="timeout")
    if isinstance(e, litellm.APIConnectionError):
        return LLMError(message=f"网络连接失败: {e}", code="network")
    if isinstance(e, litellm.ContentPolicyViolationError):
        return LLMError(message=f"内容审核未通过: {e}", code="content_filter")
    if isinstance(e, litellm.APIError):
        return LLMError(message=f"API 错误: {e}", code="api_error")
    return LLMError(message=f"未知错误: {e}", code="unknown")
```

## 与现有逻辑的关系

```
ReActAgent
    │
    ├── model / api_key / api_base (新增属性)
    │
    └── ReActRunner
            │
            └── LLMHandler(model, api_key, api_base)
                    │
                    ├── litellm.completion()  ← 替代 client.chat.completions.create()
                    │
                    ├── ToolCallAccumulator   ← 保持不变
                    │
                    └── StreamDelta / LLMResponse  ← 保持不变
```

数据流：
1. `ReActRunner._step()` 调用 `self.llm.stream(messages, tools)`
2. `LLMHandler.stream()` 调用 `litellm.completion(..., stream=True)`
3. LiteLLM 返回流式生成器，chunk 格式与 OpenAI SDK 兼容
4. `ToolCallAccumulator` 累积 tool_calls（逻辑不变）
5. yield `StreamDelta` 或 `LLMResponse`（格式不变）
6. `ReActRunner` 处理事件（逻辑不变）
