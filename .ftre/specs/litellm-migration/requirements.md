# LiteLLM Migration

> **目标：** 将 LLM 调用层从「OpenAI SDK + 自研协议适配器」全面切换为 LiteLLM

## 简介

当前项目使用自研的协议适配器架构（`llm/` 模块）来支持不同 LLM API 协议。这套架构包括：
- `BaseProtocolAdapter` 抽象基类
- `ResponsesAdapter` 实现 OpenAI Responses API 转换
- `create_client()` 工厂函数
- 一系列模拟 OpenAI SDK 的数据类（`FakeResponse`、`FakeChunk` 等）

LiteLLM 是一个成熟的开源库，已经内置了 100+ LLM 后端的协议适配，可以完全替代自研的适配器架构，减少维护成本。

## 术语表

- **LiteLLM**: Python 库，提供统一的 `completion()` 接口调用各种 LLM API
- **provider/model 格式**: LiteLLM 的模型命名规范，如 `openai/gpt-4`、`anthropic/claude-3`
- **软取消**: 通过标志位 + 生成器内检查实现的取消机制（对比原来的硬取消：强关 HTTP 连接）

## 需求

### 需求 1：删除自研适配器架构

**用户故事：** 作为开发者，我希望移除不再需要的自研代码，以便减少维护负担。

#### 验收标准
1. `llm/` 目录被完全删除（包括 `adapters/`、`base.py`、`types.py`、`registry.py`、`__init__.py`）
2. 所有对 `llm/` 模块的 import 被移除或替换
3. `pyproject.toml` 中的 `openai` 依赖被替换为 `litellm`

### 需求 2：Agent 构造参数变更

**用户故事：** 作为使用者，我希望直接传 model/api_key/api_base 给 Agent，而不是先创建 client 再传入。

#### 验收标准
1. `Agent` 和 `ReActAgent` 的构造参数从 `client: OpenAI` 变更为 `model: str, api_key: str, api_base: str = None`
2. WHEN 创建 Agent 时传入 `model="openai/qwen-plus", api_key="sk-xxx", api_base="https://..."` 
   THEN Agent 能正常调用 LLM

### 需求 3：LLMHandler 使用 LiteLLM

**用户故事：** 作为开发者，我希望 LLMHandler 内部使用 LiteLLM 调用 LLM，以便支持更多后端。

#### 验收标准
1. `LLMHandler` 内部调用 `litellm.completion()` 替代 `client.chat.completions.create()`
2. 流式调用正常工作：`litellm.completion(..., stream=True)` 返回的生成器被正确消费
3. tool_calls 流式拼接逻辑（`ToolCallAccumulator`）继续正常工作
4. token usage 统计继续正常工作（LiteLLM 返回的 usage 格式兼容）

### 需求 4：取消机制变更为软取消

**用户故事：** 作为使用者，我希望能取消正在进行的 LLM 调用。

#### 验收标准
1. `LLMHandler.cancel()` 设置取消标志位
2. `LLMHandler.stream()` 生成器在每次 yield 前检查标志位，检测到取消时退出循环
3. WHEN 调用 `cancel()` THEN 生成器在下一个 chunk 到达时停止（接受延迟）

### 需求 5：异常处理适配

**用户故事：** 作为开发者，我希望 LLM 调用错误被正确分类和处理。

#### 验收标准
1. `LLMError.classify()` 适配 LiteLLM 的异常类型
2. 网络错误、超时、频率限制、内容审核等场景仍能被正确识别

## 边界情况

- **取消延迟**: 软取消只能在下一个 chunk 到达时生效，如果 LLM 响应慢，取消会有延迟。已确认可以接受。
- **LiteLLM 不支持的模型**: 如果传入 LiteLLM 不认识的 provider/model 格式，会抛出异常，由 `LLMError.classify()` 捕获。
- **api_base 为空**: 当 `api_base=None` 时，LiteLLM 使用该 provider 的默认端点（如 OpenAI 官方 API）。
