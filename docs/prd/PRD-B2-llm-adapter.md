# PRD-B2-llm-adapter

> 从 PRD-TEMPLATE.md 复制。所有内容使用中文。
> 状态生命周期：草稿 → 评审 → approved（定稿）→ 开发中 → 已验收

## 元信息

| 字段 | 值 |
|---|---|
| 阶段 | B2 |
| 名称 | LLM 协议适配层 |
| 状态 | 草稿 |
| 创建日期 | 2026-08-18 |
| 定稿日期 | （approved 时填写） |
| 验收日期 | （已验收时填写） |
| 关联文档 | docs/TODO.yaml 阶段 B2；docs/litellm-migration.md（历史决策）；docs/response-to-ai-base.md（历史架构）；AGENTS.md |

## 1. 背景与目标

- **背景**：混合协议 provider 已经真实出现。OpenCode Go 订阅的 27 个模型里，Muse Spark 1.2 / GPT 5.6 Luna / Grok 4.5 走 OpenAI Responses 协议，其余走 chat/completions——同一 provider 内协议按模型混合（实测 2026-08-18：`muse-spark-1.2` 双协议均 200，但 `reasoning_effort` 仅 responses 路径生效，chat/completions 下被网关静默吞掉）。现状 `LLMHandler` 是单类双 if 分支（completion.py 764 行），协议边界不显式，新协议接入必然继续膨胀该文件。
  - 历史决策考古：本项目曾存在 adapter 架构（`handler/llm/{base,completion,responses,handler}.py` + `register_adapter` 注册表），litellm 迁移时因统一入口让适配层冗余而被移除（docs/litellm-migration.md「已移除的 API」）。此后 litellm 又被换回 openai SDK 直连，移除前提已失效。
- **目标**：建立 LLM 协议适配层——`LLMAdapter` 契约 + 协议注册表，协议选择是数据（`api_type` 字符串）而非代码分支；新协议接入 = 新增一个适配器文件 + 注册表一行，消费方（react_runner / compact_manager / title_gen）零改动。
- **非目标**：
  - 不搬 DSH 的 LlmRuntime 运行时路由表（ftre 的 provider 路由由 config.json 承担）
  - 不搬事件瀑布（FtreCoreHookManager 已有等价挂点）
  - 不改 `LLMEvent` 事件家族（TextDelta/ReasoningDelta/ToolInputDelta/ToolCall/StepFinish 已是等价的 provider 中立契约）
  - 不引入新依赖

## 2. 需求范围

### 2.1 功能需求

- [ ] FR1：`LLMAdapter` 契约（`src/ftre_agent_core/llm/base.py`）——抽象基类，`stream(messages, tools) -> AsyncGenerator[LLMEvent]` + `cancel()` 两个抽象方法；`OpenAIAdapterBase` 提供共享骨架（AsyncOpenAI 客户端构造、cancel 机制、LLM 日志生命周期、异常统一 `LLMError.classify` 包裹）。
- [ ] FR2：协议注册表（`src/ftre_agent_core/llm/registry.py`）——`PROTOCOLS: dict[str, type[LLMAdapter]]` 映射 `api_type` 字符串到适配器类；`create_llm_handler(api_type, **kwargs) -> LLMAdapter` 工厂函数；`supported_protocols() -> list[str]`；未知 `api_type` 抛 `LLMError(code="INVALID_API_TYPE")` 并在消息中列出全部支持协议。
- [ ] FR3：completions 适配器（`adapters/openai_completions.py`）——迁移现有 `_stream_completions` 路径全部逻辑：params 组装（`reasoning_effort` 透传 + deepseek thinking 特判）、`_ToolCallAccumulator` 增量聚合、finish_reason 映射、usage 口径映射。**行为与现状逐字节等价**（纯搬迁，不改逻辑）。
- [ ] FR4：responses 适配器（`adapters/openai_responses.py`）——迁移现有 `_stream_responses` 路径全部逻辑：消息/工具 schema 转换（`_convert_messages_to_responses_input` / `_convert_tools_to_responses`）、流事件解析（TextDelta/ReasoningDelta/FunctionCallArgumentsDelta/OutputItemAdded/Done/Completed）、`reasoning: {effort}` 透传。行为与现状逐字节等价。
- [ ] FR5：目录结构落位——`llm/` 拆分为 `events.py`（事件类型）、`errors.py`（LLMError）、`base.py`（契约）、`registry.py`（注册表）、`adapters/`（一协议一文件）、`wire/normalize.py`（协议共用的消息归一化 + usage 映射）；`completion.py` 删除，不留兼容别名或 re-export 尾巴。
- [ ] FR6：消费方切换——`react_runner.py`、`compact_manager.py`（ftre 仓库）、`title_gen.py`（ftre 仓库）三处 `LLMHandler(...)` 改为 `create_llm_handler(api_type, ...)`；`llm/__init__.py` 公共导出面更新（`LLMAdapter`、`create_llm_handler`、`LLMError`、事件五件套），`LLMHandler` 类名移除。
- [ ] FR7：`fake_llm.py` 对齐——测试替身改为实现 `LLMAdapter` 契约（duck typing `stream()`/`cancel()` 不受影响则零改动，引用具体类名处同步更新）。

### 2.2 非功能需求

- 性能：零运行时开销差异——工厂在 Handler 构造期解析一次，请求路径无额外间接层。
- 兼容性：ReActAgent 的 `api_type` 构造参数、`LLMHandler` 全部构造参数语义不变；`config.json` 的 `api_type` 字段（model 条目级）由 ftre 仓库 A1 变更承接（见 PRD-A1 变更记录）。
- 可测性：每个适配器可独立构造、独立 mock 流事件序列。

## 3. 技术方案

### 3.1 目录结构（目标态）

```
ftre-agent-core/src/ftre_agent_core/llm/
├── __init__.py                    # 包门面：契约类型 + 工厂 re-export（唯一公共入口）
├── events.py                      # TextDelta / ReasoningDelta / ToolInputDelta / ToolCall / StepFinish / LLMEvent
├── errors.py                      # LLMError + classify()
├── base.py                        # LLMAdapter(ABC) + OpenAIAdapterBase（共享骨架）
├── registry.py                    # PROTOCOLS 注册表 + create_llm_handler 工厂 + supported_protocols()
├── adapters/
│   ├── __init__.py
│   ├── openai_completions.py      # chat/completions 适配器
│   └── openai_responses.py        # responses 适配器
├── wire/
│   ├── __init__.py
│   └── normalize.py               # _normalize_chat_messages / normalize_usage（两协议共用）
└── utils.py                       # LLMLogger（原位不动）
```

### 3.2 关键数据结构

```python
# registry.py
PROTOCOLS: dict[str, type[LLMAdapter]] = {
    "completions": OpenAICompletionsAdapter,
    "responses": OpenAIResponsesAdapter,
}

def create_llm_handler(api_type: str = "completions", **kwargs) -> LLMAdapter:
    adapter_cls = PROTOCOLS.get(api_type)
    if adapter_cls is None:
        raise LLMError(
            f"unknown api_type {api_type!r}; supported: {supported_protocols()}",
            "INVALID_API_TYPE",
        )
    return adapter_cls(**kwargs)
```

### 3.3 新协议接入 SOP（本阶段的交付物之一，写入代码注释与 PRD）

1. `adapters/<protocol>.py` 新建，继承 `LLMAdapter`（openai SDK 系可继承 `OpenAIAdapterBase`）
2. 实现 `stream()`：messages+tools → 协议请求 → 流事件 → 翻译成 LLMEvent；错误走 `LLMError` 稳定码
3. `registry.py` 的 `PROTOCOLS` 加一行
4. config.json 给目标模型加 `"api_type": "<protocol>"`
5. 消费方零改动

### 3.4 依赖选型

无新依赖。沿用 openai SDK 的 `AsyncOpenAI` 客户端（completions 走 `client.chat.completions.create`，responses 走 `client.responses.create`）。

## 4. 接口定义

```python
# 公共 API（llm/__init__.py 导出，消费方唯一入口）
class LLMAdapter(ABC):
    async def stream(self, messages: list[dict], tools: list[dict] | None = None) -> AsyncGenerator[LLMEvent, None]: ...
    def cancel(self) -> None: ...

def create_llm_handler(
    api_type: str = "completions",
    *, model: str, api_key: str, api_base: str | None = None,
    timeout: float = 120.0, max_retries: int = 3,
    max_tokens: int | None = None, temperature: float | None = None,
    reasoning_effort: str = "",
) -> LLMAdapter: ...

def supported_protocols() -> list[str]: ...
```

构造参数签名与现有 `LLMHandler.__init__` 完全一致（除首位 `api_type` 从位置参数变为工厂的第一参数），保证三处消费点只改调用形式不改参数语义。

## 5. 验收标准

- [ ] AC1：`ftre-agent-core` 现有 20 个测试文件全部原样通过（行为保持证明——测试的是契约不是类名；`test_completion_params`、`test_llm_error_classification` 重点回归）。
- [ ] AC2：`ftre` 全仓测试套通过（327 个用例，验证 react_runner/compact_manager/title_gen 三处消费点切换无回归）。
- [ ] AC3：`create_llm_handler("banana")` 抛 `LLMError`，code 为 `INVALID_API_TYPE`，message 列出 `['completions', 'responses']`。
- [ ] AC4：`create_llm_handler("responses", ...)` 返回的实例 `stream()` 产出事件序列与重构前 `_stream_responses` 逐事件一致（用 fake 流事件序列做 A/B 对比测试）。
- [ ] AC5：`completion.py` 文件不存在；全仓 `grep LLMHandler` 无残留引用（含文档，litellm-migration.md 等历史文档除外——它们描述的是历史状态）。
- [ ] AC6：真实对话回归——completions 路径（deepseek-v4-flash）与 responses 路径（muse-spark-1.2，config.json 已配 `api_type: "responses"`）各跑一轮完整 agent 对话，工具调用与 reasoning 输出正常。

## 6. 测试计划

- 单元测试：
  - `tests/test_registry.py`（新增）：工厂分发 / INVALID_API_TYPE / supported_protocols
  - `tests/test_completion_params.py`：适配到新类名后原样通过（params 组装断言不变）
  - `tests/test_llm_error_classification.py`：原样通过
  - AC4 的 A/B 对比测试：同一 fake 事件序列分别喂旧逻辑（重构前快照）与新适配器，断言产出一致
- 手动验证：AC6 双协议真实对话
- 回归基线：重构前先跑全量测试记录通过清单，重构后逐文件比对

## 7. 变更记录

| 日期 | 变更内容 | 理由 |
|---|---|---|
| YYYY-MM-DD | 初始定稿 | — |
