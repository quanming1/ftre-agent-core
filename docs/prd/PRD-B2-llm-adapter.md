# PRD-B2-llm-adapter

> 从 PRD-TEMPLATE.md 复制。所有内容使用中文。
> 状态生命周期：草稿 → 评审 → approved（定稿）→ 开发中 → 已验收

## 元信息

| 字段 | 值 |
|---|---|
| 阶段 | B2 |
| 名称 | LLM 协议适配层 |
| 状态 | 已验收 |
| 创建日期 | 2026-08-18 |
| 定稿日期 | 2026-08-19 |
| 验收日期 | 2026-08-19 |
| 关联文档 | docs/TODO.yaml 阶段 B2；docs/litellm-migration.md（历史决策）；docs/response-to-ai-base.md（历史架构）；DSH 参考 deepseek-harness packages/llm/llm/src/types.ts（StreamChunk 协议参考来源）；AGENTS.md |

## 1. 背景与目标

- **背景**：混合协议 provider 已经真实出现。OpenCode Go 订阅的 27 个模型里，Muse Spark 1.2 / GPT 5.6 Luna / Grok 4.5 走 OpenAI Responses 协议，其余走 chat/completions——同一 provider 内协议按模型混合（实测 2026-08-18：`muse-spark-1.2` 双协议均 200，但 `reasoning_effort` 仅 responses 路径生效，chat/completions 下被网关静默吞掉）。现状 `LLMHandler` 是单类双 if 分支（completion.py 764 行），协议边界不显式，新协议接入必然继续膨胀该文件。
  - 历史决策考古：本项目曾存在 adapter 架构（`handler/llm/{base,completion,responses,handler}.py` + `register_adapter` 注册表），litellm 迁移时因统一入口让适配层冗余而被移除（docs/litellm-migration.md「已移除的 API」）。此后 litellm 又被换回 openai SDK 直连，移除前提已失效。
- **目标**：建立 LLM 协议适配层——`LLMAdapter` 契约 + 协议注册表 + **DSH StreamChunk 流协议**（协议选择是数据（`api_type` 字符串）而非代码分支；新协议接入 = 新增一个适配器文件 + 注册表一行，消费方（react_runner / compact_manager / title_gen）零协议感知）。
- **非目标**：
  - 不搬 DSH 的 LlmRuntime 运行时路由表（ftre 的 provider 路由由 config.json 承担）
  - 不搬事件瀑布（FtreCoreHookManager 已有等价挂点）
  - 不留 LLMEvent→StreamChunk 翻译骑墙层——一步到位搬彻底，旧事件家族随本阶段移除
  - 不引入新依赖

### 1.1 为什么采用 DSH StreamChunk 协议（而非保留 LLMEvent）

- **与 Msg ContentBlock 同构**：`block-end` 携带的完整 block（text/reasoning/tool-call）与 `ftre_agent_core/message` 的 `TextBlock/ReasoningBlock/ToolCallBlock` 一一对应，react_runner 无需再手工把 ToolCall 事件拼回 `ToolCallBlock`。
- **块级渲染就绪**：ftre-desktop 将来做多块交错显示（文本 + thinking + 工具块）时，StreamChunk 的 block 语义是现成的。
- **error finish 通道**：`finish {kind: 'error', failure}` 是 ftre 现有 `StepFinish` 没有的显式错误通道，能接住畸形 provider 响应（如 Muse chat/completions 实测的 `finish_reason: null`）。
- **对齐 DSH 适配器生态**：与 deepseek-harness 的适配器契约一致，未来可参考其适配器实现（llm-deepseek / llm-pi-ai 双范式）。

## 2. 需求范围

### 2.1 功能需求

- [ ] FR1：`LLMAdapter` 契约（`src/ftre_agent_core/llm/base.py`）——抽象基类，`stream(messages, tools) -> AsyncGenerator[StreamChunk]` + `cancel()` 两个抽象方法；`OpenAIAdapterBase` 提供共享骨架（AsyncOpenAI 客户端构造、cancel 机制、LLM 日志生命周期、异常统一 `LLMError.classify` 包裹并转终止性 error/aborted finish chunk）。
- [ ] FR2：协议注册表（`src/ftre_agent_core/llm/registry.py`）——`PROTOCOLS: dict[str, type[LLMAdapter]]` 映射 `api_type` 字符串到适配器类；`create_llm_handler(api_type, **kwargs) -> LLMAdapter` 工厂函数；`supported_protocols() -> list[str]`；未知 `api_type` 抛 `LLMError(code="INVALID_API_TYPE")` 并在消息中列出全部支持协议。
- [ ] FR3：completions 适配器（`adapters/openai_completions.py`）——迁移现有 `_stream_completions` 路径的 provider 调用与 wire 解析逻辑，输出改写为 StreamChunk：params 组装（`reasoning_effort` 透传 + deepseek thinking 特判）、工具调用增量聚合转 `tool-call-delta` 序列 + `block-end`、finish_reason 映射到 DSH 词汇（stop / tool-calls / max-tokens / error）、usage 口径映射（`usage` chunk 在 `finish` 前）。
- [ ] FR4：responses 适配器（`adapters/openai_responses.py`）——迁移现有 `_stream_responses` 路径：消息/工具 schema 转换（`_convert_messages_to_responses_input` / `_convert_tools_to_responses`）、流事件解析（TextDelta/ReasoningDelta/FunctionCallArgumentsDelta/OutputItemAdded/Done/Completed）转 StreamChunk、`reasoning: {effort}` 透传。
- [ ] FR5：StreamChunk 协议定义 + 组装器——`events.py` 定义七种 chunk（block-start / text-delta / reasoning-delta / tool-call-delta / block-end / usage / finish，dataclass 形态；finish 携带 FinishReason：stop / tool-calls / max-tokens / error / aborted，error/aborted 带 failure {message, code}）；`block_assembler.py` 实现 BlockAssembler——按 index 组装交错 delta 为完整 block，校验配对完整（block-start 必有 block-end、index 单调、finish 收尾后无内容）。
- [ ] FR6：目录结构落位——`llm/` 拆分为 `events.py`（StreamChunk 定义）、`block_assembler.py`（组装器）、`errors.py`（LLMError）、`base.py`（契约）、`registry.py`（注册表）、`adapters/`（一协议一文件）、`wire/normalize.py`（协议共用的消息归一化 + usage 映射）；`completion.py` 删除，**LLMEvent 家族（TextDelta/ReasoningDelta/ToolInputDelta/ToolCall/StepFinish）随之移除**，不留兼容别名或 re-export 尾巴。
- [ ] FR7：消费方迁移（一步到位，无骑墙层）——
  - `react_runner.py`：事件消费循环重写为 StreamChunk + BlockAssembler；block-end 的 ToolCallBlock 直接写入 Msg（消除手工拼接）；StepFinish 消费点改为 finish chunk（含 error/aborted 分支处理）。
  - `compact_manager.py`（ftre 仓库）：TextDelta 收集改 text-delta chunk。
  - `title_gen.py`（ftre 仓库）：同上。
  - `llm/__init__.py` 公共导出面更新为 StreamChunk 家族 + create_llm_handler + LLMError。
- [ ] FR8：`fake_llm.py` 对齐——测试替身改产 StreamChunk 序列（含 block 配对），供 runner 测试与契约测试复用。

### 2.2 非功能需求

- 性能：工厂在 Handler 构造期解析一次，请求路径无额外间接层；BlockAssembler 为 O(1) 每 chunk 的 map 操作。
- 兼容性：ReActAgent 的 `api_type` 构造参数语义不变；config.json 的 `api_type` 字段（model 条目级）由 ftre 仓库 A1 变更承接。
- 可测性：BlockAssembler 与适配器可独立测试；fake_llm 可注入任意（含畸形）chunk 序列做契约测试。

## 3. 技术方案

### 3.1 目录结构（目标态）

```
ftre-agent-core/src/ftre_agent_core/llm/
├── __init__.py                    # 包门面：StreamChunk 家族 + BlockAssembler + 工厂 re-export（唯一公共入口）
├── events.py                      # StreamChunk 七种 chunk + FinishReason（DSH 协议的 Python/dataclass 形态）
├── block_assembler.py             # BlockAssembler：交错 delta → 完整 block；配对校验
├── errors.py                      # LLMError + classify()
├── base.py                        # LLMAdapter(ABC) + OpenAIAdapterBase（共享骨架）
├── registry.py                    # PROTOCOLS 注册表 + create_llm_handler 工厂 + supported_protocols()
├── adapters/
│   ├── __init__.py
│   ├── openai_completions.py      # chat/completions 适配器（→ StreamChunk）
│   └── openai_responses.py        # responses 适配器（→ StreamChunk）
├── wire/
│   ├── __init__.py
│   └── normalize.py               # _normalize_chat_messages / normalize_usage（两协议共用）
└── utils.py                       # LLMLogger（原位不动）
```

### 3.2 关键数据结构

```python
# events.py —— DSH StreamChunk 的 Python 形态
@dataclass
class BlockStart:           index: int; block_type: str       # "text" | "reasoning" | "tool-call"
@dataclass
class TextDeltaChunk:       index: int; text: str
@dataclass
class ReasoningDeltaChunk:  index: int; text: str
@dataclass
class ToolCallDeltaChunk:   index: int; call_id: str; name: str; arguments_delta: str
@dataclass
class BlockEnd:             index: int; block: ContentBlock   # TextBlock / ReasoningBlock / ToolCallBlock
@dataclass
class UsageChunk:           usage: TokenUsage
@dataclass
class FinishChunk:          reason: FinishReason               # stop / tool-calls / max-tokens / error / aborted

StreamChunk = BlockStart | TextDeltaChunk | ReasoningDeltaChunk | ToolCallDeltaChunk | BlockEnd | UsageChunk | FinishChunk

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

### 3.3 StreamChunk 协议契约（适配器必须遵守，BlockAssembler 校验）

- 每个 block-start 必须有配对的 block-end（同 index），index 从 0 单调递增
- delta 只作用于已 start、未 end 的 block
- usage 必须在 finish 之前；finish 必须是最后一个 chunk，其后无内容
- 适配器可抛异常，但 OpenAIAdapterBase 统一将其转为终止性 finish {kind: error}（调用方 abort 转为 aborted）——消费方永不面对裸异常

### 3.4 新协议接入 SOP（本阶段的交付物之一，写入代码注释与 PRD）

1. `adapters/<protocol>.py` 新建，继承 LLMAdapter（openai SDK 系可继承 OpenAIAdapterBase）
2. 实现 `stream()`：messages+tools → 协议请求 → 流事件 → 翻译成 StreamChunk（遵守 3.3 契约）；错误走 LLMError 稳定码
3. `registry.py` 的 PROTOCOLS 加一行
4. config.json 给目标模型加 `"api_type": "<protocol>"`
5. 消费方零改动（BlockAssembler 与 react_runner 消费的是协议，不是具体适配器）

### 3.5 依赖选型

无新依赖。沿用 openai SDK 的 AsyncOpenAI 客户端（completions 走 `client.chat.completions.create`，responses 走 `client.responses.create`）。

## 4. 接口定义

```python
# 公共 API（llm/__init__.py 导出，消费方唯一入口）
class LLMAdapter(ABC):
    async def stream(self, messages: list[dict], tools: list[dict] | None = None) -> AsyncGenerator[StreamChunk, None]: ...
    def cancel(self) -> None: ...

def create_llm_handler(
    api_type: str = "completions",
    *, model: str, api_key: str, api_base: str | None = None,
    timeout: float = 120.0, max_retries: int = 3,
    max_tokens: int | None = None, temperature: float | None = None,
    reasoning_effort: str = "",
) -> LLMAdapter: ...

def supported_protocols() -> list[str]: ...

class BlockAssembler:
    def feed(self, chunk: StreamChunk) -> None: ...
    def blocks(self) -> list[ContentBlock]: ...      # finish 后的完整 block 序列
    def validate(self) -> None: ...                  # 配对/顺序校验，违规抛 LlmError
```

构造参数签名与现有 `LLMHandler.__init__` 完全一致（除首位 `api_type` 从位置参数变为工厂的第一参数）。

## 5. 验收标准

- [x] AC1：StreamChunk 协议契约测试通过——合法序列（配对完整、usage→finish 顺序、index 单调）被 BlockAssembler 正确组装；畸形序列（缺 block-end、finish 后有余 chunk、index 跳跃）被 validate() 拒绝。
- [x] AC2：ftre 全仓测试套通过（消费方迁移后无回归——react_runner / compact_manager / title_gen 的测试断言更新为新协议语义）。
- [x] AC3：`create_llm_handler("banana")` 抛 LLMError，code 为 INVALID_API_TYPE，message 列出 `['completions', 'responses']`。
- [x] AC4：适配器单元测试——fake openai 流事件分别喂两个适配器，断言产出的 chunk 序列符合 3.3 契约且 block 内容正确（文本聚合、reasoning 聚合、tool-call arguments 聚合 + call_id/name 传递）。
- [x] AC5：completion.py 与 LLMEvent 家族不存在；全仓 grep `TextDelta|ReasoningDelta|ToolInputDelta|StepFinish|LLMHandler` 无代码引用（历史文档 docs/litellm-migration.md、docs/response-to-ai-base.md 及 src/tests/ 历史存档目录除外）。
- [x] AC6：真实对话回归——completions 路径（deepseek-v4-flash，带工具调用）与 responses 路径（gpt-5.6-luna，完整对话 + usage 含 reasoning_tokens + finish 映射）各跑一轮真实调用成功；`finish_reason: null` 类畸形响应映射为 error finish（无产出时）或宽容 stop（有产出时）且不崩溃。（备注：muse-spark-1.2 在验收时被 OpenCode 网关整体下线——两协议均 401 Model not supported，与代码无关；responses 协议由 Luna 完成验证。）

## 6. 测试计划

- 单元测试：
  - `tests/test_stream_chunk_contract.py`（新增）：BlockAssembler 组装 + 畸形序列拒绝（AC1）
  - `tests/test_registry.py`（新增）：工厂分发 / INVALID_API_TYPE（AC3）
  - `tests/test_adapters_chunk.py`（新增）：两适配器的 fake 流 → chunk 序列断言（AC4）
  - `tests/test_execute_reasoning.py` / `test_react_runner_*.py`：断言更新为 StreamChunk 语义（fake_llm 产新协议）
- 手动验证：AC6 双协议真实对话
- 回归基线：开发前先跑全量测试记录通过清单，迁移后逐文件比对（断言语义变化但覆盖点不减）

## 7. 变更记录

| 日期 | 变更内容 | 理由 |
|---|---|---|
| 2026-08-19 | 评审前修订：适配器输出协议由「保留 LLMEvent」改为「DSH StreamChunk」（七种 chunk + BlockAssembler + error/aborted finish 通道）；FR5/FR6/FR7/FR8 与 AC1/AC2/AC4/AC5 相应重写；目录结构新增 block_assembler.py；非目标删除「不改 LLMEvent」、新增「不留骑墙层」 | 用户评审决策：采用 DSH 协议（与 Msg ContentBlock 同构、desktop 块渲染就绪、error finish 通道、对齐 DSH 适配器生态） |
| 2026-08-19 | 状态 草稿 → approved（定稿） | 用户评审通过 |
| 2026-08-19 | FR 勾选 + AC 全部验收通过（AC1-AC6）；状态 → 已验收。AC6 备注：muse-spark-1.2 验收时被网关下线（401 Model not supported，与代码无关），responses 协议由 gpt-5.6-luna 完成真实回归；completions 协议由 deepseek-v4-flash 完成（工具调用 + usage 全通） | 开发完成，验收记录留痕 |
