# PRD-B2-llm-adapter

> 从 PRD-TEMPLATE.md 复制。所有内容使用中文。
> 状态生命周期：草稿 → 评审 → approved（定稿）→ 开发中 → 已验收

## 元信息

| 字段 | 值 |
|---|---|
| 阶段 | B2 |
| 名称 | LLM 协议适配层 |
| 状态 | 开发中（原始 B2 已验收；Responses 状态重放补充修复中） |
| 创建日期 | 2026-08-18 |
| 定稿日期 | 2026-08-19 |
| 验收日期 | 原始验收：2026-08-19；补充验收：待定 |
| 关联文档 | docs/TODO.yaml 阶段 B2；docs/litellm-migration.md（历史决策）；docs/response-to-ai-base.md（历史架构）；DSH 参考 deepseek-harness packages/llm/llm/src/types.ts（StreamChunk 协议参考来源）；AGENTS.md |

## 1. 背景与目标

- **背景**：混合协议 provider 已经真实出现。OpenCode Go 订阅的 27 个模型里，Muse Spark 1.2 / GPT 5.6 Luna / Grok 4.5 走 OpenAI Responses 协议，其余走 chat/completions——同一 provider 内协议按模型混合（实测 2026-08-18：`muse-spark-1.2` 双协议均 200，但 `reasoning_effort` 仅 responses 路径生效，chat/completions 下被网关静默吞掉）。现状 `LLMHandler` 是单类双 if 分支（completion.py 764 行），协议边界不显式，新协议接入必然继续膨胀该文件。
  - 历史决策考古：本项目曾存在 adapter 架构（`handler/llm/{base,completion,responses,handler}.py` + `register_adapter` 注册表），litellm 迁移时因统一入口让适配层冗余而被移除（docs/litellm-migration.md「已移除的 API」）。此后 litellm 又被换回 openai SDK 直连，移除前提已失效。
- **目标**：建立 LLM 协议适配层——`LLMAdapter` 契约 + 协议注册表 + **DSH StreamChunk 流协议**（协议选择是数据（`api_type` 字符串）而非代码分支；新协议接入 = 新增一个适配器文件 + 注册表一行，消费方（react_runner / compact_manager / title_gen）零协议感知）。
- **非目标**：
  - 不搬 DSH 的 LlmRuntime 运行时路由表（ftre 的 provider 路由由 config.json 承担）
  - 不在 B2 搬运宿主事件瀑布（当前 Hook 统一由 C1 的 HookDispatcher 提供）
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
- [ ] FR4：responses 适配器（`adapters/openai_responses.py`）——迁移现有 `_stream_responses` 路径：消息/工具 schema 转换（`_convert_messages_to_responses_input` / `_convert_tools_to_responses`）、流事件解析（TextDelta/ReasoningDelta/FunctionCallArgumentsDelta/OutputItemAdded/Done/Completed）转 StreamChunk、`reasoning: {effort}` 透传；Responses 标准用量字段 `input_tokens` / `output_tokens` 映射为消费方统一的 `prompt_tokens` / `completion_tokens`。
- [ ] FR5：StreamChunk 协议定义 + 组装器——`events.py` 定义七种 chunk（block-start / text-delta / reasoning-delta / tool-call-delta / block-end / usage / finish，dataclass 形态；finish 携带 FinishReason：stop / tool-calls / max-tokens / error / aborted，error/aborted 带 failure {message, code}）；`block_assembler.py` 实现 BlockAssembler——按 index 组装交错 delta 为完整 block，校验配对完整（block-start 必有 block-end、index 单调、finish 收尾后无内容）。
- [ ] FR6：目录结构落位——`llm/` 拆分为 `events.py`（StreamChunk 定义）、`block_assembler.py`（组装器）、`errors.py`（LLMError）、`base.py`（契约）、`registry.py`（注册表）、`adapters/`（一协议一文件）、`wire/normalize.py`（协议共用的消息归一化 + usage 映射）；`completion.py` 删除，**LLMEvent 家族（TextDelta/ReasoningDelta/ToolInputDelta/ToolCall/StepFinish）随之移除**，不留兼容别名或 re-export 尾巴。
- [ ] FR7：消费方迁移（一步到位，无骑墙层）——
  - `react_runner.py`：事件消费循环重写为 StreamChunk + BlockAssembler；block-end 的 ToolCallBlock 直接写入 Msg（消除手工拼接）；StepFinish 消费点改为 finish chunk（含 error/aborted 分支处理）。
  - `compact_manager.py`（ftre 仓库）：TextDelta 收集改 text-delta chunk。
  - `title_gen.py`（ftre 仓库）：同上。
  - `llm/__init__.py` 公共导出面更新为 StreamChunk 家族 + create_llm_handler + LLMError。
- [ ] FR8：`fake_llm.py` 对齐——测试替身改产 StreamChunk 序列（含 block 配对），供 runner 测试与契约测试复用。
- [ ] **FR9：Responses 历史 Output Item 边界**——Responses 适配器不得把持久化的
  `reasoning_content` 重新伪造成完整的 provider Output Item，也不得在请求 input 中手工补充
  仅由 API 返回时填充的 `status`。适配器必须在 `ResponseOutputItemDoneEvent` 到达时保留原始
  Output Item 的传输字段（包括 `id`、`summary`、`content`、`encrypted_content` 等可用字段），
  通过 `response_metadata` 暴露给宿主；宿主负责持久化和下一轮原样重放。UI 展示用的
  `ThinkingBlock` 与 Responses 传输 Item 必须是两个边界，不能互相替代。
- [ ] **FR10：Responses Vision 输入契约**——Responses 请求中的图片必须使用
  `{"type": "input_image", "image_url": "<url-or-data-url>"}` 或 `file_id`；文本和图片
  同处于 user content 数组时分别使用 `input_text` 与 `input_image`。Chat Completions 的
  `{"type": "image_url", "image_url": {"url": "..."}}` 只能作为适配器入口形态，不能
  原样发送到 Responses。`detail` 省略时使用供应商默认值，不得为了转换强行写入无关字段。

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

### 3.6 Responses 多轮状态重放（补充协议）

Responses 的历史消息不是普通的“把思考文本塞回 assistant content”。一次响应由多个
Output Item 组成，尤其是 reasoning item、function call item 和 message item；下一轮需要
重放的是供应商返回的 Item 边界和传输字段，而不是 UI 为了展示而生成的 `ThinkingBlock`。

规范依据：OpenAI [Responses 迁移指南](https://developers.openai.com/api/docs/guides/migrate-to-responses)
和 [Reasoning models](https://developers.openai.com/api/docs/guides/reasoning)；接入方必须以实际
供应商接受的 input schema 为准，不把返回对象字段默认视为可写入请求的字段。

当前错误链路已经复现：

```text
持久化 ThinkingBlock 文本
    ↓
转换成 reasoning_content
    ↓
Core 人工生成 reasoning input item
    ↓
手工补 status="completed"
    ↓
Console Go 拒绝：Unknown parameter: input[n].status
```

目标链路：

```text
ResponseOutputItemDoneEvent.item
    ↓
Responses 适配器保留原始 id/summary/content/encrypted_content 等字段
    ↓
response_metadata.output_items
    ↓
ftre Host/Session 持久化传输元数据
    ↓
下一轮 Responses 请求按原始 Item 形状重放
```

约束：

- `status` 可以出现在 API 返回的 Output Item 中，但它是返回态字段；Core 不得在手工构造
  的 Responses input 中臆造 `status`，也不得把返回对象未经筛选地当成新的 input。
- 已持久化的旧会话如果只有 `reasoning_content` 文本，没有原始 Output Item，必须走明确的
  安全降级/诊断路径；不得用“伪造 status”掩盖缺失的传输状态。
- UI 的思考展示和 Responses 的传输 Item 分离：前者可以继续使用 ThinkingBlock，后者必须
  使用可重放的 provider metadata。

### 3.7 Responses 图片输入

Responses 适配器对图片使用官方 Item/content 形状。文本和图片混合时，每个 content part
必须带自己的类型：

规范依据：OpenAI [Images and vision](https://developers.openai.com/api/docs/guides/images-vision)。

```json
{
  "role": "user",
  "content": [
    {"type": "input_text", "text": "请描述图片"},
    {
      "type": "input_image",
      "image_url": "data:image/png;base64,...",
      "detail": "auto"
    }
  ]
}
```

`image_url` 可以是公开 URL 或 data URL，也可以使用 `file_id`；`detail` 可省略并使用供应商
默认值。Chat Completions 的 `{"type":"image_url","image_url":{"url":"..."}}`
只是适配器入口形态，不能原样发送给 Responses。

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
- [x] AC4：适配器单元测试——fake openai 流事件分别喂两个适配器，断言产出的 chunk 序列符合 3.3 契约且 block 内容正确（文本聚合、reasoning 聚合、tool-call arguments 聚合 + call_id/name 传递）；Responses 标准 usage 的 `input_tokens` / `output_tokens` 映射为下游统一字段。
- [x] AC5：completion.py 与 LLMEvent 家族不存在；全仓 grep `TextDelta|ReasoningDelta|ToolInputDelta|StepFinish|LLMHandler` 无代码引用（历史文档 docs/litellm-migration.md、docs/response-to-ai-base.md 及 src/tests/ 历史存档目录除外）。
- [x] AC6：真实对话回归——completions 路径（deepseek-v4-flash，带工具调用）与 responses 路径（gpt-5.6-luna，完整对话 + usage 含 reasoning_tokens + finish 映射）各跑一轮真实调用成功；`finish_reason: null` 类畸形响应映射为 error finish（无产出时）或宽容 stop（有产出时）且不崩溃。（备注：muse-spark-1.2 在验收时被 OpenCode 网关整体下线——两协议均 401 Model not supported，与代码无关；responses 协议由 Luna 完成验证。）
- [ ] AC7：Responses 手工构造的历史 input 不包含 output-only 的 `status` 字段；多轮请求顺序保持不变，Console Go 不再报 `input[n].status` unknown parameter。
- [ ] AC8：Responses 适配器在 `ResponseOutputItemDoneEvent` 收到时捕获原始 Output Item 元数据，Host 可以持久化并在下一轮原样重放；ThinkingBlock 仍只承担 UI 展示，不冒充传输 Item。
- [ ] AC9：Vision 回归覆盖公开 URL、Base64 data URL、`file_id` 至少一种实际调用路径；断言 Responses 请求使用 `input_text`/`input_image`，而不是 Chat Completions 的嵌套 `image_url` 对象。
- [ ] AC10：缺少原始 reasoning Output Item 的旧会话走显式安全诊断/降级，不合成供应商未知字段；图片输入不会触发 reasoning 状态错误。

## 6. 测试计划

- 单元测试：
  - `tests/test_stream_chunk_contract.py`（新增）：BlockAssembler 组装 + 畸形序列拒绝（AC1）
  - `tests/test_registry.py`（新增）：工厂分发 / INVALID_API_TYPE（AC3）
  - `tests/test_adapters_chunk.py`（新增）：两适配器的 fake 流 → chunk 序列断言（AC4）
  - `tests/test_responses_input_contract.py`（新增）：历史 Output Item 重放不带手工 `status`、原始
    response metadata 保存、`input_image` 的 URL/data URL/file_id 形态（AC7-AC10）
  - `tests/test_execute_reasoning.py` / `test_react_runner_*.py`：断言更新为 StreamChunk 语义（fake_llm 产新协议）
- 手动验证：AC6 双协议真实对话
- 回归基线：开发前先跑全量测试记录通过清单，迁移后逐文件比对（断言语义变化但覆盖点不减）

## 7. 变更记录

| 日期 | 变更内容 | 理由 |
|---|---|---|
| 2026-08-19 | 补齐 Responses 标准 usage 字段映射：`input_tokens` / `output_tokens` → `prompt_tokens` / `completion_tokens`；AC4 已重跑验证 | Muse Spark 真实调用返回 Responses 字段，未映射导致 Runner 丢弃本轮精确 token 用量 |
| 2026-08-19 | 评审前修订：适配器输出协议由「保留 LLMEvent」改为「DSH StreamChunk」（七种 chunk + BlockAssembler + error/aborted finish 通道）；FR5/FR6/FR7/FR8 与 AC1/AC2/AC4/AC5 相应重写；目录结构新增 block_assembler.py；非目标删除「不改 LLMEvent」、新增「不留骑墙层」 | 用户评审决策：采用 DSH 协议（与 Msg ContentBlock 同构、desktop 块渲染就绪、error finish 通道、对齐 DSH 适配器生态） |
| 2026-08-19 | 状态 草稿 → approved（定稿） | 用户评审通过 |
| 2026-08-19 | FR 勾选 + AC 全部验收通过（AC1-AC6）；状态 → 已验收。AC6 备注：muse-spark-1.2 验收时被网关下线（401 Model not supported，与代码无关），responses 协议由 gpt-5.6-luna 完成真实回归；completions 协议由 deepseek-v4-flash 完成（工具调用 + usage 全通） | 开发完成，验收记录留痕 |
| 2026-08-25 | 修复 Responses 适配器漏识别 OpenAI SDK 的 `ResponseReasoningTextDeltaEvent` / `ResponseReasoningSummaryTextDeltaEvent`；新增真实事件名回归测试 | OpenCode 直连 DeepSeek V4 Flash 已返回 `response.reasoning_text.delta`，旧判断只识别 `ResponseReasoningDeltaEvent`，导致思考增量被静默丢弃 |
| 2026-08-25 | 修复 Thinking Tool Loop 的 reasoning 回传：Responses 转换器把持久化 `reasoning_content` 重建为标准 `type=reasoning/content=reasoning_text` input item，并保留 reasoning-only 历史 | DeepSeek V4 Flash 在 thinking 模式的 Tool Result 后要求上一轮 reasoning_text 原样回传，否则下一轮返回 400 |
| 2026-08-25 | Tool Call 结束事件增加完整原始 `arguments`；Completions 适配器按 wire index 缓存并补发 call_id 晚到前的参数片段 | 客户端实时增量可能缺片，但结束快照必须能覆盖恢复；参数先于 call_id 到达时不能静默丢弃 |
| 2026-08-25 | `Msg.append_event` 以 `TOOL_CALL_END.arguments` 重建持久化入参，只有旧事件缺字段时才回退 delta 缓冲；新增事件重建回归 | 结束事件是唯一完整快照，不能让持久化层继续只依赖可能丢失的实时增量 |
| 2026-08-25 | 官方 Responses 协议复核：`status` 是 API 返回 Output Item 的状态字段，不得由 Core 人工拼进请求 input；修复方案改为保存/重放原始 Output Item，并补充 `input_image` Vision 契约（FR9/FR10、AC7-AC10） | `ws_sess_e2a68a947984` 重现 `input[2].status` 400；图片形态本身正确，根因是 reasoning 历史传输边界丢失 |
