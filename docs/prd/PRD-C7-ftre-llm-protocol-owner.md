# PRD-C7 ftre-llm 协议 Owner 收敛

## 元信息

| 字段 | 值 |
|---|---|
| 阶段 | C7 |
| 名称 | ftre-llm 协议 Owner 收敛 |
| 状态 | 已验收 |
| 创建日期 | 2026-08-26 |
| 关联文档 | `docs/TODO.yaml` C7；`E:\\ftre` F30；`PRD-C6-llm-stream-fallback.md` |

## 1. 背景与目标

C6 之后 Core 和 ftre 仍各自声明一套同名 StreamChunk。Python 的类身份不同，Host
不得不复制事件才能让 Core Runner 识别。目标是把 LLM 流协议迁移到 `ftre-llm`，
Core 只重导出并消费该协议，运行时不再存在转换桥。

本阶段同时收敛错误决策：Retry 统一由 Core `llm/error` Hook 决定，LlmService 不再
发布返回值无人消费的平行错误 Hook。

## 2. 需求范围

- [x] FR1：`ftre_llm.events` 是 StreamChunk 七种事件和 `ToolCall` 的唯一实现。
- [x] FR2：Core Runner、Core Adapter 和 BlockAssembler 直接从 `ftre_llm.events`
  导入 StreamChunk；Core 不再保留 `ftre_agent_core.llm.events` 兼容模块。
- [x] FR2.1：Core `llm/stream` Spec 与 Host 重导出同一对象，Payload 使用 ftre-llm
  契约，避免 Agent 外层和 LlmService 内层重复注册同名 Spec。
- [x] FR3：Agent Runtime 使用无输出转换的 `LlmServiceAdapter`；Core Runner 支持在
  `ReActAgent(..., llm=...)` 构造阶段注入该 seam，删除 Host Core bridge 和运行中
  直接写入 Runner 私有字段的接线。
- [x] FR4：Retry Plugin 监听 Core `llm/error` 并返回 `LLMErrorDecision`，决策实际影响
  Core 的 retry/stop；LlmService 不重复派发错误决策 Hook。
- [x] FR5：ConfigService 的 LLM 解析快照包含 `context_window`、`vision`、
  `reasoning_effort_values`。

## 3. 非目标

- 不修改 Core Agent 的业务状态机、Tool、Session 或客户端协议。
- 不在 LlmService 内新增第二套 Retry Loop。

## 4. 验收标准

- [x] AC1：Core 与 ftre 导入的 `TextDeltaChunk`、`FinishChunk` 为同一对象。
- [x] AC2：Agent 通过 LlmService 输出的 chunk 能被 Core Runner 直接识别，无转换函数。
- [x] AC2.1：Core 与 Host 的 `LLM_STREAM_SPEC` 对象身份相同，Fallback 只被调用一次。
- [x] AC2.2：注入 LLM 时 Core 不提前构造默认 OpenAI 客户端；运行中的 Runner 拒绝
  替换适配器，避免单轮请求跨 Provider。
- [x] AC3：Recovery Plugin 返回 `LLMErrorDecision` 后，Core RetryEvent/Stop 行为改变。
- [x] AC4：无 Recovery Plugin 时 Core 原有默认重试行为保持不变。
- [x] AC5：`resolve_llm()` 返回完整模型能力字段。
- [x] AC6：Core/ftre 全量 pytest、ruff、diff check 通过。

## 5. 测试计划

- Core：协议 Owner 身份、Runner 直接消费、错误决策回归。
- ftre：LlmServiceAdapter 透传、Recovery Plugin Core Hook 集成、配置能力快照。

## 6. 变更记录

| 日期 | 变更内容 | 理由 |
|---|---|---|
| 2026-08-26 | 新增 C7：将 StreamChunk Owner 从 Core 迁入 ftre-llm，并统一 `llm/error` Retry 决策 | 消除运行时两套 Chunk 类型、Host 转换桥和无效错误 Hook |

审计补充：删除 `ftre_agent_core.llm.events` 旧模块入口，所有 Core 适配器和测试直接
使用 `ftre_llm.events`；不保留兼容 re-export。
| 2026-08-26 | Core Runner 增加构造期 LLM 注入和受运行锁保护的 `set_llm()`；ftre 不再写 `runner._llm` | 避免空凭据客户端先初始化，并收紧宿主接线边界 |
