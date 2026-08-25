# PRD-C6 LLM Stream Fallback Metadata

## 元信息

| 字段 | 值 |
|---|---|
| 阶段 | C6 |
| 名称 | LLM Stream Fallback Metadata |
| 状态 | 已验收 |
| 创建日期 | 2026-08-25 |
| 定稿日期 | 2026-08-25 |
| 验收日期 | 2026-08-25 |
| 关联文档 | `docs/TODO.yaml` C6；`E:\\ftre` F29；`PRD-C5-llm-error-hook.md`；`AGENTS.md` |

## 1. 背景与目标

C5/F28 只负责 `llm/error` 的 retry/stop 决策。备用模型切换属于单次流包装行为，必须由
`llm/stream` Plugin 拥有；但当前 `LLMStreamPayload` 没有 `attempt` 和 `max_attempts`，
Plugin 无法保证“先完成 Core Retry，最后一次 attempt 才 fallback”。

本阶段只给现有 `llm/stream` Payload 增加稳定的尝试元数据，不把 fallback 算法下沉 Core。

### 非目标

- 不在 Core 创建备用 Adapter 或解析 ftre 配置；
- 不新增 `llm/fallback` Hook；
- 不修改 Core RetryExecutor 的错误分类、RetryEvent、取消和收尾；
- 不让 fallback 访问 Session、Inbox、AgentState 或客户端协议。

## 2. 需求范围

- [x] **FR1：扩展 `LLMStreamPayload`**
  - 增加 1-based `attempt`；
  - 增加本次 Reasoning 的 `max_attempts`；
  - 保持 payload 不可变，既有字段和 `invoke` 语义不变。
  - 默认构造值为 `attempt=1`、`max_attempts=1`，只兼容直接构造的旧宿主；Core
    真实 dispatch 必须显式传入本轮坐标。

- [x] **FR2：每个 attempt 重新触发 `llm/stream`**
  - Core Retry 进入下一次调用时重新构造 Payload 并 dispatch；
  - `attempt` 与 Core 的 `RetryEvent.attempt` 使用同一 1-based 语义；
  - 没有 Hook 时实际 LLM 行为保持一致。

- [x] **FR3：取消和失败边界**
  - `payload.cancellation` 已置位时，Plugin 不得切换备用模型；
  - Core/Plugin 必须区分直接异常和 `FinishChunk(kind="error")`；
  - 已产生正文、思考或 Tool Call 后，不允许无感切换，避免重复消息。

- [x] **FR4：为最后一次 fallback 提供充分信息**
  - fallback Plugin 使用 `attempt == max_attempts` 判断是否进入备用模型；
  - Core 不负责判断 fallback 错误码或选择备用模型；
  - 备用流必须继续输出同一 `StreamChunk` 协议。

- [x] **FR5：跨仓版本与边界**
  - Core 只增加元数据和校验，不持有备用 Adapter、ConfigService 或 Plugin 状态；
  - ftre 复用同一 `LLMStreamPayload`，不得复制 DTO 或定义第二份 `llm/stream` Spec；
  - `attempt` 从 1 开始且不得大于 `max_attempts`，非法坐标在 Payload 构造时拒绝。

## 3. 技术方案

```text
attempt 1..N-1
  └─ llm/stream → 主模型 → 失败 → Core Retry

attempt N
  └─ llm/stream → 主模型 → 无输出失败 → fallback Plugin → 备用模型
```

Core 只负责在 `_execute_reasoning.py` 创建 Payload 时填入：

```python
LLMStreamPayload(
    ...,
    attempt=attempt + 1,
    max_attempts=max_attempts,
)
```

Fallback Plugin 通过已有 Waterfall Hook 包装流；Core 不感知备用模型，也不改变 Agent 的固定
主 Adapter。

## 4. 验收标准

- [x] **AC1**：首次调用、Tool 后下一轮和每次 Core Retry 的 `attempt` 从 1 单调递增。
- [x] **AC2**：`max_attempts == 1 + max_retries`，无 Hook 行为与基线一致。
- [x] **AC3**：直接异常、错误 FinishChunk、取消、空响应和已有有效输出均有测试。
- [x] **AC4**：Core 只增加元数据，不创建或持有备用模型 Adapter。
- [x] **AC5**：Core pytest、ruff、wheel、洁净安装和 C5 跨仓合同通过。
- [x] **AC6**：ftre Host 重导出身份、每次真实 Core Retry 的 attempt 顺序和 Package fallback
  集成测试通过。

## 5. 测试计划

- `tests/test_llm_stream_payload.py`：字段、不可变性、1-based attempt；
- `tests/test_execute_reasoning.py`：RetryEvent 与 Payload attempt 对齐；
- `tests/test_adapters_chunk.py`：错误 FinishChunk 不改变协议；
- Hook 生命周期和取消回归。

## 6. 变更记录

| 日期 | 变更内容 | 理由 |
|---|---|---|
| 2026-08-25 | 初始草案：为 `llm/stream` 增加 attempt/max_attempts，支持 F29 在最后一次 attempt 执行 fallback | C5/F28 只拥有 llm/error；fallback 需要知道当前是否已完成 Core Retry |
| 2026-08-25 | 完成 Payload 坐标校验、每次 Retry 重建 dispatch、Core 0.2.2 发行元数据及 ftre 跨仓集成验证 | 让 fallback 只依赖稳定 attempt 边界，不把备用模型状态下沉到 Core |
