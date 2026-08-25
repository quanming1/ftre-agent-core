# PRD-C5 LLM Error Hook

> 本阶段只定义并落地 Core-facing 的失败决策边界；不把 ReAct 重试执行器搬到宿主 Plugin。

## 元信息

| 字段 | 值 |
|---|---|
| 阶段 | C5 |
| 名称 | LLM Error Hook |
| 状态 | 已验收 |
| 创建日期 | 2026-08-25 |
| 定稿日期 | 2026-08-25 |
| 验收日期 | 2026-08-25 |
| 关联文档 | `docs/TODO.yaml` C5；`docs/PROCESS.md`；`E:\\ftre` F28；`AGENTS.md` |

## 1. 背景与目标

### 1.1 代码评审结论

当前失败重试全部位于 `src/ftre_agent_core/agent/runner/_execute_reasoning.py` 的
`ReasoningExecutor.stream()`：

1. 适配器把供应商异常归一化为 `FinishChunk(kind="error")`，执行器再还原为 `LLMError`；
2. 执行器根据 `LLMError.UNRETRYABLE_CODES` 和 `max_retries` 决定是否重试；
3. 重试时由 Core 发送 `RetryEvent`、等待退避、重新读取消息、清空半截 block 收集器；
4. 最终失败才由 ftre 的 `agent/run-error` Hook 接收，现有 `RetryRequest` 只能重跑 Turn，
   不能改变已经创建的 Core LLM Adapter。

当前 `llm/stream` 只有静态契约和 ftre 重导出，没有生产 Plugin 注册监听器。本阶段不实现它的
fallback 行为；该能力另由 C6/F29 负责。

### 1.2 目标

在每次 LLM attempt 已被 Core 归一化为失败后，发布一个类型化的
`llm/error` Hook，让 Plugin 可以决定“沿用 Core 默认策略、重试或停止”；
Core 继续拥有重试执行、状态、事件、取消和流式一致性。

### 1.3 非目标

- 不把 `ReasoningExecutor`、`RetryEvent`、`RunState` 或消息收集器迁移到 Plugin；
- 不通过 Hook 直接替换 Core 私有的 `_llm` 实例；
- 本阶段不实现备用模型切换；`llm/stream` fallback 另见 C6/F29；
- 不改变空响应重试语义、Tool 重试语义、取消语义和现有 `agent/run-error` 语义；
- 不引入 `AgentControlPort`、Service Locator、全局 setter 或第二套 Retry Loop。

## 2. 需求范围

### 2.1 功能需求

- [x] **FR1：新增稳定 Hook 名称 `llm/error`**
  - Hook 只在一次真实 LLM attempt 产生 `LLMError` 后触发；
  - `CancelledError`、`aborted` 和空响应不触发；
  - Hook 的定义 Owner 是 Core，宿主只提供 Dispatcher。

- [x] **FR2：定义最小失败 Payload**
  - 至少包含 `session_id`、`turn_id`、`iteration`、`model`、`error_code`、`error_message`、
    `attempt`、`max_attempts` 和 `cancellation`；
  - 不携带 API Key、原始异常对象或可变 AgentState；
  - `error_message` 允许脱敏后传递，不能要求 Plugin 解析供应商私有异常类型。

- [x] **FR3：定义恢复决策结果**
  - `None` 表示不干预，Core 使用当前默认策略；
  - `LLMErrorDecision(action="retry")` 表示请求 Core 重试；
  - `LLMErrorDecision(action="stop")` 表示立即结束本次 Reasoning；
  - `delay` 只能作为建议值，Core 对其做非负化并执行最大尝试次数和取消屏障；
  - 本阶段不定义 `fallback_model` 字段，避免让 Hook 侵入 Core Adapter 生命周期。

- [x] **FR4：保留 Core 重试执行器为唯一 Owner**
  - Core 继续维护 attempt 计数、硬上限、RetryEvent、退避、消息重读、半截输出清理和最终
    `TurnResult(error)`；
  - Plugin 不能通过返回值突破 `max_retries`、取消或重试 token 限制。

- [x] **FR5：默认行为逐字节保持一致**
  - 没有监听器、监听器返回 `None`、或可选监听器失败时，现有可重试/不可重试分类和次数不变；
  - Hook 失败采用 fail-open 语义：记录诊断后回到 Core 默认策略，不把可选 Plugin 故障升级为
    Agent 失败。

- [x] **FR6：Hook 顺序和生命周期可逆**
  - Hook 为 Agent scope 的 waterfall 控制边界；
  - Listener 由宿主 Cordis Fiber/Effect 管理，unload 后不再接收新 attempt；
  - in-flight listener 必须等待收敛，不能取消正在执行的 Core LLM 调用。

- [x] **FR7：冻结错误 attempt 元数据**
  - `LLMErrorPayload.attempt` 为 1-based 当前尝试次数，`max_attempts` 为本次 Reasoning 的硬上限；
  - `llm/error` 在该 attempt 失败归一化后、Core 执行 retry/stop 前触发；
  - `max_attempts = 1 + max_retries` 的现有语义不变；`llm/stream` 的 attempt 元数据属于 C6。

### 2.2 非功能需求

- **无状态**：Core 不持有 Plugin 注册表、配置缓存或进程级重试策略。
- **安全**：禁止 API Key、完整消息正文和供应商 Authorization 信息进入 Payload/日志。
- **兼容**：无 Hook 监听器时现有 Core、ftre 和独立测试行为不变。
- **可观测**：实际是否发生 Retry 仍以 Core 发出的 `RetryEvent` 为准，不能由 Plugin 私自伪造。

## 3. 技术方案

### 3.1 调用时序

```text
ReasoningExecutor.stream()
        │
        ├─ adapter.stream()
        ├─ FinishChunk(error) → LLMError
        │
        ▼
llm/error
        │
        ├─ None → Core 默认策略
        ├─ retry → Core RetryEvent → delay → 下一 attempt
        └─ stop  → TurnResult(error)
```

### 3.2 契约草案

```python
LLM_ERROR = "llm/error"

@dataclass(frozen=True, slots=True)
class LLMErrorPayload:
    session_id: str
    turn_id: str
    iteration: int
    model: str
    error_code: str
    error_message: str
    attempt: int
    max_attempts: int
    cancellation: asyncio.Event

@dataclass(frozen=True, slots=True)
class LLMErrorDecision:
    action: Literal["retry", "stop"]
    reason: str = ""
    delay: float | None = None
```

`HookSpec` 使用 `HookMode.WATERFALL`，默认返回 `None`，失败策略为 `OBSERVE`。
Core 在原有错误分支进入默认策略前 dispatch；返回结果只改变“是否继续”，不改变执行器所有权。

### 3.3 与 `llm/stream`、`agent/run-error` 的边界

| 边界 | 责任 |
|---|---|
| `llm/stream` | C6/F29 负责包装单次流和备用模型切换，本阶段不实现 |
| `llm/error` | 单次 attempt 失败后的 retry/stop 决策 |
| `agent/run-error` | Core/Turn 已终止后的宿主维护或有限 Turn 重跑 |
| Core RetryExecutor | 实际循环、事件、状态、取消、消息重读和收尾 |

上下文溢出仍交给 ftre-compaction；不得用本 Hook绕过压缩门控。

## 4. 接口定义

### 4.1 Core 导出

`ftre_agent_core.hooks` 新增：

- `LLM_ERROR`；
- `LLM_ERROR_SPEC`；
- `LLMErrorPayload`；
- `LLMErrorDecision`。

ftre 只能重导出同一对象，不得复制 DTO 或定义第二个 Spec。

### 4.2 默认策略

```text
没有 Plugin / Plugin 返回 None
  ├─ error_code 属于 UNRETRYABLE_CODES → stop
  ├─ attempt 已到上限 → stop
  └─ 其他 → retry
```

### 4.3 版本与兼容

这是 Core 公共 Hook 面新增，不修改既有 Hook 名称；Core、ftre 必须同步版本约束和洁净安装
验证。旧客户端、旧 Tool、旧 Session 数据不参与该阶段协议。C6/F29 依赖本阶段先冻结的
`LLMErrorPayload` 和 Core attempt 上限。

## 5. 验收标准

- [x] **AC1**：Core 无监听器运行现有重试测试，结果、事件数量和错误码与基线一致。
- [x] **AC2**：监听器返回 `retry` 时，Core 仍受 `max_retries` 和取消约束，并只产生一次合法
  `RetryEvent`。
- [x] **AC3**：监听器返回 `stop` 时，不再执行后续 attempt，`TurnResult.error` 保留原错误。
- [x] **AC4**：监听器返回 `None` 或抛异常时，回到默认策略，并生成脱敏诊断。
- [x] **AC5**：`FinishChunk(kind="error")`、直接异常、`aborted`、取消、空响应和最后一次
  attempt 均有回归测试。
- [x] **AC6**：Hook unload/restart、in-flight drain、Agent scope 隔离测试通过。
- [x] **AC7**：Core pytest、ruff、wheel 和洁净安装通过；ftre 适配层可导入同一 Spec。

## 6. 测试计划

- `tests/test_llm_error_hook.py`：契约、默认策略、retry/stop、失败安全；
- `tests/test_execute_reasoning.py`：真实执行器在每个 attempt 的触发次数、RetryEvent 和边界；
- `tests/test_adapters_chunk.py`：FinishChunk error 与直接异常两条路径；
- `tests/test_hook_lifecycle.py`：unload、restart、in-flight 和 scope；
- ftre 跨仓合同测试：重导出对象身份、无重复 Owner、Gateway 无 Plugin 仍可运行。

## 7. 评审结论与架构债务

当前代码没有发现阻断该 Hook 的状态机缺口；22 个 Core/ftre 失败与生命周期回归测试已通过。
主要风险不是“能否接入”，而是职责误划：若把 Retry Loop 搬进 Plugin，会复制 Core 的流式收尾和
消息状态；若用 `agent/run-error` 控制 attempt，则触发时机已经太晚。C5 只开放决策边界，
保持 Core 的执行机制，符合“轻内核 + Plugin-first”。

## 8. 变更记录

| 日期 | 变更内容 | 理由 |
|---|---|---|
| 2026-08-25 | 历史草案曾使用 `llm/attempt-failed`（已废弃），随后改为稳定的 `llm/error`；只开放失败后的恢复决策，保留 Core RetryExecutor | 当前 `llm/stream` 无生产消费者，现有 `agent/run-error` 触发过晚；需要一个不复制状态机的稳定边界 |
| 2026-08-25 | Hook 名称收敛为 `llm/error`；将 fallback 时序和 `LLMStreamPayload.attempt/max_attempts` 移至后续 C6/F29，C5 只冻结错误决策契约 | 避免 C5/F28 同时承担 Retry Policy 和 Stream Fallback 两个独立 Owner |
| 2026-08-25 | Core/ftre 实现完成并通过全量测试、ruff、wheel；状态更新为已验收 | 先冻结可验证的错误决策边界，再进入后续 Stream Fallback 阶段 |
