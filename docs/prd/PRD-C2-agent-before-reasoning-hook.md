# PRD-C2：Agent Before-Reasoning Hook

## 元信息

| 字段 | 值 |
|---|---|
| 阶段 | C2 |
| 名称 | Agent Before-Reasoning Hook 与宿主上下文注入 |
| 状态 | 已验收 |
| 创建日期 | 2026-08-23 |
| 定稿日期 | 2026-08-23 |
| 验收日期 | 2026-08-23 |
| 关联文档 | `docs/TODO.yaml` C2；`docs/prd/PRD-C1-core-hook-integration.md`；`E:\ftre\docs\prd\PRD-F12-session-inbox-protocol.md` |

## 1. 背景与目标

C1 已经让 Core 在 Tool、LLM stream 和 turn-stopping 边界消费宿主注入的 Hook，
但仍缺少一个“当前 Turn 内、下一次 LLM Reasoning 开始前”的稳定插槽。ftre 的
`agent/pre-step` 只发生在 Turn 交付前，无法让 Inbox 在运行中的 Tool → Reasoning
边界消费 `next-step`。

本阶段增加一个最小、通用、无 ftre 依赖的 Core Hook：`agent/before-reasoning`。
它只负责让宿主在每次 LLM Reasoning 前贡献结构化上下文；队列、Session、Inbox、
Compaction 和具体消息来源仍由宿主/Plugin 拥有。

### 非目标

- Core 不 import `ftre`、`ftre_inbox`、Cordis、Session 或 Channel；
- Core 不保存 pending、QueueItem、next-turn/next-step 或队列 worker；
- 不把 `agent/before-turn`（一次 Turn 开始前）与本 Hook 合并；
- 不改变 Tool、LLM、turn-stopping 既有 Hook 的 payload 和生命周期；
- 不在 Core 内实现 Inbox 的 claim、持久化或压缩策略。

## 2. 需求范围

### 2.1 功能需求

- [x] **FR1：稳定 Hook 契约**
  - Core 定义并导出 `AGENT_BEFORE_REASONING_SPEC`；
  - Hook 名称固定为 `agent/before-reasoning`，作用域为 `AGENT`，模式为 `WATERFALL`；
  - Payload 只包含 Agent/Session/Turn/iteration/cancellation 等通用坐标；
  - Result 只允许贡献零条或多条结构化 message，不暴露 Inbox 类型。

- [x] **FR2：每次 Reasoning 前触发**
  - 首次 LLM Reasoning 前触发一次；
  - Tool 执行后进入下一次 Reasoning 前再次触发；
  - `ContinueTurn` continuation 进入下一次 Reasoning 前再次触发；
  - Exit、错误、取消和权限挂起不额外触发虚假的 Reasoning Hook。

- [x] **FR3：上下文贡献进入 Core Memory**
  - Hook 返回的 message 在 LLM snapshot 构建前写入 AgentState context；
  - message 按返回顺序追加，不覆盖已有用户、工具或系统消息；
  - Hook 无监听器时走空结果，既有 Core 行为保持不变。

- [x] **FR4：失败与取消语义**
  - Hook 异常按 HookSpec failure policy 传播；
  - cancellation 已设置时不得继续注入新消息；
  - 非法 Hook result 立即产生类型错误，不静默丢弃。

- [x] **FR5：公共导出与宿主复用**
  - `ftre_agent_core.hooks` 和顶层公开导出提供该 Spec、Payload、Result；
  - ftre 可直接复用 Core 的同一个 Spec，不创建第二份同名协议；
  - Core 测试覆盖 Fake Dispatcher 和无 Hook 默认路径。

### 2.2 非功能需求

- Core 仍为无状态算法库；Hook 不新增全局注册表或后台 Task；
- Payload/Result 使用 frozen 数据结构和只读 message 快照；
- 每次 Reasoning 至多触发一次本 Hook；
- 现有 Tool/LLM/turn-stopping、取消、重试和权限恢复行为不回归。

## 3. 技术方案

### 3.1 契约

```python
AGENT_BEFORE_REASONING = "agent/before-reasoning"

@dataclass(frozen=True, slots=True)
class BeforeReasoningPayload:
    agent: object
    session_id: str
    turn_id: str
    iteration: int
    cancellation: asyncio.Event

@dataclass(frozen=True, slots=True)
class BeforeReasoningResult:
    messages: tuple[Mapping[str, Any], ...] = ()
```

Core 只解释这些 message 是要加入当前 Memory 的结构化输入，不解释其来源。
ftre-inbox 可以把已 claim 的 `next-step` 映射成普通 user/context message，
但 Core 不会看到 QueueItem。

### 3.2 调用位置

```text
decide → Reasoning
           ↓
    agent/before-reasoning
           ↓
    append messages to AgentState
           ↓
       LLM stream
```

调用点位于 `ReActRunner._loop()` 的 `Reasoning` 分支，早于
`ReasoningExecutor.stream()` 的 message snapshot。

## 4. 接口定义

```python
class HookDispatcher(Protocol):
    async def dispatch(
        self,
        spec: HookSpec,
        payload: Any,
        *,
        context: object | None = None,
    ) -> Any: ...
```

共享名称：

```text
agent/before-turn        # ftre：一次 InboundMessage 开始前
agent/before-reasoning   # Core：每次 LLM Reasoning 开始前
agent/after-turn          # ftre：一次 Turn 完成后
```

## 5. 验收标准

- [x] **AC1**：Core HookSpec、Payload、Result 类型校验和公共导出测试通过。
- [x] **AC2**：Fake Dispatcher 证明首个 Reasoning、Tool 后 Reasoning、continuation
  后 Reasoning 均各触发一次。
- [x] **AC3**：Hook 返回 message 后，Fake LLM 收到包含该 message 的最新上下文。
- [x] **AC4**：无 Hook、Hook 异常、非法 result、取消和权限挂起测试通过。
- [x] **AC5**：`python -m pytest -q` 与 `python -m ruff check .` 通过。
- [x] **AC6**：ftre 复用同一 Core Spec，Inbox 可在运行中消费 `next-step`；ftre 全量
  测试、Gateway smoke 和 active-steer 集成场景通过。
- [x] **AC7**：Core 不出现 ftre/Inbox import，最终缓存和空目录清理完成。

## 6. 测试计划

- `tests/test_hooks.py`：Spec、默认结果、类型校验和公共导出；
- `tests/test_react_runner_step_hook.py`：Reasoning 次数、Tool 后边界、message 注入、
  cancellation、异常和 continuation；
- ftre：`tests/contracts/test_f7_hook_pipeline.py`、Inbox worker/协议/生命周期测试；
- 联合验证：Core pytest + ruff，ftre pytest + ruff，Gateway start/close 和 steer smoke。

## 7. 变更记录

| 日期 | 变更内容 | 理由 |
|---|---|---|
| 2026-08-23 | 创建并批准 C2：新增 `agent/before-reasoning`，将 ftre Turn 级 Hook 与 Core Step 级 Hook 分名 | `agent/pre-step` 与真实 Core Step 边界语义混淆；F12 需要在运行中消费 `next-step` |
| 2026-08-23 | C2 验收：Core 238 passed；ftre 421 passed；ruff、diff check、Gateway start/close 与 ftre-inbox + Core ReAct active-steer 集成测试通过 | 完成每次 Reasoning 前的通用消息注入，不让 Core 依赖 Inbox 或队列模型 |
| 2026-08-23 | F12 收尾复审：Core 238 passed；ftre 425 passed；补齐 Inbox receipt 去重和 CompletionRegistry shutdown 回归，独立 Inbox/Compaction wheel 构建通过 | 关闭跨仓库迁移后的生命周期与重复 request 资源泄漏债务 |
