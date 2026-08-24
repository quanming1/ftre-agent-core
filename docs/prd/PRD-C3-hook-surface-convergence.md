# PRD-C3-Core Hook 面终局收敛

> 本阶段与 `E:\ftre\docs\prd\PRD-F16-core-hook-surface-convergence.md` 配对。
> C3 只拥有 Agent Core 的协议、DTO 和算法调用点；Host、Package、Cordis 生命周期由 F16 负责。

## 元信息

| 字段 | 值 |
|---|---|
| 阶段 | C3 |
| 名称 | Core Hook 面终局收敛 |
| 状态 | 已验收 |
| 创建日期 | 2026-08-24 |
| 定稿日期 | 2026-08-24 |
| 验收日期 | 2026-08-24 |
| 关联文档 | `docs/TODO.yaml` C3；`docs/PROCESS.md`；配对阶段 F16 |

## 1. 背景与目标

### 1.1 背景

C1/C2 为宿主提供了直接可消费的 Hook 协议，但当前 Tool 面仍有四个名称：
`tools/pre-execute`、`tools/execute`、`tools/post-execute`、`tools/result`。
其中 `execute` 是只包裹 Core 自有执行器的 around 层，`result` 是不改变结果的观察层。
它们把同一次 Tool 调用拆成过多概念，增加宿主注册、失败语义和版本迁移成本。
Agent 停止决策也使用了 `agent/turn-stopping`，名称描述的是过程而非 Hook 的真实职责。

### 1.2 目标

在不改变 Agent 行为的前提下，将 Core 公共 Hook 从 7 个收敛为 5 个：

```text
tools/pre-execute  ─┐
tools/execute       ├─ 删除/合并为 tool/before
tools/post-execute  ┘             tool/after
tools/result        ─────────── 删除
agent/turn-stopping ─────────── 改名为 agent/stop-decision
```

最终保留：`tool/before`、`tool/after`、`agent/stop-decision`、
`agent/before-reasoning`、`llm/stream`。

### 1.3 非目标

- 不改变 ReAct 的消息、State、ToolResult 归一化或并发模型。
- 不新增 `Port`、Coordinator、Facade、Service Locator、Service Bag 或第二执行器。
- 不把 Host 的队列、Session、Compaction、Command、Cordis 类型带入 Core。
- 不在 C3 修改 `E:\ftre`、客户端、`E:\cordis-py`，不发布 PyPI、不 push、不 merge。
- 不为了“对称”增加 `tool/during`、`tool/observe` 等新 Hook。

## 2. 现状与目标边界

| 领域 | 当前 Core | C3 目标 | Owner |
|---|---|---|---|
| Tool 进入前 | `tools/pre-execute`，允许/拒绝/替换参数 | `tool/before`，语义不变 | Core 定义，Host 注册监听器 |
| Tool 实际执行 | `tools/execute` around continuation | 删除；由 Core 私有 `invoke` 直接执行 | Core 私有算法 |
| Tool 执行后 | `tools/post-execute`，可替换结果 | `tool/after`，接收归一化结果并可替换 | Core 定义，Host 注册监听器 |
| Tool 观察 | `tools/result` EMIT | 删除；观察者监听 `tool/after` 或 Host 自有日志 | Core 不再发布独立事件 |
| 自然停止 | `agent/turn-stopping` | `agent/stop-decision`，DTO 语义不变 | Core 定义，Host 提供策略 |
| LLM 调用 | `llm/stream` | 保持不变 | Core/Host |
| Reasoning 边界 | `agent/before-reasoning` | 保持不变 | Core/Host |

## 3. 功能需求

- [x] FR1：新增 `tool/before` HookSpec，保留当前 WATERFALL、Agent scope、默认放行以及 `ToolAllow`、`ToolDeny`、`ToolArguments` 结果类型。
- [x] FR2：新增 `tool/after` HookSpec，保留当前 WATERFALL、Agent scope、归一化 `ToolExecutionResult` 输入/输出和“不得重新执行 Tool”不变量。
- [x] FR3：删除 `tools/execute` around 协议；原始 Tool 只由 `ToolHandler` 私有 continuation 执行，Hook 不再拥有执行权。
- [x] FR4：删除 `tools/result` EMIT 协议；`tool/after` 完成后由调用方获得最终 `ToolResult`，观察需求由宿主在同一阶段实现。
- [x] FR5：将 `agent/turn-stopping`、对应 Spec、Payload/导出和调用点改为 `agent/stop-decision`；`StopTurn`/`ContinueTurn` 的行为和类型校验保持不变。
- [x] FR6：Core 对旧名称不提供 alias、兼容常量、双 dispatch 或字符串 fallback；旧名称在源代码、测试和公共导出中为零。
- [x] FR7：未注入 Dispatcher 时，五个 Hook 均按默认实现运行，Core 可独立执行 Tool 和 Agent Turn。
- [x] FR8：Hook 异常、Tool 异常、取消和 `ContinueTurn` 的现有失败/取消/续写语义不改变；没有新增吞错。
- [x] FR9：Core 测试覆盖顺序、改参、拒绝、结果改写、异常、取消、停止和继续；使用 fake Tool/LLM，不调用真实 API。
- [x] FR10：更新 Core 版本/变更记录/构建元数据，生成不含旧协议和缓存的 wheel；发行候选不在 C3 直接推送。

## 4. 协议定义

### 4.1 `tool/before`

```text
HookSpec:
  name: "tool/before"
  mode: WATERFALL
  payload: ToolPreExecutePayload
  result: ToolAllow | ToolDeny | ToolArguments
  default: ToolAllow()
  scope: AGENT
```

Payload 继续提供不可变的 `ToolCallIdentity`、参数快照和取消事件。监听器可以拒绝调用或返回
替换参数；不能直接修改 Core Registry、State 或执行器。

### 4.2 `tool/after`

```text
HookSpec:
  name: "tool/after"
  mode: WATERFALL
  payload: ToolPostExecutePayload
  result: ToolExecutionResult
  default: payload.result
  scope: AGENT
```

此 Hook 在 Core 已完成一次实际执行并归一化结果后触发。监听器可脱敏、补充 metadata、标记
失败或替换展示输出，但不能再次调用原始 Tool。异常仍按现有 Hook failure policy 处理。

### 4.3 `agent/stop-decision`

Payload 继续包括 Agent/session/turn/request 坐标、自然停止状态、最后一段助手文本、finish reason、
iteration、continuation_count、max_continuations 和 cancellation。结果仍为 `StopTurn` 或
`ContinueTurn(prompt, reason, source)`；Core 只根据结果决定结束或把 continuation prompt 放回下一次 Reasoning。

## 5. 实施边界与迁移矩阵

| 当前符号/调用点 | C3 目标 | 操作 | 验证 |
|---|---|---|---|
| `TOOLS_PRE_EXECUTE` / `TOOLS_PRE_EXECUTE_SPEC` | `TOOL_BEFORE` / `TOOL_BEFORE_SPEC` | 改名并迁移导出 | Hook 契约测试 |
| `ToolPreExecutePayload`、`ToolAllow`、`ToolDeny`、`ToolArguments` | 保持类型语义 | 仅按新协议引用 | 参数/拒绝测试 |
| `TOOLS_EXECUTE`、`ToolExecutePayload`、`TOOLS_EXECUTE_SPEC` | 无 | 删除 around 层和 DTO | AST 零引用 |
| `TOOLS_POST_EXECUTE` / `ToolPostExecutePayload` | `TOOL_AFTER` / 同语义 Payload | 改名并迁移调用 | 结果改写测试 |
| `TOOLS_RESULT`、`ToolResultPayload`、`TOOLS_RESULT_SPEC` | 无 | 删除观察事件 | AST/导出测试 |
| `agent/turn-stopping` / `AGENT_TURN_STOPPING_SPEC` | `agent/stop-decision` | 改名，DTO 行为不变 | 停止/继续测试 |
| `ToolHandler.run_one` around dispatch | 私有 `invoke` + before/after 两次 dispatch | 移除 execute dispatch | 顺序断言 |
| `_execute_acting.py` 停止 dispatch | 新 Stop Decision dispatch | 只改 Spec 名称/文案 | 状态机回归 |
| Core `__all__`、测试、文档 | 新公共表面 | 清理旧导出/旧名称 | `rg` 门禁 |

## 6. 兼容与版本策略

C3 是有意的破坏性公共协议收敛。Core 不提供旧名称别名；ftre F16 必须先升级 Core 版本再迁移。
版本号、`CHANGELOG.md` 和 wheel 元数据必须明确列出删除项。未升级的 Host 失败应是清晰的导入/契约
错误，而不是静默降级或双协议执行。

## 7. 验收标准

- [x] AC1：`rg` 在 `src`、`tests`、公共导出和文档中找不到旧五个活动名称（历史迁移记录除外，必须明确标注）。
- [x] AC2：HookSpec 清单恰好为 5 个，名称与第 1.2 节一致，无重复 key。
- [x] AC3：Tool 顺序为 `tool/before → 私有 invoke → tool/after`；拒绝不执行，after 不重复执行。
- [x] AC4：工具失败、取消、malformed arguments 和 Hook 异常的结果/日志/取消传播与 C2 基线一致。
- [x] AC5：`agent/stop-decision` 的 Stop/Continue 与 continuation 次数限制回归通过。
- [x] AC6：无 Dispatcher 的独立 Core smoke 通过；注入 fake Dispatcher 的 Hook 合同通过。
- [x] AC7：`pytest -q`、`ruff check .`、`git diff --check` 全部通过，测试不依赖真实 API Key。
- [x] AC8：wheel build 后解包扫描不含 `__pycache__`、`.pyc`、测试临时数据或旧协议活动代码。
- [x] AC9：版本和变更记录说明破坏性改名；未授权的 push、merge、release 未发生。

## 8. 分批任务

| 批次 | 交付 |
|---|---|
| 00 | 本 PRD、TODO、配对矩阵/共同 AC 和执行报告壳；等待评审 |
| 01 | 只读消费者、行为、版本和测试基线 |
| 02 | Tool 两段协议实现与旧四段删除 |
| 03 | Stop Decision 改名与旧协议删除 |
| 04 | Core 全量验证、wheel、版本与收尾 |
| 05 | ftre 升级 Core、迁移 Host/Package/测试 |
| 06 | 两仓洁净安装、E2E、清理和最终验收 |

## 9. 测试计划

- `tests/test_hooks.py`：Spec 名称、类型、默认值、导出清单。
- `tests/agent/runner/`：Tool before/after 顺序、拒绝、改参、结果改写、异常和取消。
- `tests/agent/runner/`：Stop/Continue、自然停止条件、最大 continuation 和失败路径。
- 独立 smoke：不注入 Hook Dispatcher 时可运行一个 fake Tool Turn。
- 发行验证：wheel build、临时虚拟环境安装、导入和最小 fake Turn。

## 10. 变更记录

| 日期 | 变更内容 | 理由 |
|---|---|---|
| 2026-08-24 | 建立 C3 草案，定义 Core 7→5、Tool 4→2 与停止决策改名 | F15 已冻结 Host 面，需要独立 Core 阶段降低协议复杂度 |
| 2026-08-24 | 用户授权继续执行，PRD 进入开发中 | 批次 00 评审门已通过，按 01→06 顺序落地 |
| 2026-08-24 | 完成 C3.1–C3.4：Core Hook 7→5、版本 0.2.0、wheel 与全量回归 | F16 Host 迁移需要稳定的新协议发行候选 |
