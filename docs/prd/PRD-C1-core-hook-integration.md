# PRD-C1：Agent Core 直接消费 ftre Hook 协议

## 元信息

| 字段 | 值 |
|---|---|
| 阶段 | C1 |
| 名称 | Agent Core 直接消费 ftre Hook 协议与 Turn-stopping Continuation |
| 状态 | 已验收 |
| 创建日期 | 2026-08-21 |
| 定稿日期 | 2026-08-21 |
| 验收日期 | 2026-08-21 |
| 关联文档 | `docs/TODO.yaml` C1；`E:/ftre/docs/prd/PRD-F7-agent-core-hook-integration.md` |

## 1. 背景与目标

`ftre-agent-core` 当前自带 `FtreCoreHookManager`，由 Core 自己持有注册表和旧的
`ON_*` 回调模型。ftre 后端已经拥有基于 Cordis 的类型化 `HookRuntime`，两套 Hook
系统之间通过 `ToolHookBridge`、`HookedToolRegistry` 和 `HookedLLMAdapter` 转换，导致
核心执行路径存在重复 Owner、上下文丢失和生命周期不一致。

本阶段将 Core 改为无状态算法层：Core 只定义稳定的跨仓库 Hook 契约并在工具、LLM
stream、Agent 停止决策处调用一个宿主注入的 Dispatcher；监听器注册、Cordis scope、
插件生命周期和诊断全部由 ftre 负责。

非目标：不把 Core 依赖到 ftre、Cordis、Session 或 Gateway；不迁移 ftre 专属的
mailbox、prompt、compaction、request-error 等 Hook；不改变既有 LLM provider 协议。

## 2. 需求范围

### 2.1 功能需求

- [x] FR1：Core 提供无状态 `HookDispatcher` 协议和类型化 `HookSpec`，Dispatcher
  接受 `dispatch(spec, payload, context=None)`，context 为宿主的 opaque scope carrier。
- [x] FR2：Core 定义并导出选定的公共契约：`tools/pre-execute`、
  `tools/execute`、`tools/post-execute`、`tools/result`、`llm/stream`、
  `agent/turn-stopping`，ftre 直接复用同一类和同一 Spec 实例。
- [x] FR3：ToolHandler 直接通过 Dispatcher 执行四段 Tool Hook；不再创建或依赖
  `FtreCoreHookManager`，工具参数、执行结果和 metadata 保持可追踪。
- [x] FR4：ReasoningExecutor 在原始 LLM stream 边界调用 `llm/stream` waterfall，
  Hook 可包裹、替换或拒绝 stream，取消和异常按宿主策略传播。
- [x] FR5：ExitExecutor 在完成态 finalize 之前调用 `agent/turn-stopping`；
  `StopTurn` 允许 finalize，`ContinueTurn(prompt)` 注入内部 continuation 并继续
  当前 Turn；Core 在取消或 continuation 预算耗尽时不得继续。
- [x] FR6：ReActAgent 只接受宿主注入的 `hooks` 与 `hook_context`，默认无 Hook 时
  行为与现状一致；Core 不持有进程级可变 Hook 注册表。
- [x] FR7：删除 Core 旧 `ON_*` 常量、输入输出可变对象、`FtreCoreHookManager`
  和对应生产路径；文档、测试和公共导出同步更新。
- [x] FR8：ftre 直接把 Cordis `HookRuntime` 和 Agent scope context 注入 Core，
  删除 `ToolHookBridge`、`HookedToolRegistry`、`HookedLLMAdapter` 及其引用。

### 2.2 非功能需求

- Core 不新增 ftre/Cordis 运行时依赖；仅使用标准库和自身类型。
- Hook payload 为 frozen/只读边界；hook 失败由 HookSpec 的 failure policy 决定。
- 取消、重试、并发工具和流式 chunk 的既有语义不回归。

## 3. 技术方案

### 3.1 Core 公共边界

`src/ftre_agent_core/hooks.py` 只保留契约、Spec 和 Dispatcher Protocol；公共
payload/result 定义放在同一模块，ftre 的 `services/tools/hooks.py`、
`services/llm/hooks.py`、`services/agent/hooks.py` 通过导入/重导出保持原有业务导入
路径可读，但不再声明第二份数据类。

### 3.2 运行时注入

`ReActAgent(hooks=None, hook_context=None)` 把两个对象传给 Runner 和各执行器。
Core 不知道 context 的具体实现；ftre 传入 `HookRuntime` 和由 `AgentRegistry` 生成的
Cordis scope Context。

### 3.3 停止状态机

```
模型返回纯文本
  → ExitExecutor 构造 TurnStoppingPayload
  → Dispatcher.dispatch(agent/turn-stopping)
  → StopTurn       → finalize → ReplyEnd
  → ContinueTurn   → 写入内部 HintBlock → 下一次 Reasoning
```

`continuation_count` 从 0 开始递增，`max_continuations` 来自 runtime context；
达到上限、取消或 Hook 返回非法结果时不得进入无限循环。

## 4. 接口定义

```python
class HookDispatcher(Protocol):
    async def dispatch(
        self, spec: HookSpec, payload: Any, *, context: object | None = None
    ) -> Any: ...

class ContinueTurn:
    prompt: str
    reason: str = ""
    source: str = ""
```

Core 与 ftre 共享的稳定 Hook 名称为：
`tools/pre-execute`、`tools/execute`、`tools/post-execute`、`tools/result`、
`llm/stream`、`agent/turn-stopping`。

## 5. 验收标准

- [x] AC1：Core `pytest` 全部通过，旧 HookManager 测试已替换为 Dispatcher 合同测试。
- [x] AC2：Core `ruff check .` 通过，且 `rg` 不再发现生产代码中的
  `FtreCoreHookManager`、`ON_TURN_START`、`ON_PRE_TOOL`、`ON_POST_TOOL`、`ON_STOP`、
  `ON_TURN_END`。
- [x] AC3：ftre `pytest -q`、`ruff check --no-cache src tests`、Gateway smoke 全部通过。
- [x] AC4：ftre 生产路径不再导入 `ftre.infrastructure.agent_core` 中的桥接适配器，
  该目录及其测试引用被删除。
- [x] AC5：Fake Dispatcher 测试证明 Tool pre/execute/post/result、LLM stream 和
  turn-stopping 均收到 typed payload；`ContinueTurn` 会继续且预算耗尽时停止。
- [x] AC6：取消、Hook 异常、工具失败、LLM 重试和权限恢复场景不丢失终态事件。
- [x] AC7：最终测试完成后清理 `__pycache__`、`.pytest_cache`、`.ruff_cache` 等生成物，
  再次执行 `git status`、`git diff --check` 和架构引用扫描。

## 6. 测试计划

- Core：Dispatcher/Spec 单元测试、ToolHandler Hook 管线、LLM stream around、
  Exit continuation、默认无 Hook 回归。
- ftre：真实 `HookRuntime` + Cordis scope 集成测试、Tool/LLM/turn-stopping
  端到端测试、生命周期 unload/restart 和取消屏障测试。
- 手动：启动 Gateway，确认 composition 完成、AgentLoop 可启动并关闭。

## 7. 变更记录

| 日期 | 变更内容 | 理由 |
|---|---|---|
| 2026-08-21 | 建立 Core 侧 C1 PRD；允许在独立 Core feature 分支修改实现，ftre 仍通过单向依赖消费协议 | 用户授权推进跨仓库直接协议集成 |
| 2026-08-21 | C1 验收完成：Dispatcher、Tool/LLM/turn-stopping 直连、旧 HookManager 删除、Core 全量 pytest/ruff 通过 | 完成与 ftre F7 的协调交付 |
