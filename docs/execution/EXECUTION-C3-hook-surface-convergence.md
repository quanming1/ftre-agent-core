# C3 执行报告：Core Hook 面终局收敛

> 当前批次：04（Core 已完成）
> 状态：已完成 Core 验收，配对 ftre F16 已完成跨仓验证
> 配对 Host 阶段：`E:\ftre` F16

## 1. 范围与边界

本报告记录 C3 从批次 00 立项到批次 04 Core 收尾的完整证据。批次 00 的“只写文档”是
历史边界；后续批次已在配对 F16 约束下完成 Core 协议、测试和发行候选迁移。
未修改客户端或 `E:\cordis-py`，未 push、merge、release。

## 2. 已确认的当前事实

| 项目 | 证据 | 结论 |
|---|---|---|
| Tool Hook 当前为四段 | `src/ftre_agent_core/hooks.py` 常量与 Spec | `tools/pre-execute`、`tools/execute`、`tools/post-execute`、`tools/result` |
| Tool 调用顺序 | `agent/runner/tool_handler.py:run_one` | pre → execute around → post → result |
| Stop Hook 当前名称 | `hooks.py` 与 `agent/runner/_execute_acting.py` | `agent/turn-stopping`，结果为 StopTurn/ContinueTurn |
| Core 无状态边界 | `AGENTS.md`、`docs/PROCESS.md`、现有 Hook API | Host 注入 Dispatcher，Core 不持有注册表/队列 |
| 目标 | 配对 F16/C3 提示词 | Core 7→5；全系统 F15 冻结的 17→15 |

## 3. 批次 00 交付物（历史）

- `docs/prd/PRD-C3-hook-surface-convergence.md`：C3 PRD（现已验收）。
- `docs/TODO.yaml`：C3 阶段（现已 `done / 已验收`），依赖 C2。
- 本执行报告：记录事实、边界和后续输入。
- 配对迁移矩阵与共同 AC：见 `E:\ftre\docs\execution\matrices\F16-C3-hook-migration-matrix.md`。

## 4. 后续批次输入（已完成）

1. 用户授权目标名称、破坏性改名和无兼容 alias 策略。
2. 批次 01 完成两仓消费者、行为、版本和构建基线。
3. C3.2/C3.3 完成 Core 协议迁移，C3.4 完成全量验证和 wheel。
4. 配对 F16 完成 Host/Package 迁移、洁净安装、生命周期和 Gateway smoke。

## 5. 变更记录

| 日期 | 内容 |
|---|---|
| 2026-08-24 | 建立 C3 批次 00 报告壳，等待 PRD 评审 |
| 2026-08-24 | 完成 C3.1–C3.4、版本 0.2.0、wheel 和跨仓验收 | Core 新 Hook 面已交付给 ftre F16 |

## 6. refactor-cleanup-audit 审计记录

### 范围与基线

- 仓库：`E:\ftre-agent-core`；分支：`feature/C3-hook-surface-convergence`。
- 早期审计批次只修改文档和清理危险调试测试；随后 C3 生产协议按本报告第 7 节完成。
- C3 当前状态为 `done / 已验收`；未修改客户端、`E:\cordis-py`，未 push/merge/release。

### Owner 与依赖证据

| 检查项 | 证据 | 结论 |
|---|---|---|
| Core Hook Owner | `src/ftre_agent_core/hooks.py` | HookSpec、DTO、默认行为由 Core 唯一拥有 |
| Tool 调用 Owner | `src/ftre_agent_core/agent/runner/tool_handler.py` | Core 私有执行器负责归一化和取消，Host 只能注入 Dispatcher |
| Stop 调用 Owner | `src/ftre_agent_core/agent/runner/_execute_acting.py` | Core 解释 StopTurn/ContinueTurn，不持有宿主状态 |
| 反向依赖 | `src/ftre_agent_core` import 扫描 | 未发现对 ftre、Cordis、Inbox、Compaction 的生产 import |

### 迁移前旧协议引用基线

批次 01 的只读扫描确认旧协议在迁移前被实现和回归测试真实引用：

| 旧协议 | 活动引用文件数 | 清理批次 |
|---|---:|---|
| `tools/pre-execute` | 2 | C3.2 |
| `tools/execute` | 3 | C3.2 |
| `tools/post-execute` | 2 | C3.2 |
| `tools/result` | 2 | C3.2 |
| `agent/turn-stopping` | 2 | C3.3 |

这些是当时的 Core 公共导出、ToolHandler 和状态机真实契约；消费者冻结后已在 C3.2/C3.3
删除，当前活动源码/测试已无这些引用。

### 生命周期、测试与工程卫生

- `python -m pytest -q` → **238 passed**。
- `python -m pytest -q src/tests` → **51 passed**。
- `python -m ruff check . --no-cache` → **通过**；第一次缓存模式因 Windows 临时文件权限失败，未把该失败隐藏，改用无缓存模式复核通过。
- `git diff --check` → 通过。
- 测试后清理：`__pycache__`、`.pyc`、`.pytest_cache`、`.ruff_cache` 均为 **0**；
  空 `.ftre/skills` 已移除，保留含项目运行配置的 `.ftre/mcp.json`。
- 未删除 `.git` 内部目录、用户数据或构建依赖。

### 本次明确清理项

- 删除 `src/tests/aaa.py`：文件在模块级创建 `ReActAgent`、启动线程并执行真实 LLM，且没有任何 import/测试引用；它不是当前受支持的测试入口，存在误触发真实请求风险。
- 保留 `src/tests` 中其余手工脚本，因其中仍有 51 个可收集测试；后续如要移除，必须另建清理任务并补齐覆盖说明。
- 二次审计修正 `react_runner.py`、Core `AGENTS.md` 和 continuation 测试中的旧 `turn-stopping` 文字，统一为 `stop-decision`；运行时代码与活动测试扫描为零旧协议引用。
- 二次构建重新验证 Core wheel 仍为 47 个文件，无测试、缓存、字节码或旧 Hook 活动代码；报告已同步最终 SHA256。

### 审计结论

C3 批次 00 的文档、静态边界和工程卫生审计通过；后续迁移证据证明 Core 已完成 7→5，
没有保留旧 alias、双 dispatch 或兼容入口。

## 7. 批次 01–04 执行证据

### 批次 01：消费者与行为基线

- Core 当前唯一 Spec 清单：`tool/before`、`tool/after`、`llm/stream`、
  `agent/before-reasoning`、`agent/stop-decision`。
- 迁移前旧协议消费者已冻结：Core 的 `hooks.py`、ToolHandler、ExitExecutor、公共导出、
  `tests/test_direct_hook_pipeline.py`、`tests/test_hooks.py`、`tests/test_react_runner_step_hook.py`
  及 ftre 的 Service 重导出、Contract/Architecture 测试。
- 版本边界：Core `0.1.2 → 0.2.0`；ftre 需使用 `>=0.2.0,<0.3.0`，不提供旧名 alias。

### 批次 02–03：协议迁移

- `tools/pre-execute` → `tool/before`，DTO 改为 `ToolBeforePayload`。
- `tools/post-execute` → `tool/after`，DTO 改为 `ToolAfterPayload`。
- 删除 `tools/execute` around continuation、`ToolExecutePayload`、`TOOLS_EXECUTE_SPEC`；
  ToolHandler 只在 Core 私有 `invoke()` 执行一次真实 Tool。
- 删除 `tools/result` EMIT、`ToolResultPayload` 和 `TOOLS_RESULT_SPEC`。
- `agent/turn-stopping` → `agent/stop-decision`，DTO 改为 `StopDecisionPayload`；
  `StopTurn`、`ContinueTurn`、continuation 预算和自然停止条件保持不变。
- Core 源码、公共导出和活动测试中的旧符号/旧字符串为零；历史 PRD/迁移报告中的说明保留为历史证据。

### 批次 04：验证与发行候选

- Core `python -m pytest -q`：**238 passed**。
- Core `python -m pytest -q src/tests`：**51 passed**。
- Core `python -m ruff check . --no-cache`：通过。
- 构建：`ftre_agent_core-0.2.0-py3-none-any.whl`，47 个文件；
  SHA256 `e9dccea258c5632389b2e1546c7056fa12447065436203083ed4bf0709ecd7d1`。
- wheel 解包扫描：无 `tests/`、`.pyc`、缓存和旧 Hook 引用；`pyproject.toml` 已限制
  setuptools 只打包 `ftre_agent_core*`，避免把 `src/tests` 手工脚本带入发行物。

### C3 状态

Core C3 的 FR/AC 已按以上证据完成；配对 F16 的洁净安装、Package 生命周期、直接 Agent Turn
和 Gateway smoke 均已通过。
