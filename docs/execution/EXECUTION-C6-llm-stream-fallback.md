# C6 执行报告：`llm/stream` attempt 元数据

## 结果

- 状态：已完成
- 范围：Core `LLMStreamPayload`、ReasoningExecutor 每次 attempt 接线、版本元数据与跨仓合同。
- 未修改：Core fallback 算法、ConfigService、Agent/Session/Inbox、客户端和 Cordis。

## 实现证据

| 语义 | 代码位置 | 结果 |
|---|---|---|
| Payload 坐标 | `src/ftre_agent_core/hooks.py` | 新增 1-based `attempt`、`max_attempts`，构造时校验边界 |
| 每次重试 dispatch | `src/ftre_agent_core/agent/runner/_execute_reasoning.py` | 每个真实 Core attempt 重新构造 `llm/stream` Payload |
| 默认兼容 | 同上 | 直接构造旧 Payload 默认 `1/1`；无 Hook 时仍直连 Adapter |
| 发行 | `pyproject.toml`、`CHANGELOG.md` | Core 版本提升至 0.2.2，未引入 ftre/Cordis 依赖 |

## 验证

```text
python -m pytest -q
255 passed

python -m pytest -q tests/test_llm_stream_payload.py tests/test_execute_reasoning.py
14 passed

python -m ruff check --no-cache src tests
All checks passed
```

`ftre` 跨仓合同确认 Host 重导出仍指向同一 Core Spec；F29 集成覆盖前置 Retry 不切换、
最后一次坐标和 fallback 成功流。

## 收尾

PRD `PRD-C6-llm-stream-fallback.md` 与 `TODO.yaml` 已同步为已验收。运行中的 Gateway 未被
kill/restart；未执行 commit、push 或发布。

## Refactor Cleanup Audit（2026-08-25）

- **Owner**：`LLMStreamPayload`、`LLM_STREAM_SPEC` 和 attempt 坐标由
  `src/ftre_agent_core/hooks.py` 定义；`ReasoningExecutor._stream()` 只负责构造本次
  Payload；Core Retry、取消屏障、`RetryEvent` 和最终错误仍由
  `src/ftre_agent_core/agent/runner/_execute_reasoning.py` 持有。
- **边界**：对 `src/` 做 AST 反向依赖扫描，结果为 0；Core 没有 import ftre Host、ConfigService
  或 Package 实现。生产代码没有旧 `llm/attempt-failed`、`llm/retry-decision` 或
  `LLMRecoveryDecision` 引用；文档中的旧名字均标注为历史草案/已废弃。
- **生命周期**：Core 不保存 Plugin/Adapter 全局状态；每次 attempt 重新 dispatch，Hook 失败
  仍回到 Core 默认路径。F29 的跨仓集成覆盖了 attempt 坐标和备用流资源关闭。
- **卫生**：清理本轮测试生成的 `.pytest_cache`、`.ruff_cache`、`build` 和根级 `__pycache__`；
  未触碰 `data/sessions.db` 等本地会话数据。最终 diff check 通过。
- **工作树**：当前分支仍包含执行前已存在的 B2/C5 及其他用户修改；本审计未回滚、未提交，
  所以不宣称工作树干净。
