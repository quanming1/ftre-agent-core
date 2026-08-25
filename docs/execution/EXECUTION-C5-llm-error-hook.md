# C5 执行报告：LLM `llm/error` Hook

## 结果

- 状态：已完成
- 范围：Core Hook 契约、Reasoning 失败决策边界、Core 测试、公共导出和发行验证。
- 未修改：ftre、Cordis、客户端、Inbox、Session 协议；未实现 C6 Stream Fallback。

## 实现证据

| 语义 | 代码位置 | 结果 |
|---|---|---|
| 失败事实契约 | `src/ftre_agent_core/hooks.py` | 新增 `LLM_ERROR_SPEC`、`LLMErrorPayload`、`LLMErrorDecision` |
| 触发时机 | `src/ftre_agent_core/agent/runner/_execute_reasoning.py` | 归一化 `LLMError` 后、Core retry/stop 前 dispatch 一次 |
| Core 唯一执行 Owner | 同上 | attempt 上限、RetryEvent、退避、取消、消息重读和收尾仍由 Core 持有 |
| 失败安全 | 同上 | 无监听器/`None`/监听器异常均回到原有分类，Hook 采用 fail-open |
| 公共导出 | `src/ftre_agent_core/__init__.py`、`agent/__init__.py` | ftre 可复用同一 Spec 对象 |

## 验证

```text
python -m pytest -q
250 passed

python -m ruff check src tests
All checks passed

python -m compileall -q src/ftre_agent_core
通过

python -m build --wheel --sdist --outdir <临时目录>
wheel/sdist 构建成功

洁净 venv（--no-index --no-deps）安装 Core wheel
安装成功
```

专项测试覆盖：retry/stop、最后一次 attempt、Hook 异常 fail-open、Payload 边界、错误流和
已有重试回归。C6 的 `llm/stream` attempt 元数据和 fallback 仍保留为后续阶段。

## 收尾

PRD `PRD-C5-llm-error-hook.md` 与 `TODO.yaml` 已同步为已验收；工作树中既有的 B2 及其它
用户修改保持不动，未执行 commit/push，也未重启运行中的后端。

## Refactor Cleanup Audit

- **Owner**：`LLM_ERROR_SPEC`、Payload/Decision 的唯一实现是 Core `hooks.py`；
  `ReasoningExecutor` 是唯一 retry/stop 执行 Owner；Core 没有反向 import ftre。
- **旧入口**：生产代码和测试没有 `llm/attempt-failed`、`llm/retry-decision` 或
  `LLMRecoveryDecision`；旧名称只在本 PRD 的历史变更记录中明确标注为已废弃。
- **生命周期**：Core 不持有 Plugin 注册表或进程状态；HookRuntime/Fiber 的卸载与 in-flight
  屏障由宿主负责，C5 只消费异步 Dispatcher 协议。
- **卫生**：最终清理了两仓源码/测试/文档范围内 81 个缓存或构建目录；Core 剩余数量为
  0，空目录为 0。工作树仍有验收前已存在的未提交修改，未擅自清理。
