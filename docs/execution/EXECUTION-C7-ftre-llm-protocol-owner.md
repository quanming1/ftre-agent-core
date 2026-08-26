# C7 执行报告：ftre-llm 协议 Owner 收敛

## 范围

本批只修改 `E:\ftre-agent-core` 的 Core Runner/Hook 协议入口，并同步 `E:\ftre`
的 LLM Service 接线。没有修改客户端、Provider 账号或外部仓库。

## 已完成

- `ftre_llm.events` 成为 StreamChunk 和 ToolCall 的唯一运行时实现；删除 Core 的
  `ftre_agent_core.llm.events` 兼容模块，Core 内部直接从 `ftre_llm.events` 导入。
- Core `LLM_STREAM_SPEC` 使用 `ftre_llm.contracts.LlmStreamPayload`，ftre Host
  重导出同一 Spec；`LlmServiceAdapter` 只组装 `LlmRequest`，不复制流事件，且
  关闭 Service 内层重复派发。
- Core Runner 支持 `ReActAgent(..., llm=...)` 构造期注入；`set_llm()` 校验
  `stream/cancel` 并拒绝 in-flight 替换。ftre 不再写 `runner._llm`，解决统一
  Service 尚未接管前就初始化空凭据 OpenAI 客户端的问题。
- Retry 统一由 Core `llm/error` 决策；Recovery Plugin 返回的
  `LLMErrorDecision` 会实际控制 retry/stop，LlmService 不再维护无人消费的
  平行错误 Hook。
- `ConfigService.resolve_llm()` 快照补齐 `context_window`、`vision` 和
  `reasoning_effort_values`。

## 验证

在设置 `PYTHONPATH` 指向 Core、ftre-llm 和 ftre 源码目录后：

| 仓库 | pytest | ruff | diff check |
|---|---:|---|---|
| `E:\ftre-agent-core` | **265 passed** | 通过 | 通过 |
| `E:\ftre` | **615 passed** | 通过 | 通过 |

额外身份断言确认：

```text
ftre_agent_core.llm.TextDeltaChunk is ftre_llm.events.TextDeltaChunk -> True
ftre.services.llm.hooks.LLM_STREAM_SPEC is ftre_agent_core.hooks.LLM_STREAM_SPEC -> True
```

Core 与 ftre wheel 均已完成 `python -m build --wheel --no-isolation` 构建检查。

## 当前状态

- Core 分支：`feature/C7-ftre-llm-protocol-owner`
- ftre 分支：`feature/F30-llm-service-package`
- 当前未执行 commit/push；工作区还包含执行前已有的 F14/F30 等修改，未做清理性
  覆盖或回滚。

## 2026-08-26 重构收尾审计

- Owner：`ftre_llm.events` 是唯一 StreamChunk 实现；Core 的旧
  `ftre_agent_core.llm.events` 已删除，Core 顶层仅重新公开同一对象，不保留旧模块入口。
- 接线：ftre 生产代码不存在 `runner._llm`、`core_bridge` 或 Host→Core 私有导入；
  `llm/stream` 与 `llm/error` 只保留一份运行时 Spec。
- 生命周期：Runner 注入要求 `stream/cancel`，in-flight 替换拒绝；Provider 注册、
  Hook 注册和 LLM Service 关闭均有可逆句柄，已有生命周期测试覆盖。
- 静态审计：两仓生产源码未命中退役入口；ftre/ftre-llm vulture 高置信度扫描无结果。
  Core 仍有少量既有测试替身的 unreachable-yield/参数告警，不属于生产死代码。
- 最终门禁：Core `265 passed`、ftre `615 passed`；两仓 ruff 与 diff check 通过。
- 生成物盘点：测试后存在 Core `__pycache__` 13 个、ftre `__pycache__` 53 个，以及
  `E:\ftre-agent-core\build`、`src\ftre_agent_core.egg-info`、
  `E:\ftre\packages\ftre-llm\build`、`src\ftre_llm.egg-info`。这些路径均已解析且
  不在 Git 变更中；当前执行环境拒绝递归删除命令，因此未冒险触碰它们，需在本地
  手工删除后再做最终工作区卫生验收。
