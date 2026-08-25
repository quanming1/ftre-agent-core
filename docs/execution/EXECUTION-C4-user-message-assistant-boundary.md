# C4 执行报告：UserMessage 与 Assistant `message_id` 边界

## 结果

- 状态：已完成
- 分支：`feature/C4-user-message-boundary`
- 范围：Core Message/Event/Runner/Hook 契约、测试和文档；未引入 ftre、Inbox、Cordis、Session 或 Channel 依赖。

## 实现证据

| 语义 | 代码位置 | 结果 |
|---|---|---|
| 一次 run 的运行坐标与消息坐标分离 | `src/ftre_agent_core/agent/runner/_state.py`、`react_runner.py` | `reply_id` 全程稳定，`message_id` 在正式 UserMessage 边界旋转 |
| Hook 仍只有 `messages` | `src/ftre_agent_core/hooks.py` | 没有增加 `start_new_assistant` 或第二个 Hook |
| 事件携带 `message_id` | `src/ftre_agent_core/event/_event.py`、`react_runner.py` | Runner 统一在出口补齐，保留 `reply_id` |
| Tool/权限结果归属当前 Assistant | `message_context.py`、`agent/runner/_execute_acting.py` | ToolCall/ToolResult 使用当前 `message_id`，权限恢复从 ToolCall 所属 Msg 恢复 |
| Msg 按消息坐标校验 | `src/ftre_agent_core/message/_msg.py` | `message_id` 优先，旧事件才回退 `reply_id` |

## 验证

```text
python -m pytest -q
240 passed in 61.89s

python -m ruff check src tests --no-cache
All checks passed

python -m build --wheel --outdir <临时目录>
ftre_agent_core-0.2.0-py3-none-any.whl 构建成功

洁净 venv 安装 wheel（补齐公开运行依赖 pydantic/openai）
clean wheel import ok
```

专项回归覆盖：Tool→Steer→Reasoning 的 A→User→B、唯一 Msg.id、事件双坐标、LLM
重试、取消、ContinueTurn、Tool 权限暂停/恢复、重复 User mapping 幂等。

## 收尾审计

- `src/`、`tests/` 中没有 `reply_segment`、旧 split helper 或 `reply_id:segment:*` 生产/测试引用。
- 已删除本次验证生成的 Core `__pycache__`、`.pytest_cache`、`.ruff_cache`、`build` 和临时 wheel/venv；复核数量为 0。
- 工作树仍保留本批未提交的源码、PRD、TODO 和执行报告修改，未触碰用户其它修改；未执行 commit/push。

## 交付边界

Core 只定义消息坐标和自动边界，不持有 Inbox/Session，也不负责持久化。ftre F23
负责安全边界的落库与 claim，Desktop B4 负责按服务端 `message_id` 投影；三端通过
同一个 `reply_id`/`message_id` 协议衔接。
