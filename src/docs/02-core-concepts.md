# 核心概念

## ReAct 循环

```
用户输入 → [LLM 思考 → 调用工具 → 观察结果] × N → 最终回复
```

`ReActAgent` 是用户接口，`ReActRunner` 驱动 LLM 与工具，`AgentState.context`
保存可序列化的 `Msg`；`MessageContext` 负责转换和更新这些消息。

## 实时 Event 与消息 Msg

两种模型职责不同：

- `AgentStreamEvent`：细粒度实时流、取消和 trace。
- `Msg`：一条可持久化、可恢复上下文的聚合消息快照。

事件采用扁平 Pydantic 模型。文本、思考、工具参数和工具结果均使用
start/delta/end 生命周期；一次回复以 `REPLY_START` 开始、以 `REPLY_END`
结束。详见 [事件文档](./08-events.md)。

`Msg.append_event()` 按 `reply_id` 聚合这些事件。持久化层应在
`REPLY_END` 后保存一条 Msg，不应保存每个 delta。

## 执行流程

```
run(messages)
  ├── 写入 AgentState.context
  ├── REPLY_START
  ├── MODEL_CALL_START
  ├── 内容块 / 工具调用 / 工具结果事件
  ├── MODEL_CALL_END（含 token usage）
  └── REPLY_END（completed / error / interrupted）
```

## 状态

`RunStatus` 包含 `IDLE`、`RUNNING`、`COMPLETED`、`ERROR` 和
`CANCELLED`。结束原因由 `ReplyFinishedReason` 表达。

## 下一步

- [工具系统](./03-tools.md)
- [Memory](./05-memory.md)
