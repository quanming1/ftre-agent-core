# B2 执行报告：LLM Responses 协议边界补充

## 结果

- 状态：已完成
- 范围：Responses 原始 Output Item 捕获与重放、`status`/`content` 请求字段隔离、Vision
  `input_image` 归一化、Core/Host metadata 传递。
- 未修改：客户端、运行中的 Gateway、Session 文件数据和外部供应商配置。

## 实现证据

| 语义 | 代码位置 | 结果 |
|---|---|---|
| 原始 Output Item 捕获 | `src/ftre_agent_core/llm/adapters/openai_responses.py` | `ResponseOutputItemDoneEvent.item` 转为 JSON-safe `response_metadata.output_items` |
| 返回态字段隔离 | `src/ftre_agent_core/llm/adapters/openai_responses.py` | 持久化保留完整快照；GPT/未知模型请求重放只允许 `id`/`summary`/`encrypted_content`，移除 `status`/`content` |
| 旧会话降级 | 同上 | GPT/未知模型缺少可重放字段时记录 warning 并省略 reasoning；DeepSeek 保留明确的旧 thinking 兼容路径 |
| Core 持久化桥 | `event/_event.py`、`message/_msg.py` | `ModelCallEndEvent.response_metadata` 写入 `responses_output_item_groups` |
| 下一轮 Core 重放 | `message_context.py` | Msg metadata 转回 `responses_output_items`，不进入可见 content |
| ftre Host 重放 | `E:\ftre\src\ftre\services\session\message\converter.py` | Session Msg metadata 在下一次 Agent 输入中保留原始 Item |
| Vision | `openai_responses.py` | 支持 URL、data URL、`file_id` 和可选 `detail`，统一为 `input_image` |

## 验证

```text
Core:
python -m pytest -q
261 passed
python -m ruff check --no-cache src tests
All checks passed

ftre:
python -m pytest -q
567 passed
python -m ruff check --no-cache src tests packages
All checks passed
```

专项回归覆盖：

- `input[n].status` 不再出现在手工构造的 Responses input；
- 新会话 Output Item 经 Event → Msg metadata → Host provider message 往返；
- 旧会话不可重放 reasoning 的省略诊断；
- Responses reasoning input 不携带 `content` 数组；
- `input_text` / `input_image`、Base64 data URL 和 `file_id` 形态；
- Chat Completions 不接收 Responses 私有 metadata。

## Git 状态

- B2 provider-safe reasoning replay 修复已通过 PR #15 合入 `develop`（merge commit
  `1c5716f`，实现提交 `4118167`）。
- 当前发布候选从 `develop` 创建 `release/0.2.2`；ftre 工作区的既有审计修改未纳入本次发布。
- 发布仍按远程 PR 合入 `master`，不在本地直接 merge 受保护分支。
