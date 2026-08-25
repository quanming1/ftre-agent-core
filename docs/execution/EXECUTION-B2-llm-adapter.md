# B2 执行报告：LLM Responses 协议边界补充

## 结果

- 状态：已完成
- 范围：Responses 原始 Output Item 捕获与重放、`status` 请求字段隔离、Vision
  `input_image` 归一化、Core/Host metadata 传递。
- 未修改：客户端、运行中的 Gateway、Session 文件数据和外部供应商配置。

## 实现证据

| 语义 | 代码位置 | 结果 |
|---|---|---|
| 原始 Output Item 捕获 | `src/ftre_agent_core/llm/adapters/openai_responses.py` | `ResponseOutputItemDoneEvent.item` 转为 JSON-safe `response_metadata.output_items` |
| 返回态字段隔离 | `src/ftre_agent_core/llm/adapters/openai_responses.py` | 请求重放只允许 reasoning input 字段，明确移除 `status` |
| 旧会话降级 | 同上 | 缺少原始 Item 时记录 warning，使用不含 `status` 的最小 reasoning item |
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
- 旧会话 status-free 降级；
- `input_text` / `input_image`、Base64 data URL 和 `file_id` 形态；
- Chat Completions 不接收 Responses 私有 metadata。

## Git 状态

- Core 当前基线已保存于 `130e7ca`；B2 补充实现待提交于当前 `feature/B2-responses-reasoning-event`。
- Core 仓库此前没有本地或远程 `develop`；已从 `130e7ca` 创建本地 `develop` 分支。
- ftre 当前工作区已在 `feature/F29-llm-stream-fallback` 保存为 `8c4760c`，Host metadata 接线在该分支继续提交。
- 按仓库规范，后续通过远程 PR 合入 `develop`，不直接向受保护分支推送。
