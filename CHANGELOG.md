# 版本变更公告

## [0.2.1] - 2026-08-25

### C4 UserMessage 与 Assistant message_id 边界

- `reply_id` 继续关联整次 Agent Reply，新增 `message_id` 关联具体 AssistantMsg；一次
  run 可以自然形成 `Assistant A → UserMessage → Assistant B`。
- `BeforeReasoningResult` 保持只有 `messages`；正式 `role=user` mapping 自动让下一次
  Reasoning 旋转到新的 Assistant message_id，不增加边界布尔字段。
- Reasoning、Acting、Tool/权限恢复、Msg.append_event 和流式事件统一使用 message_id；
  Core 不持有 Inbox、Session 或队列状态。
- C4 全量 240 tests、ruff、wheel 与临时 venv 洁净安装通过。

### C4 发布说明

- 这是一个需要宿主同步升级的协议版本：事件新增 `message_id`，用于区分同一 Reply
  中的多条 Assistant 消息；旧版 Core 不提供该边界。

## [0.2.0] - 2026-08-24

### C3 Hook 面终局收敛

- Tool Hook 从 `tools/pre-execute`、`tools/execute`、`tools/post-execute`、`tools/result` 收敛为
  `tool/before` 与 `tool/after`；真实 Tool 执行回归 Core 私有执行边界，避免宿主拥有第二个 continuation。
- `agent/turn-stopping` 改名为 `agent/stop-decision`，`StopTurn`/`ContinueTurn` 和 continuation 预算语义保持不变。
- 删除旧 Hook 常量、Spec、DTO、公共导出和测试引用；这是需要 Host 同步升级的破坏性协议变更。
- 版本提升至 `0.2.0`。

### C2 Agent before-reasoning Hook

- 新增 `agent/before-reasoning` typed contract，在每次 ReAct Reasoning（首次、Tool 后、
  continuation 后）调用 LLM 前让宿主贡献普通 message。
- Core 只消费 `BeforeReasoningPayload/Result`，不依赖 ftre、Inbox 或队列模型；ftre-inbox
  可通过同一个 `HookSpec` 在 active Turn 中原子消费 `next-step`。
- 版本提升至 `0.1.2`；Core 238 项测试与 ruff 全量通过。

## [0.1.1] - 2026-08-22

### C1 Agent Core 直接 Hook 协议

- 删除 Core 自持有的 `FtreCoreHookManager`、旧 `ON_*` 输入输出和注册入口。
- 新增无状态 `HookDispatcher`、共享 `HookSpec`，以及 Tool、LLM stream、turn-stopping typed contracts。
- ReAct Core 直接在 ToolHandler、ReasoningExecutor 和 ExitExecutor 注入宿主 Dispatcher；支持有界 `ContinueTurn`。
- Core 独立回归：234 项 pytest、ruff 全量通过；ftre 侧由 Cordis HookRuntime 作为唯一调度实现。

### B2 LLM 协议适配层（DSH StreamChunk 协议）

**重构**：`llm/completion.py`（764 行单类双 if 分支）拆分为协议适配层：

- `events.py`：StreamChunk 七种 chunk（block-start / text-delta / reasoning-delta / tool-call-delta / block-end / usage / finish，DSH 协议的 Python/dataclass 形态）+ FinishReason（stop / tool-calls / max-tokens / error / aborted，error/aborted 带 failure）
- `block_assembler.py`：BlockAssembler——按 index 组装交错 delta 为完整 ContentBlock，配对校验（fail-fast）
- `errors.py`：LLMError + classify（从 completion.py 迁出，逻辑等价）
- `base.py`：LLMAdapter ABC + OpenAIAdapterBase 共享骨架（异常统一收敛为终止性 error/aborted finish，消费方永不面对裸异常）
- `registry.py`：协议注册表 + `create_llm_handler(api_type)` 工厂（未知协议抛 INVALID_API_TYPE）——协议是数据不是代码，新协议接入 = adapters/ 加文件 + 注册表一行
- `adapters/openai_completions.py` / `adapters/openai_responses.py`：双协议适配器（responses 路径支持 reasoning effort 透传）
- `wire/normalize.py`：协议共用消息归一化

**移除**：`LLMHandler` 类与 LLMEvent 家族（TextDelta/ReasoningDelta/ToolInputDelta/ToolCall/StepFinish）——消费方（react_runner / compact_manager / title_gen / fake_llm）一步到位迁移，无兼容层。

**语义增强**：畸形 finish_reason（如 Muse 的 null）→ 有内容产出时宽容映射 stop、无产出时 error finish（不再静默）。

**测试**：基线 168 → 201 全过（新增契约测试 14 + registry 5 + 适配器 13）。

## 变更日期：2026-04-10

---

## 1. LiteLLM 日志优化

**问题**：LiteLLM 默认 DEBUG 日志在生产环境中会导致 2-5 秒的额外延迟。

**改动**：
```python
# src/ftre_agent_core/__init__.py
os.environ.setdefault("LITELLM_LOG", "WARNING")
```

**影响**：关闭 DEBUG/INFO 日志，仅保留 WARNING 及以上级别。

---

## 2. LLM 调用重试机制

**新增 Event**：`RETRY`

```python
# Event 结构
{
    "type": "retry",
    "data": {
        "code": "timeout",           # 错误码
        "message": "请求超时",        # 错误信息
        "attempt": 1,                # 当前第几次重试
        "max_attempts": 3            # 最大重试次数
    }
}
```

**可重试错误类型**：

| 错误码 | 说明 | 可重试 |
|--------|------|--------|
| `timeout` | 请求超时 | ✅ |
| `network` | 网络连接失败 | ✅ |
| `api_error` | API 服务端错误 | ✅ |
| `rate_limit` | 频率超限 | ✅ |
| `unknown` | 未知错误 | ✅ |
| `auth_error` | 认证失败 | ❌ |
| `bad_request` | 请求无效 | ❌ |
| `content_filter` | 内容审核未通过 | ❌ |

**配置参数**：

```python
from ftre_agent_core.agent import ReActAgent

agent = ReActAgent(
    model="openai/gpt-4",
    api_key="xxx",
    max_retries=3,  # 默认重试 3 次
)
```

**重试行为**：
- 固定 3 秒间隔
- 重试时保存已输出的部分内容
- 外部可监听 `RETRY` event 获取重试进度

**使用示例**：

```python
for event in agent.run("你好"):
    if event["type"] == "retry":
        print(f"重试中 ({event['data']['attempt']}/{event['data']['max_attempts']})")
        print(f"错误: {event['data']['code']} - {event['data']['message']}")
```

---

## 3. 涉及文件

| 文件 | 变更类型 |
|------|----------|
| `src/ftre_agent_core/__init__.py` | 新增 |
| `src/ftre_agent_core/agent/event.py` | 修改 |
| `src/ftre_agent_core/agent/react.py` | 修改 |
| `src/ftre_agent_core/agent/runner/react_runner.py` | 修改 |
| `src/tests/test_retry.py` | 新增 |

---

## 兼容性说明

- `max_retries` 有默认值（3），向后兼容
- 新增的 `RETRY` Event 不影响现有代码
- 如需禁用重试，设置 `max_retries=0`
