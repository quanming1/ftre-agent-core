# 版本变更公告

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
