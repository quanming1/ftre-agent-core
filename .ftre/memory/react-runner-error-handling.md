# React Runner 错误处理

> React Loop 的统一错误捕获与分层处理架构。core 层负责错误分类和事件派发，重试逻辑由上层（ai-base）统一控制。

## 核心文件

| 文件 | 职责 |
|------|------|
| `src/ftre_agent_core/agent/runner/react_runner.py` | 主循环错误捕获、事件派发、善后处理 |
| `src/ftre_agent_core/agent/runner/handler/llm_handler.py` | `LLMError.classify()` 异常分类 |
| `src/ftre_agent_core/agent/event.py` | `EventType.RETRY`, `retry_event()`, `EventType.ERROR` |

## 错误捕获链路

### 取消错误 (CancelledError)
```
_loop() → _step() → 网络层
  ↑_______________|
     CancelledError 统一捕获
```

- `_loop()` 是 **CancelledError 唯一捕获点**，执行善后处理
- `_step()` 只捕获保存部分内容，然后 `raise` 重新抛出
- 取消时网络异常在 `_step()` 的 `except Exception` 中识别为 `state.is_cancelled` 并转为 CancelledError

### 普通异常 (Exception)
```
_step() → LLMError.classify() → yield error_event() → done_event()
```

- 所有非取消错误在 `_step()` 的 `except Exception` 中捕获
- 调用 `LLMError.classify(e)` 分类异常类型 → error code
- 直接派发 `ERROR` event，**不自动重试**
- 重试决策由 ai-base 层的 `/chat/retry` 路由控制

## 错误分类

```python
LLMError.classify(e)  # 映射原始异常到标准错误码
```

**错误码定义**（`src/ftre_agent_core/agent/runner/handler/llm/types.py`）：

| 错误码 | 说明 | 是否可重试（上层决策参考） |
|--------|------|------------------------|
| `timeout` | LLM 请求超时 | ✅ |
| `network` | 网络连接问题 | ✅ |
| `api_error` | API 服务端错误 | ✅ |
| `rate_limit` | 请求频率限制 | ✅ |
| `unknown` | 未知错误 | ✅ |
| `auth_error` | 认证失败 | ❌ |
| `bad_request` | 请求参数错误 | ❌ |
| `content_filter` | 内容被过滤 | ❌ |

## 关键数据结构

`LLMError`:
```python
{
    code: str,       # error_code
    message: str,    # 人类可读错误信息
    classify(e)      # 类方法：从原始异常映射到 LLMError
}
```

`RetryEvent`（保留供上层消费）：
```python
{
    type: EventType.RETRY,
    data: {
        code: str,           # 错误码
        message: str,        # 错误信息
        attempt: int,        # 当前重试次数（1-based）
        max_attempts: int    # 最大重试次数
    }
}
```

`ErrorEvent`:
```python
{
    type: EventType.ERROR,
    data: {
        message: str,        # 错误信息
        code: str            # 错误码
    }
}
```

## 架构设计

### 分层职责

```
┌─────────────────────────────────────┐
│           ai-base 层                │
│  /chat/retry 路由 → 重试决策与控制   │
│  session_node.retry() → 复用逻辑     │
└─────────────────────────────────────┘
                  │
                  ▼ 消费事件
┌─────────────────────────────────────┐
│         ftre-agent-core 层          │
│  错误分类 → 派发 ERROR/RETRY 事件    │
│  不负责重试，只暴露事件供上层决策    │
└─────────────────────────────────────┘
```

### 为什么移除 core 层重试

1. **去重需求**：重试时不应重复创建 user message，需要在 session 层处理
2. **history 管理**：重试时需要从历史中截去最后一轮，避免消息重复
3. **parent_id 复用**：重试产生的新消息需要挂在同一 user message 下
4. **统一控制**：避免多层级（code_agent + runner）双重重试冲突

### 历史实现（已废弃）

早期版本在 `react_runner.py` 中实现了自动重试机制，后因上述原因移除：

```python
# 早期实现（已删除）
RETRY_DELAY = 3  # 固定 3 秒间隔（未使用指数退避）
MAX_RETRIES = 10

# 在 _step() 中捕获可重试错误后递归调用自身
if retry_count < MAX_RETRIES:
    yield retry_event(...)
    time.sleep(RETRY_DELAY)
    yield from self._step()  # 递归重试
```

**关键 Bug 修复**：递归调用 `_step()` 前错误重置 `self._retry_count = 0`，导致每次重试计数永远为 1。正确做法是在**成功完成后**重置计数器。

### 协作流程

```
第一次调用:
  core:run() → LLM 异常 → classify → yield ERROR → yield DONE
     ↓
  ai-base: 显示错误，提供"重试"按钮

用户点击重试:
  ai-base:/chat/retry → 跳过 user message 创建 → 调用 core:run()
     ↓
  core:run() → 正常执行（从历史截断处开始）
```

## 设计决策

- **分层解耦**：core 层只负责错误分类和事件派发，不重试；重试策略由上层统一控制
- **事件暴露**：保留 `RETRY` 事件定义供上层消费，但 core 不再主动派发（上层自行构造）
- **统一善后**：取消逻辑集中在 `_loop()`，避免多处重复处理清理工作
- **异常转义**：网络层异常在取消场景下不直接暴露，转为 CancelledError 保持语义一致
- **分类映射**：所有 LLM 相关错误通过 `classify()` 统一入口，便于维护错误码映射表
- **重试间隔**：早期实现选择固定 3 秒而非指数退避，保证用户可预期的响应时间

## 注意事项

- `ERROR` 消息在 `to_openai_messages()` 中天然被过滤，不影响后续 LLM 上下文
- `RetryEvent` 保留在 event.py 中，供 ai-base 层构造 SSE 推送使用
- core 层不再维护重试计数器，由上层 session_node 控制重试次数
- 重试时不创建新的 user message，复用上一次的 user message（由 ai-base 层保证）
- 早期 core 层重试实现存在计数器重置 bug，如需参考历史实现请注意该问题
