# 事件流（AgentEvent）

ReAct 循环的所有输出通过事件流返回。每个事件是一个 dict：

```python
{"type": EventType, "data": {...}}
```

`type` 是 `EventType` 枚举（继承 `str`），`data` 是对应的 TypedDict。

## 事件类型总览

| EventType | 产出时机 | 持久/临时 |
|-----------|---------|-----------|
| `MESSAGE` | LLM 流式输出每个 token | 临时（增量） |
| `MESSAGE_COMPLETE` | 一段文本输出结束 | 持久（最终值） |
| `REASONING` | LLM 输出推理过程（DeepSeek R1 等） | 临时（增量） |
| `TOOL_CALL_STREAMING` | LLM 流式输出工具调用参数 | 临时（增量） |
| `TOOL_CALL` | 工具开始执行 | 持久 |
| `TOOL_RESULT` | 工具执行完成 | 持久 |
| `TOOL_CANCEL_REQUESTED` | 工具取消请求已发出 | 持久 |
| `TOOL_CANCELLED` | 工具已被取消 | 持久 |
| `TOOL_TIMED_OUT` | 工具执行超时 | 持久 |
| `USAGE_UPDATE` | LLM 调用返回 token 用量 | 临时 |
| `ERROR` | LLM 调用失败 | 持久 |
| `RETRY` | LLM 调用失败正在重试 | 临时 |
| `DONE` | 整个 ReAct 循环结束 | 持久（终止信号） |

## 事件产出顺序

一次完整的 ReAct 循环（用户发一条消息到最终回复）的事件序列：

### 场景 1：纯文本回复（无工具调用）

```
MESSAGE(content="你")
MESSAGE(content="好")
MESSAGE(content="！")
MESSAGE_COMPLETE(content="你好！")
DONE(success=true, reason="completed")
```

### 场景 2：调用工具后回复

```
# 第 1 次迭代：LLM 决定调工具
TOOL_CALL_STREAMING(tool_calls=[{id, name, arguments_delta}])  × N（增量）
MESSAGE_COMPLETE(content="")  ← 如果 LLM 在 tool_call 前有文本
USAGE_UPDATE(usage={...})
TOOL_CALL(id="call_1", name="bash", arguments={command: "ls"})
TOOL_RESULT(id="call_1", name="bash", result="file1.py\nfile2.py", status="completed")

# 第 2 次迭代：LLM 根据工具结果生成回复
MESSAGE(content="当前")
MESSAGE(content="目录")
MESSAGE(content="有...")
MESSAGE_COMPLETE(content="当前目录有 file1.py 和 file2.py")
DONE(success=true, reason="completed")
```

### 场景 3：多工具并行

```
TOOL_CALL(id="call_1", name="read", arguments={path: "a.py"})
TOOL_CALL(id="call_2", name="read", arguments={path: "b.py"})
TOOL_RESULT(id="call_1", name="read", result="...", status="completed")
TOOL_RESULT(id="call_2", name="read", result="...", status="completed")
```

工具并行执行，TOOL_RESULT 的顺序取决于哪个先完成。

### 场景 4：用户取消

```
MESSAGE(content="让我")
MESSAGE(content="来...")
# ← 用户取消
MESSAGE_COMPLETE(content="让我来...")  ← 已输出的部分仍会 complete
DONE(success=false, reason="cancelled")
```

### 场景 5：工具执行中取消

```
TOOL_CALL(id="call_1", name="bash", arguments={...})
# ← 用户取消
TOOL_RESULT(id="call_1", name="bash", result="[用户取消]", status="cancelled")
DONE(success=false, reason="cancelled")
```

### 场景 6：达到最大迭代次数

```
... (多轮工具调用)
DONE(success=false, reason="max_iterations")
```

### 场景 7：LLM 调用失败

```
ERROR(message="请求超时: ...", code="timeout")
DONE(success=false, reason="error")
```

### 场景 8：带推理过程

```
REASONING(content="用户问的是天气...")
REASONING(content="我需要调用工具...")
TOOL_CALL_STREAMING(...)
...
```

REASONING 事件在 MESSAGE 之前或之间出现，取决于模型。

---

## 各事件详细规范

### MESSAGE

LLM 流式输出的文本增量。每个 token 一个事件。

```python
{
    "type": EventType.MESSAGE,  # "message"
    "data": {
        "content": str  # 本次增量文本（不是累积值）
    }
}
```

**注意**：`content` 是增量（delta），不是累积值。消费者需要自行拼接。

### MESSAGE_COMPLETE

一段文本输出结束时产出。`content` 是该段的完整文本。

```python
{
    "type": EventType.MESSAGE_COMPLETE,  # "message_complete"
    "data": {
        "content": str  # 该段完整文本
    }
}
```

**产出时机**：
1. LLM 纯文本回复结束（无 tool_call）→ 产出 MESSAGE_COMPLETE + DONE
2. LLM 回复含 tool_call → 先产出 MESSAGE_COMPLETE（如果有前置文本），再进入工具执行
3. 用户取消时，已输出的部分也会产出 MESSAGE_COMPLETE

**与 MESSAGE 的关系**：
- MESSAGE 是增量流，MESSAGE_COMPLETE 是最终确认
- 消费者可以只听 MESSAGE_COMPLETE 忽略 MESSAGE（非流式场景）
- 也可以用 MESSAGE 做实时渲染，MESSAGE_COMPLETE 做最终校正

### REASONING

LLM 输出的推理过程（如 DeepSeek R1 的 `reasoning_content`）。增量。

```python
{
    "type": EventType.REASONING,  # "reasoning"
    "data": {
        "content": str  # 推理文本增量
    }
}
```

**注意**：不是所有模型都产出 REASONING。只有支持 `reasoning_content` 字段的模型才会。

### TOOL_CALL_STREAMING

LLM 流式输出工具调用参数的增量。

```python
{
    "type": EventType.TOOL_CALL_STREAMING,  # "tool_call_streaming"
    "data": {
        "tool_calls": [
            {
                "index": int,              # 工具调用索引
                "id": str | None,          # call_id（首次出现时有值）
                "name": str | None,        # 工具名（首次出现时有值）
                "arguments_delta": str | None  # JSON 参数片段
            }
        ]
    }
}
```

**与 TOOL_CALL 的关系**：
- TOOL_CALL_STREAMING 是参数的流式增量（用于 UI 实时展示"正在输入参数"）
- TOOL_CALL 是参数解析完成后、工具开始执行时产出
- 消费者可以忽略 TOOL_CALL_STREAMING，只听 TOOL_CALL

### TOOL_CALL

工具开始执行。参数已完整解析。

```python
{
    "type": EventType.TOOL_CALL,  # "tool_call"
    "data": {
        "id": str,                    # 唯一 call_id
        "name": str,                  # 工具名
        "arguments": dict[str, Any]   # 已解析的参数（dict，不是 JSON 字符串）
    }
}
```

**注意**：如果 LLM 返回的 JSON 参数解析失败，不会产出 TOOL_CALL，而是直接产出 TOOL_RESULT(error="[PARSE_ERROR]...")。

### TOOL_RESULT

工具执行完成（成功、失败或取消）。

```python
{
    "type": EventType.TOOL_RESULT,  # "tool_result"
    "data": {
        "id": str,                    # 对应的 call_id
        "name": str,                  # 工具名
        "result": str,                # 执行结果（字符串）
        "error": str | None,          # 错误信息（成功时为 None）
        "status": str,                # "completed" | "failed" | "cancelled"
        "error_code": str | None,     # 错误码（可选）
        "metadata": dict | None       # 附加元数据（可选）
    }
}
```

**status 取值**：
- `"completed"` — 正常完成
- `"failed"` — 执行出错（error 字段有值）
- `"cancelled"` — 被用户取消

### TOOL_CANCEL_REQUESTED

工具取消请求已发出（信号已设置，但工具可能还在执行）。

```python
{
    "type": EventType.TOOL_CANCEL_REQUESTED,  # "tool_cancel_requested"
    "data": {
        "id": str,
        "name": str,
        "reason": str,           # "user_cancelled"
        "status": "cancelling",
        "error_code": str | None,
        "result_status": str | None
    }
}
```

### TOOL_CANCELLED

工具已确认取消完成。

```python
{
    "type": EventType.TOOL_CANCELLED,  # "tool_cancelled"
    "data": {
        "id": str,
        "name": str,
        "reason": str,           # "user_cancelled"
        "status": "cancelled",
        "error_code": "cancelled",
        "result_status": "cancelled"
    }
}
```

### TOOL_TIMED_OUT

工具执行超时。

```python
{
    "type": EventType.TOOL_TIMED_OUT,  # "tool_timed_out"
    "data": {
        "id": str,
        "name": str,
        "reason": "timed_out",
        "status": "timed_out",
        "error_code": "timed_out",
        "result_status": "timed_out"
    }
}
```

### USAGE_UPDATE

LLM 调用返回的 token 用量。

```python
{
    "type": EventType.USAGE_UPDATE,  # "usage_update"
    "data": {
        "usage": {
            "prompt_tokens": int,
            "completion_tokens": int,
            ...  # 其他供应商特定字段
        }
    }
}
```

**产出时机**：
- 流式结束时（StreamDelta 带 usage）
- LLMResponse 返回时（tool_call 场景）

### ERROR

LLM 调用失败。

```python
{
    "type": EventType.ERROR,  # "error"
    "data": {
        "message": str,  # 人类可读错误描述
        "code": str      # 错误码
    }
}
```

**错误码**：
| code | 说明 |
|------|------|
| `rate_limit` | 频率超限 |
| `timeout` | 请求超时 |
| `network` | 网络连接失败 |
| `auth_error` | 认证失败 |
| `bad_request` | 请求无效 |
| `content_filter` | 内容审核未通过 |
| `api_error` | API 错误 |
| `unknown` | 未知错误 |

**注意**：ERROR 之后一定跟着 DONE(success=false, reason="error")。

### RETRY

LLM 调用失败正在重试。

```python
{
    "type": EventType.RETRY,  # "retry"
    "data": {
        "code": str,          # 错误码
        "message": str,       # 错误描述
        "attempt": int,       # 当前重试次数
        "max_attempts": int   # 最大重试次数
    }
}
```

### DONE

ReAct 循环结束。**每次 `agent.run()` 调用保证最终产出且仅产出一个 DONE 事件。**

```python
{
    "type": EventType.DONE,  # "done"
    "data": {
        "success": bool,       # 是否成功完成
        "reason": DoneReason,  # 结束原因
        "usage": dict | None   # 累计用量（可选）
    }
}
```

**DoneReason 取值**：
| reason | 说明 |
|--------|------|
| `"completed"` | LLM 给出最终回复，不再调用工具 |
| `"max_iterations"` | 达到最大迭代次数 |
| `"error"` | LLM 调用失败 |
| `"cancelled"` | 用户取消 |

---

## 事件流的保证

1. **DONE 是终止信号** — 每次 `run()` 保证最终产出一个 DONE，之后不再有事件
2. **MESSAGE_COMPLETE 在 DONE 之前** — 如果有文本输出，一定先 MESSAGE_COMPLETE 再 DONE
3. **TOOL_CALL 在 TOOL_RESULT 之前** — 每个 tool_call 一定先 CALL 再 RESULT
4. **ERROR 在 DONE 之前** — 出错时先 ERROR 再 DONE
5. **取消时已输出的内容仍会 complete** — 不会丢失已流式输出的文本

## 消费者实现建议

```python
content_buffer = ""
tool_calls = {}

for event in agent.run("..."):
    t = event["type"]
    d = event["data"]

    if t == EventType.MESSAGE:
        content_buffer += d["content"]
        # 实时渲染

    elif t == EventType.MESSAGE_COMPLETE:
        content_buffer = d["content"]  # 用最终值校正
        # 确认渲染

    elif t == EventType.TOOL_CALL:
        tool_calls[d["id"]] = {"name": d["name"], "args": d["arguments"], "status": "running"}
        # 显示"正在执行..."

    elif t == EventType.TOOL_RESULT:
        tool_calls[d["id"]]["status"] = d["status"]
        tool_calls[d["id"]]["result"] = d["result"]
        # 更新 UI

    elif t == EventType.DONE:
        if d["success"]:
            # 完成
        else:
            # 处理失败原因
```
