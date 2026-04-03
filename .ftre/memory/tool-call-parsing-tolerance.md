# Tool Call JSON 解析容错机制

> 当 LLM streaming 响应被截断导致 tool call JSON 不完整时，提供自动重试能力，避免因 JSONDecodeError 中止执行

## 核心文件

| 文件 | 职责 |
|------|------|
| `src/ftre_agent_core/agent/runner/handler/tool_handler.py` | parse_tool_call 方法在 JSON 解析失败时返回 `(call_id, tool_name, None)` 而非抛异常 |
| `src/ftre_agent_core/agent/runner/react_runner.py` | _handle_tool_calls 检查 arguments=None 并构造 [PARSE_ERROR] 结果写入 memory |
| `src/ftre_agent_core/agent/runner/handler/interrupt_handler.py` | resume_tool_calls 对当前和剩余 tool_calls 都处理解析失败情况 |

## 业务流程

### JSON 截断容错流程
tool_handler:parse_tool_call → (arguments=None 标识失败) → react_runner:_handle_tool_calls → 构造错误消息 → 写入 memory → LLM 下一轮看到错误后自动重试

## 设计决策

- **利用 ReAct 循环自然机制**：不抛异常，而是返回错误结果让 LLM 看到失败后自动重试，对架构侵入最小
- **统一返回值类型**：始终返回 `(call_id, tool_name, arguments)` 元组，用 `arguments=None` 标识解析失败
- **完整错误上下文**：错误消息包含字节数、具体解析错误，帮助 LLM 理解问题本质

## 注意事项

- 所有调用 parse_tool_call 的地方（react_runner 和 interrupt_handler）都必须检查 arguments=None
- 错误结果通过 tool_result_event 写入 memory，确保上下文完整性
- 适用于所有 JSON 格式错误场景（截断、畸形 JSON 等），不仅限于 streaming 截断
- write 工具的参数通常 >100 字节，34 字节明显异常，可作为早期检测信号