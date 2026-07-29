# ReActRunner 状态机重构设计

> 日期：2026-07-28
> 状态：已确认，待写实施计划

## 背景与目标

当前 `react_runner.py`（897 行）将 ReAct 决策逻辑内联在 `_loop → _run_turn → _stream_turn` 三层控制流中。决策条件（是否继续推理、是否执行工具、是否结束）散落在约 300 行的 `_stream_turn` 里，难以测试和推理。

AgentScope 的 `_next_action()` 纯决策函数模式更优雅，但缺少生产环境必需的重试、空响应恢复、ON_STOP Hook 和 Tracing 能力。

本次目标：**将两者优点合并**——引入显式动作类型和纯决策函数，同时完整保留现有生产能力。

## 设计决策摘要

| 决策项 | 选择 |
|---|---|
| 改造强度 | C：彻底重构，API 可不兼容 |
| HITL 暂停/恢复 | 暂不做，预留 `Suspend` 动作扩展点 |
| 取消机制 | A：仅 `asyncio.Task.cancel()`，不引入 CancellationToken |
| Reply 生命周期 | A：一次 `run()` 只产一对 `ReplyStartEvent / ReplyEndEvent` |
| 保留的能力 | LLM 重试、空响应恢复、ON_STOP Hook、Tracing、工具并发、成组写入 Memory、max_iterations |
| 删除的能力 | length 截断自动续写、CancellationToken |
| 结构化输出 | 暂不加入，`Reasoning` 动作预留 `tool_choice` 字段 |
| 并发约束 | 同一 Agent 禁止并发 `run()` |
| 取消 API | 保留 `cancel_nowait()`，内部调用 `Task.cancel()` |

## 架构

### 动作模型

```python
class Reasoning(BaseModel):
    hint: str | None = None
    tool_choice: str | None = None
    force_no_tools: bool = False

class Acting(BaseModel):
    tool_calls: list[ToolCall]

class Exit(BaseModel):
    finished_reason: ReplyFinishedReason
    exit_msg: Msg | None = None
    error: str | None = None
    error_code: str | None = None
```

### TurnResult

```python
@dataclass
class TurnResult:
    text: str
    reasoning: str
    tool_calls: list[ToolCall]
    finish_reason: str
    usage: dict | None = None
    error: LLMError | None = None
```

### ExitOutcome

```python
@dataclass
class ExitOutcome:
    should_continue: bool = False
    continue_hint: str | None = None
```

### 决策函数 `_decide()`

纯函数，不执行 I/O，不 yield 事件。输入上一轮 `TurnResult` 和当前迭代计数，输出下一步动作。

判断优先级：

1. `prev.error` 非空 → `Exit(ERROR)`
2. `prev.tool_calls` 非空 → `Acting(tool_calls)`
3. `prev.text.strip()` 非空且无工具调用 → `Exit(COMPLETED)`
4. 空响应 + `in_finalization` → `Exit(ERROR, "空响应...")`
5. 空响应 + `empty_retries < MAX` → `empty_retries += 1`，`Reasoning()` 重试
6. 空响应 + 重试耗尽 → `in_finalization = True`，`Reasoning(hint=最终化提示, force_no_tools=True)`
7. `iteration >= max_iterations` → `Exit(EXCEED_MAX_ITERS)`
8. 默认 → `Reasoning()`

副作用仅限修改 `empty_retries` 和 `in_finalization` 两个计数器。

### 主循环 `_loop()`

```
prev = None
while iteration < max_iterations:
    check_cancel()
    iteration += 1
    trigger_on_turn_start()

    action = _decide(prev)

    match action:
        Reasoning → prev = _execute_reasoning(action)
        Acting    → _execute_acting(action); prev = None
        Exit      → outcome = _execute_exit(action)
                     if outcome.should_continue:
                         write hint to memory; prev = None; continue
                     return

# 超限
yield _finalize(EXCEED_MAX_ITERS)
```

异常处理：

- `CancelledError` → `yield _finalize(INTERRUPTED)`
- 其他 `BaseException` → `yield _finalize(ERROR)`，re-raise

### `_finalize()`

统一终态写入，所有退出路径经过此函数。设置状态并返回 `ReplyEndEvent`，由调用方 `_loop()` yield：

```python
def _finalize(self, reason) -> ReplyEndEvent:
    self.state.status = STATUS_MAP[reason]
    self.state.done_reason = reason
    return ReplyEndEvent(
        session_id=...,
        reply_id=self.state.reply_id,
        finished_reason=reason,
    )
```

### 执行器

#### `_execute_reasoning(action: Reasoning) -> TurnResult`

职责：LLM 调用 + 流式消费 + 重试 + 产出事件 + 返回 TurnResult。

- `force_no_tools=True` 时 tools 传 None。
- `hint` 非空时先写入 Memory 再调模型，同时 yield `HintBlockEvent`。
- 重试逻辑内聚：`for attempt in range(max_attempts)`，可重试错误等待 `retry_delay` 后重试，产出 `RetryEvent`。
- 流式消费期间每个 chunk 之间 `check_cancel()`。
- 产出事件：`ModelCallStart/End`、`TextBlock*`、`ThinkingBlock*`、`ToolCallStart/End`。
- 异常路径：写入半截文本到 Memory，re-raise（不产 ReplyEndEvent）。
- 正常返回 `TurnResult`；重试耗尽返回 `TurnResult(error=LLMError)`。

#### `_execute_acting(action: Acting) -> None`

职责：工具并发执行 + 成组写入 Memory + 产出事件。

- 复用 `ToolHandler.spawn()` + `gather_results()`。
- 写入顺序：`assistant(tool_calls) → tool(result_1) → tool(result_2) → ...`。
- 工具返回的 hint 延后追加，确保在所有 tool_result 之后。
- 产出事件：`ToolResultStart/Delta/End`、`HintBlockEvent`。
- 异常路径：`drain()` 取消未完成任务后 re-raise。
- 执行完成后不设终态，主循环回到 `_decide(None)`。

#### `_execute_exit(action: Exit) -> ExitOutcome`

职责：ON_STOP Hook + 产出 ReplyEndEvent + 设置终态。

- 仅 `COMPLETED` 触发 `ON_STOP`；`ERROR` 和 `EXCEED_MAX_ITERS` 直接退出。
- `ON_STOP` 返回 `block` 时返回 `ExitOutcome(should_continue=True, continue_hint=...)`，不产 `ReplyEndEvent`。
- `ON_STOP` 返回 `allow` 或无 hook 时：yield `ReplyEndEvent`，`_finalize()`，返回 `ExitOutcome()`。

### 取消协议

```python
async def run(self, message, runtime_context=None):
    if self._run_task is not None and not self._run_task.done():
        raise RuntimeError("Agent is already running")
    self._run_task = asyncio.current_task()
    try:
        ...
    finally:
        self._run_task = None

def cancel_nowait(self) -> None:
    if self._run_task is not None and not self._run_task.done():
        self._run_task.cancel()
```

取消传播路径：

```
cancel_nowait() → task.cancel()
    → CancelledError 注入到当前 await 点
        → _execute_reasoning 的 LLM stream / retry sleep
        或 _execute_acting 的 gather_results
    → _loop except CancelledError
        → _finalize(INTERRUPTED)
    → run() finally
        → _run_task = None
```

- `_check_cancel()` 仅在循环顶部和 LLM chunk 之间作为快速出口。
- `CancelledError` 是唯一取消信号。
- `_finalize(INTERRUPTED)` 后不 re-raise，取消作为 `INTERRUPTED` 结果正常结束。

### 事件生命周期

```
ReplyStartEvent           ← run() 入口，只产一次
  ├── ModelCallStartEvent ← _execute_reasoning 每次 LLM 调用
  │     ├── TextBlockStart/Delta/End
  │     ├── ThinkingBlockStart/Delta/End
  │     ├── ToolCallStart/Delta/End
  │     └── ModelCallEndEvent
  ├── RetryEvent           ← _execute_reasoning 重试时
  ├── ToolResultStart/Delta/End ← _execute_acting
  ├── HintBlockEvent      ← _execute_reasoning (空响应提示) / _execute_acting (工具 hint) / _execute_exit (ON_STOP 续写)
  └── ... (多轮循环)
ReplyEndEvent             ← _execute_exit (正常) / _finalize (取消/异常/超限)
```

`ReplyEndEvent` 产出规则：
- `_execute_exit()` 正常退出：yield `ReplyEndEvent(COMPLETED)`
- `_finalize(INTERRUPTED)`：yield `ReplyEndEvent(INTERRUPTED)`
- `_finalize(ERROR)`：yield `ReplyEndEvent(ERROR)`
- 主循环超限：yield `ReplyEndEvent(EXCEED_MAX_ITERS)`
- 互斥：`_execute_exit()` 正常返回后 `_finalize()` 不会再产。

### Memory 写入策略

| 写入点 | 时机 | 写入内容 |
|---|---|---|
| `run()` 入口 | 消息进入 | 用户消息 / 列表消息 |
| `_execute_reasoning` | LLM stream 结束后 | `assistant(text + tool_calls)` |
| `_execute_reasoning` 异常 | 半截文本 | `assistant(text)` (如果有) |
| `_execute_acting` | 工具 gather 完成后 | `tool(result_1) → tool(result_2) → ...` |
| `_execute_reasoning` hint | 调 LLM 前 | `user(hint)` |
| `_execute_exit` ON_STOP block | 续写 | `user(continue_hint)` |

### Tracing 集成

| Span | 创建位置 | 类型 | 收尾位置 |
|---|---|---|---|
| 根 span | `run()` | `AGENT` | `run()` finally |
| LLM 子 span | `_execute_reasoning` 每次 attempt | `LLM` | `_execute_reasoning` 内 |
| 工具子 span | `_execute_acting` 每个 tool call | `TOOL` | `ToolHandler._run_one_traced` 内 |

根 span 收尾统一在 `run()` 的 `finally`，依据 `RunState.status` 映射 trace 状态。

### RunState 精简

```python
@dataclass
class RunState:
    # 生命周期
    status: RunStatus = RunStatus.IDLE
    iteration: int = 0
    done_reason: ReplyFinishedReason | None = None
    error: str | None = None
    error_code: str | None = None

    # 取消
    _run_task: asyncio.Task | None = field(default=None, repr=False)

    # Tracing
    trace_span: TraceSpan | None = None

    # 运行上下文
    runtime_context: dict = field(default_factory=dict)
    reply_id: str = ""
    turn_id: str = ""

    # 空响应恢复（仅 _decide 读写）
    empty_retries: int = 0
    in_finalization: bool = False

    # token 统计
    token_usage: dict = field(default_factory=lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "llm_calls": 0,
    })
```

删除的字段：
- `cancel_token` — 不再用 CancellationToken
- `force_no_tools_once` — 移入 `Reasoning.force_no_tools`
- `finalization_retrying` — 重命名为 `in_finalization`
- `reply_msg` — 改为 `run()` 内局部变量
- `_turn_start_ts` / `_first_token_logged` / `_ttft_ms` — 移入 `_execute_reasoning` 局部变量

### 模块结构

```
src/ftre_agent_core/agent/runner/
    __init__.py
    _actions.py              # Reasoning, Acting, Exit, TurnResult, ExitOutcome
    _decide.py               # _decide() 纯决策逻辑
    _execute_reasoning.py    # LLM 调用 + 流式 + 重试
    _execute_acting.py       # 工具并发 + 成组写入
    _execute_exit.py         # ON_STOP + ReplyEnd + 收尾
    _state.py                # RunState, RunStatus, CancelledError
    react_runner.py          # ReActRunner: run() + _loop() + cancel_nowait() + _finalize()
    tool_handler.py          # 保持不变
```

设计原则：
- `_decide.py` 不 import 任何 I/O 模块，可独立单测。
- 每个执行器文件只 import 它需要的依赖。
- `react_runner.py` 只做组装：创建执行器实例、驱动主循环、取消入口。

## 不在本次范围内

- HITL 暂停/恢复（预留 `Suspend` 动作扩展点）
- 结构化输出（`Reasoning.tool_choice` 预留字段）
- length 截断自动续写（已删除）
- CancellationToken（已删除）

## 测试策略

- `_decide()` 纯函数单测：构造各种 `TurnResult` + 迭代计数组合，验证返回的动作类型和副作用。
- 执行器单测：mock LLMHandler / ToolHandler，验证事件产出和 Memory 写入。
- 集成测试：保留并适配 `test_react_runner_continuation.py`，验证取消、重试、空响应恢复、ON_STOP、max_iterations 端到端行为。
