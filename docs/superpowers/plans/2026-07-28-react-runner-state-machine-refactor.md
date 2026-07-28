# ReActRunner 状态机重构 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 ReActRunner 的决策逻辑从三层控制流中提取为纯决策函数 + 显式动作类型 + 独立执行器，同时保留所有现有生产能力。

**Architecture:** 引入 `Reasoning / Acting / Exit` 动作模型和 `_decide()` 纯决策函数；三个执行器各自独立处理 LLM 调用、工具执行、退出收尾；主循环只做 match 分发。取消改为纯 `asyncio.Task.cancel()`。

**Tech Stack:** Python 3.12+, asyncio, pydantic, pytest

## Global Constraints

- 取消机制：仅 `asyncio.Task.cancel()`，不引入新的 CancellationToken 调用（保留 RunState.cancel_token 字段仅供 ToolHandler 兼容，从不主动 cancel）
- Reply 生命周期：一次 `run()` 只产一对 `ReplyStartEvent / ReplyEndEvent`
- 同一 Agent 禁止并发 `run()`
- 保留能力：LLM 重试、空响应恢复、ON_STOP Hook、Tracing、工具并发、成组写入 Memory、max_iterations
- 删除能力：length 截断自动续写
- 模块路径：`src/ftre_agent_core/agent/runner/`
- 测试命令：`cd E:\ftre-agent-core && python -m pytest tests/ -v`
- tool_handler.py 不改动

---

### Task 1: 创建 `_state.py` — RunState / RunStatus / CancelledError

**Files:**
- Create: `src/ftre_agent_core/agent/runner/_state.py`
- Test: `tests/test_state.py`

**Interfaces:**
- Produces: `RunStatus(str, Enum)`, `CancelledError(Exception)`, `RunState(dataclass)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_state.py
"""RunState / RunStatus / CancelledError 单元测试。"""
import asyncio
import pytest
from ftre_agent_core.agent.runner._state import RunState, RunStatus, CancelledError
from ftre_agent_core.types import ReplyFinishedReason


def test_run_status_values():
    assert RunStatus.IDLE == "idle"
    assert RunStatus.RUNNING == "running"
    assert RunStatus.COMPLETED == "completed"
    assert RunStatus.ERROR == "error"
    assert RunStatus.CANCELLED == "cancelled"


def test_cancelled_error_is_exception():
    err = CancelledError("test")
    assert isinstance(err, Exception)


def test_run_state_defaults():
    state = RunState()
    assert state.status == RunStatus.IDLE
    assert state.iteration == 0
    assert state.done_reason is None
    assert state.error is None
    assert state.error_code is None
    assert state.trace_span is None
    assert state.runtime_context == {}
    assert state.reply_id == ""
    assert state.turn_id == ""
    assert state.empty_retries == 0
    assert state.in_finalization is False
    assert state.token_usage["prompt_tokens"] == 0
    assert state.token_usage["completion_tokens"] == 0
    assert state.token_usage["cached_tokens"] == 0
    assert state.token_usage["llm_calls"] == 0


def test_run_state_start_resets_fields():
    state = RunState()
    state.iteration = 5
    state.empty_retries = 3
    state.in_finalization = True
    state.error = "old error"
    state.runtime_context = {"session_id": "s1", "turn_id": "t1"}
    state.start()
    assert state.status == RunStatus.RUNNING
    assert state.iteration == 0
    assert state.empty_retries == 0
    assert state.in_finalization is False
    assert state.error is None
    assert state.error_code is None
    assert state.done_reason is None
    assert state.turn_id == "t1"  # 从 runtime_context 继承


def test_run_state_is_cancelled():
    state = RunState()
    assert state.is_cancelled is False
    state.status = RunStatus.CANCELLED
    assert state.is_cancelled is True


def test_run_state_is_done():
    state = RunState()
    assert state.is_done is False
    for status in (RunStatus.COMPLETED, RunStatus.ERROR, RunStatus.CANCELLED):
        state.status = status
        assert state.is_done is True
    state.status = RunStatus.RUNNING
    assert state.is_done is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd E:\ftre-agent-core && python -m pytest tests/test_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ftre_agent_core.agent.runner._state'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ftre_agent_core/agent/runner/_state.py
"""Runner 运行状态、状态枚举和内部取消异常。"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from ...types import ReplyFinishedReason
from ...tool import CancellationToken

if TYPE_CHECKING:
    from ...tracing import TraceSpan


class RunStatus(str, Enum):
    """单次 run() 调用的生命周期状态。"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class CancelledError(Exception):
    """内部取消异常，与 asyncio.CancelledError 区分。"""
    pass


@dataclass
class RunState:
    """一次 run() 执行期间的可变状态。"""
    # 生命周期
    status: RunStatus = RunStatus.IDLE
    iteration: int = 0
    done_reason: ReplyFinishedReason | None = None
    error: str | None = None
    error_code: str | None = None

    # 取消 — cancel_token 仅供 ToolHandler 兼容，从不主动 cancel
    cancel_token: CancellationToken = field(default_factory=CancellationToken)

    # Tracing
    trace_span: "TraceSpan | None" = None

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

    @property
    def is_cancelled(self) -> bool:
        return self.status == RunStatus.CANCELLED

    @property
    def is_done(self) -> bool:
        """是否处于终态。"""
        return self.status in (RunStatus.COMPLETED, RunStatus.ERROR, RunStatus.CANCELLED)

    def start(self) -> None:
        """重置全部字段，开始新一轮执行。"""
        self.status = RunStatus.RUNNING
        self.iteration = 0
        self.error = None
        self.error_code = None
        self.done_reason = None
        self.cancel_token = CancellationToken()
        self.empty_retries = 0
        self.in_finalization = False
        self.trace_span = None
        self.reply_id = ""
        self.turn_id = self.runtime_context.get("turn_id") or f"turn_{uuid.uuid4().hex[:12]}"
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "llm_calls": 0,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd E:\ftre-agent-core && python -m pytest tests/test_state.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
cd E:\ftre-agent-core && git add src/ftre_agent_core/agent/runner/_state.py tests/test_state.py && git commit -m "feat(runner): add _state.py with RunState, RunStatus, CancelledError"
```

---

### Task 2: 创建 `_actions.py` — 动作模型与数据载体

**Files:**
- Create: `src/ftre_agent_core/agent/runner/_actions.py`
- Test: `tests/test_actions.py`

**Interfaces:**
- Consumes: `ToolCall` from `ftre_agent_core.llm`, `ReplyFinishedReason` from `ftre_agent_core.types`
- Produces: `Reasoning`, `Acting`, `Exit` (Pydantic models), `TurnResult`, `ExitOutcome` (dataclasses)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_actions.py
"""动作模型与数据载体单元测试。"""
import pytest
from ftre_agent_core.agent.runner._actions import (
    Reasoning, Acting, Exit, TurnResult, ExitOutcome,
)
from ftre_agent_core.llm import ToolCall
from ftre_agent_core.types import ReplyFinishedReason


def test_reasoning_defaults():
    r = Reasoning()
    assert r.hint is None
    assert r.tool_choice is None
    assert r.force_no_tools is False


def test_reasoning_with_fields():
    r = Reasoning(hint="test hint", tool_choice="my_tool", force_no_tools=True)
    assert r.hint == "test hint"
    assert r.tool_choice == "my_tool"
    assert r.force_no_tools is True


def test_acting():
    tc = ToolCall(id="c1", name="echo", input={"text": "hi"})
    a = Acting(tool_calls=[tc])
    assert len(a.tool_calls) == 1
    assert a.tool_calls[0].id == "c1"


def test_exit_completed():
    e = Exit(finished_reason=ReplyFinishedReason.COMPLETED)
    assert e.finished_reason == ReplyFinishedReason.COMPLETED
    assert e.exit_msg is None
    assert e.error is None
    assert e.error_code is None


def test_exit_error():
    e = Exit(
        finished_reason=ReplyFinishedReason.ERROR,
        error="something broke",
        error_code="empty_response",
    )
    assert e.finished_reason == ReplyFinishedReason.ERROR
    assert e.error == "something broke"
    assert e.error_code == "empty_response"


def test_turn_result_defaults():
    tr = TurnResult(text="hello", reasoning="", tool_calls=[], finish_reason="stop")
    assert tr.text == "hello"
    assert tr.reasoning == ""
    assert tr.tool_calls == []
    assert tr.finish_reason == "stop"
    assert tr.usage is None
    assert tr.error is None


def test_exitOutcome_defaults():
    eo = ExitOutcome()
    assert eo.should_continue is False
    assert eo.continue_hint is None


def test_exitOutcome_continue():
    eo = ExitOutcome(should_continue=True, continue_hint="keep going")
    assert eo.should_continue is True
    assert eo.continue_hint == "keep going"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd E:\ftre-agent-core && python -m pytest tests/test_actions.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ftre_agent_core/agent/runner/_actions.py
"""ReAct 动作模型与执行器间数据载体。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ...llm import ToolCall, LLMError
from ...types import ReplyFinishedReason


class Reasoning(BaseModel):
    """下一步：调用大模型进行推理。"""
    hint: str | None = None
    tool_choice: str | None = None
    force_no_tools: bool = False


class Acting(BaseModel):
    """下一步：执行模型产生的工具调用。"""
    tool_calls: list[ToolCall]


class Exit(BaseModel):
    """下一步：结束（或暂停）当前回复。"""
    finished_reason: ReplyFinishedReason
    exit_msg: Any | None = None
    error: str | None = None
    error_code: str | None = None


@dataclass
class TurnResult:
    """一轮 LLM 推理的结构化产物，供 _decide() 消费。"""
    text: str
    reasoning: str
    tool_calls: list[ToolCall]
    finish_reason: str
    usage: dict | None = None
    error: LLMError | None = None


@dataclass
class ExitOutcome:
    """Exit 执行结果，可能让主循环继续而非退出。"""
    should_continue: bool = False
    continue_hint: str | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd E:\ftre-agent-core && python -m pytest tests/test_actions.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
cd E:\ftre-agent-core && git add src/ftre_agent_core/agent/runner/_actions.py tests/test_actions.py && git commit -m "feat(runner): add _actions.py with Reasoning, Acting, Exit, TurnResult, ExitOutcome"
```

---

### Task 3: 创建 `_decide.py` — 纯决策函数

**Files:**
- Create: `src/ftre_agent_core/agent/runner/_decide.py`
- Test: `tests/test_decide.py`

**Interfaces:**
- Consumes: `Reasoning, Acting, Exit, TurnResult` from `_actions.py`, `RunState` from `_state.py`
- Produces: `decide(state, prev) -> Reasoning | Acting | Exit`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_decide.py
"""_decide() 纯决策函数单元测试。"""
import pytest
from ftre_agent_core.agent.runner._decide import decide, MAX_EMPTY_RESPONSE_RETRIES, FINALIZATION_RETRY_PROMPT, EMPTY_FINAL_RESPONSE_MESSAGE
from ftre_agent_core.agent.runner._actions import Reasoning, Acting, Exit, TurnResult
from ftre_agent_core.agent.runner._state import RunState
from ftre_agent_core.llm import ToolCall, LLMError
from ftre_agent_core.types import ReplyFinishedReason


def make_state(iteration=0, empty_retries=0, in_finalization=False, max_iterations=10):
    s = RunState()
    s.iteration = iteration
    s.empty_retries = empty_retries
    s.in_finalization = in_finalization
    s.runtime_context = {"max_iterations": max_iterations}
    return s


def make_turn(text="", tool_calls=None, finish_reason="stop", error=None):
    return TurnResult(
        text=text,
        reasoning="",
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        error=error,
    )


# --- 优先级 1: error 非空 → Exit(ERROR) ---
def test_error_in_turn_result_returns_exit_error():
    state = make_state()
    prev = make_turn(error=LLMError(message="boom", code="unknown"))
    action = decide(state, prev)
    assert isinstance(action, Exit)
    assert action.finished_reason == ReplyFinishedReason.ERROR
    assert "boom" in action.error


# --- 优先级 2: 有工具调用 → Acting ---
def test_tool_calls_returns_acting():
    state = make_state()
    tc = ToolCall(id="c1", name="echo", input={})
    prev = make_turn(tool_calls=[tc])
    action = decide(state, prev)
    assert isinstance(action, Acting)
    assert len(action.tool_calls) == 1


# --- 优先级 3: 有文本无工具 → Exit(COMPLETED) ---
def test_text_no_tools_returns_exit_completed():
    state = make_state()
    prev = make_turn(text="hello world")
    action = decide(state, prev)
    assert isinstance(action, Exit)
    assert action.finished_reason == ReplyFinishedReason.COMPLETED


# --- 优先级 4: 空响应 + in_finalization → Exit(ERROR) ---
def test_empty_in_finalization_returns_exit_error():
    state = make_state(in_finalization=True)
    prev = make_turn(text="")
    action = decide(state, prev)
    assert isinstance(action, Exit)
    assert action.finished_reason == ReplyFinishedReason.ERROR
    assert action.error_code == "empty_response"
    assert EMPTY_FINAL_RESPONSE_MESSAGE in action.error


# --- 优先级 5: 空响应 + retries < MAX → Reasoning() ---
def test_empty_retry_increments_counter():
    state = make_state(empty_retries=0)
    prev = make_turn(text="")
    action = decide(state, prev)
    assert isinstance(action, Reasoning)
    assert action.hint is None
    assert action.force_no_tools is False
    assert state.empty_retries == 1


def test_empty_retry_at_max_minus_one():
    state = make_state(empty_retries=MAX_EMPTY_RESPONSE_RETRIES - 1)
    prev = make_turn(text="")
    action = decide(state, prev)
    assert isinstance(action, Reasoning)
    assert state.empty_retries == MAX_EMPTY_RESPONSE_RETRIES


# --- 优先级 6: 空响应 + retries 耗尽 → 最终化 ---
def test_empty_retries_exhausted_enters_finalization():
    state = make_state(empty_retries=MAX_EMPTY_RESPONSE_RETRIES)
    prev = make_turn(text="")
    action = decide(state, prev)
    assert isinstance(action, Reasoning)
    assert action.force_no_tools is True
    assert FINALIZATION_RETRY_PROMPT in (action.hint or "")
    assert state.in_finalization is True


# --- 优先级 7: 达到 max_iterations → Exit(EXCEED_MAX_ITERS) ---
def test_max_iterations_returns_exit_exceed():
    state = make_state(iteration=5, max_iterations=5)
    prev = None
    action = decide(state, prev)
    assert isinstance(action, Exit)
    assert action.finished_reason == ReplyFinishedReason.EXCEED_MAX_ITERS


# --- 优先级 8: 默认 → Reasoning() ---
def test_prev_none_returns_reasoning():
    state = make_state(iteration=0, max_iterations=10)
    action = decide(state, None)
    assert isinstance(action, Reasoning)
    assert action.hint is None


def test_prev_none_after_acting_returns_reasoning():
    state = make_state(iteration=1, max_iterations=10)
    action = decide(state, None)
    assert isinstance(action, Reasoning)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd E:\ftre-agent-core && python -m pytest tests/test_decide.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ftre_agent_core/agent/runner/_decide.py
"""纯决策函数：根据当前状态和上一轮结果决定下一步动作。"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._actions import Reasoning, Acting, Exit, TurnResult

if TYPE_CHECKING:
    from ._state import RunState

MAX_EMPTY_RESPONSE_RETRIES = 2

FINALIZATION_RETRY_PROMPT = "请根据上面的对话，直接给出回复用户的最终内容。"
EMPTY_FINAL_RESPONSE_MESSAGE = "模型多次重试后仍未生成可见的最终文本回复。"


def decide(state: "RunState", prev: TurnResult | None) -> Reasoning | Acting | Exit:
    """根据当前状态和上一轮 TurnResult 决定下一步动作。

    纯函数：不执行 I/O，不 yield 事件。
    副作用仅限修改 state.empty_retries 和 state.in_finalization。

    判断优先级：
        1. prev.error 非空 → Exit(ERROR)
        2. prev.tool_calls 非空 → Acting
        3. prev.text 非空且无工具调用 → Exit(COMPLETED)
        4. 空响应 + in_finalization → Exit(ERROR)
        5. 空响应 + empty_retries < MAX → Reasoning() 重试
        6. 空响应 + 重试耗尽 → Reasoning(最终化提示, force_no_tools)
        7. iteration >= max_iterations → Exit(EXCEED_MAX_ITERS)
        8. 默认 → Reasoning()
    """
    max_iterations = state.runtime_context.get("max_iterations")

    # 1. LLM 错误 → 直接退出
    if prev is not None and prev.error is not None:
        return Exit(
            finished_reason=ReplyFinishedReason_import(),
            error=f"[{prev.error.code}] {prev.error.message}",
            error_code=prev.error.code,
        )

    # 2. 有工具调用 → 执行工具
    if prev is not None and prev.tool_calls:
        return Acting(tool_calls=prev.tool_calls)

    # 3. 有非空文本且无工具调用 → 正常完成
    if prev is not None and prev.text.strip():
        return Exit(finished_reason=ReplyFinishedReason_import())

    # 4-6. 空响应处理
    if prev is not None and not prev.text.strip():
        # 4. 已在最终化阶段还是空 → 彻底失败
        if state.in_finalization:
            return Exit(
                finished_reason=ReplyFinishedReason_import_error(),
                error=EMPTY_FINAL_RESPONSE_MESSAGE,
                error_code="empty_response",
            )

        # 5. 重试次数未达上限 → 继续 Reasoning
        if state.empty_retries < MAX_EMPTY_RESPONSE_RETRIES:
            state.empty_retries += 1
            return Reasoning()

        # 6. 重试耗尽 → 进入强制最终化
        state.in_finalization = True
        return Reasoning(
            hint=FINALIZATION_RETRY_PROMPT,
            force_no_tools=True,
        )

    # 7. 达到最大迭代次数
    if max_iterations is not None and state.iteration >= max_iterations:
        return Exit(
            finished_reason=ReplyFinishedReason_import_exceed(),
        )

    # 8. 默认 → 继续推理
    return Reasoning()


# 辅助函数避免顶部 import 循环
from ...types import ReplyFinishedReason

def ReplyFinishedReason_import():
    return ReplyFinishedReason.ERROR if False else ReplyFinishedReason.COMPLETED

def ReplyFinishedReason_import_error():
    return ReplyFinishedReason.ERROR

def ReplyFinishedReason_import_exceed():
    return ReplyFinishedReason.EXCEED_MAX_ITERS
```

Wait, that's messy. Let me rewrite the implementation cleanly.

- [ ] **Step 3 (revised): Write minimal implementation**

```python
# src/ftre_agent_core/agent/runner/_decide.py
"""纯决策函数：根据当前状态和上一轮结果决定下一步动作。"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ...types import ReplyFinishedReason
from ._actions import Reasoning, Acting, Exit, TurnResult

if TYPE_CHECKING:
    from ._state import RunState

MAX_EMPTY_RESPONSE_RETRIES = 2

FINALIZATION_RETRY_PROMPT = "请根据上面的对话，直接给出回复用户的最终内容。"
EMPTY_FINAL_RESPONSE_MESSAGE = "模型多次重试后仍未生成可见的最终文本回复。"


def decide(state: "RunState", prev: TurnResult | None) -> Reasoning | Acting | Exit:
    """根据当前状态和上一轮 TurnResult 决定下一步动作。

    纯函数：不执行 I/O，不 yield 事件。
    副作用仅限修改 state.empty_retries 和 state.in_finalization。
    """
    max_iterations = state.runtime_context.get("max_iterations")

    # 1. LLM 错误 → 直接退出
    if prev is not None and prev.error is not None:
        return Exit(
            finished_reason=ReplyFinishedReason.ERROR,
            error=f"[{prev.error.code}] {prev.error.message}",
            error_code=prev.error.code,
        )

    # 2. 有工具调用 → 执行工具
    if prev is not None and prev.tool_calls:
        return Acting(tool_calls=prev.tool_calls)

    # 3. 有非空文本且无工具调用 → 正常完成
    if prev is not None and prev.text.strip():
        return Exit(finished_reason=ReplyFinishedReason.COMPLETED)

    # 4-6. 空响应处理（prev 非空但文本为空）
    if prev is not None and not prev.text.strip():
        # 4. 已在最终化阶段还是空 → 彻底失败
        if state.in_finalization:
            return Exit(
                finished_reason=ReplyFinishedReason.ERROR,
                error=EMPTY_FINAL_RESPONSE_MESSAGE,
                error_code="empty_response",
            )

        # 5. 重试次数未达上限 → 继续 Reasoning
        if state.empty_retries < MAX_EMPTY_RESPONSE_RETRIES:
            state.empty_retries += 1
            return Reasoning()

        # 6. 重试耗尽 → 进入强制最终化
        state.in_finalization = True
        return Reasoning(
            hint=FINALIZATION_RETRY_PROMPT,
            force_no_tools=True,
        )

    # 7. 达到最大迭代次数
    if max_iterations is not None and state.iteration >= max_iterations:
        return Exit(finished_reason=ReplyFinishedReason.EXCEED_MAX_ITERS)

    # 8. 默认 → 继续推理
    return Reasoning()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd E:\ftre-agent-core && python -m pytest tests/test_decide.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
cd E:\ftre-agent-core && git add src/ftre_agent_core/agent/runner/_decide.py tests/test_decide.py && git commit -m "feat(runner): add _decide.py pure decision function"
```

---

### Task 4: 创建 `_execute_reasoning.py` — LLM 调用执行器

**Files:**
- Create: `src/ftre_agent_core/agent/runner/_execute_reasoning.py`
- Test: `tests/test_execute_reasoning.py`

**Interfaces:**
- Consumes: `Reasoning, TurnResult` from `_actions.py`, `RunState` from `_state.py`, `LLMHandler` from `ftre_agent_core.llm`, `MemoryManager` from `ftre_agent_core.memory`
- Produces: `ReasoningExecutor` class with `async def stream(action) -> AsyncGenerator[AgentStreamEvent, None]` and `.result: TurnResult` attribute

- [ ] **Step 1: Write the failing test**

```python
# tests/test_execute_reasoning.py
"""ReasoningExecutor 单元测试。"""
import pytest
from ftre_agent_core.agent.runner._execute_reasoning import ReasoningExecutor
from ftre_agent_core.agent.runner._actions import Reasoning, TurnResult
from ftre_agent_core.agent.runner._state import RunState
from ftre_agent_core.agent.react import ReActAgent
from ftre_agent_core.llm import TextDelta, ToolCall, StepFinish, LLMError
from ftre_agent_core.event import EventType


def make_agent():
    return ReActAgent(
        model="fake", api_key="fake", system_prompt="test",
        max_iterations=5, max_retries=1, retry_delay=0.01,
    )


def make_state():
    s = RunState()
    s.runtime_context = {"session_id": "s1", "max_iterations": 5}
    s.start()
    s.reply_id = "r1"
    return s


@pytest.mark.asyncio
async def test_text_only_turn():
    agent = make_agent()
    state = make_state()

    async def fake_stream(messages, tools=None):
        yield TextDelta(text="hello")
        yield StepFinish(finish_reason="stop", usage={"prompt_tokens": 10, "completion_tokens": 5})

    agent.runner.llm.stream = fake_stream

    executor = ReasoningExecutor(agent, state, agent.runner.llm, agent.hook_manager)
    events = [e async for e in executor.stream(Reasoning())]

    assert executor.result.text == "hello"
    assert executor.result.tool_calls == []
    assert executor.result.finish_reason == "stop"
    assert executor.result.error is None
    # 验证事件
    types = [e.type for e in events]
    assert EventType.MODEL_CALL_START in types
    assert EventType.TEXT_BLOCK_START in types
    assert EventType.TEXT_BLOCK_DELTA in types
    assert EventType.TEXT_BLOCK_END in types
    assert EventType.MODEL_CALL_END in types
    # 验证 memory 写入
    assert len(agent.memory.messages) == 1
    assert agent.memory.messages[0]["role"] == "assistant"


@pytest.mark.asyncio
async def test_tool_call_turn():
    agent = make_agent()
    state = make_state()

    async def fake_stream(messages, tools=None):
        yield ToolCall(id="c1", name="echo", input={"text": "hi"})
        yield StepFinish(finish_reason="tool_calls")

    agent.runner.llm.stream = fake_stream

    executor = ReasoningExecutor(agent, state, agent.runner.llm, agent.hook_manager)
    events = [e async for e in executor.stream(Reasoning())]

    assert executor.result.text == ""
    assert len(executor.result.tool_calls) == 1
    assert executor.result.tool_calls[0].id == "c1"
    assert executor.result.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_hint_written_to_memory_before_llm_call():
    agent = make_agent()
    state = make_state()

    hint_seen = None

    async def fake_stream(messages, tools=None):
        nonlocal hint_seen
        # hint 应该已经写入 memory
        hint_seen = messages[-1]
        yield TextDelta(text="done")
        yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    executor = ReasoningExecutor(agent, state, agent.runner.llm, agent.hook_manager)
    events = [e async for e in executor.stream(Reasoning(hint="test hint"))]

    assert hint_seen is not None
    assert hint_seen["role"] == "user"
    assert "test hint" in hint_seen["content"]


@pytest.mark.asyncio
async def test_force_no_tools_passes_none():
    agent = make_agent()
    state = make_state()

    tools_received = "not_called"

    async def fake_stream(messages, tools=None):
        nonlocal tools_received
        tools_received = tools
        yield TextDelta(text="final")
        yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    executor = ReasoningExecutor(agent, state, agent.runner.llm, agent.hook_manager)
    events = [e async for e in executor.stream(Reasoning(force_no_tools=True))]

    assert tools_received is None


@pytest.mark.asyncio
async def test_retry_on_rate_limit():
    agent = make_agent()
    state = make_state()

    call_count = 0

    async def fake_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise LLMError(message="rate limited", code="rate_limit")
        yield TextDelta(text="success")
        yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    executor = ReasoningExecutor(agent, state, agent.runner.llm, agent.hook_manager)
    events = [e async for e in executor.stream(Reasoning())]

    assert call_count == 2
    assert executor.result.text == "success"
    # 应该有 RetryEvent
    retry_events = [e for e in events if e.type == EventType.RETRY]
    assert len(retry_events) == 1


@pytest.mark.asyncio
async def test_retry_exhausted_returns_error():
    agent = make_agent()
    state = make_state()

    async def fake_stream(messages, tools=None):
        raise LLMError(message="rate limited", code="rate_limit")

    agent.runner.llm.stream = fake_stream

    executor = ReasoningExecutor(agent, state, agent.runner.llm, agent.hook_manager)
    events = [e async for e in executor.stream(Reasoning())]

    assert executor.result.error is not None
    assert executor.result.error.code == "rate_limit"


@pytest.mark.asyncio
async def test_unretryable_error_returns_error_immediately():
    agent = make_agent()
    state = make_state()

    call_count = 0

    async def fake_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        raise LLMError(message="bad request", code="bad_request")

    agent.runner.llm.stream = fake_stream

    executor = ReasoningExecutor(agent, state, agent.runner.llm, agent.hook_manager)
    events = [e async for e in executor.stream(Reasoning())]

    assert call_count == 1
    assert executor.result.error is not None
    assert executor.result.error.code == "bad_request"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd E:\ftre-agent-core && python -m pytest tests/test_execute_reasoning.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ftre_agent_core/agent/runner/_execute_reasoning.py
"""LLM 调用执行器：流式消费 + 重试 + 产出事件 + 返回 TurnResult。"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, AsyncGenerator

from ...event import (
    AgentStreamEvent,
    ModelCallStartEvent, ModelCallEndEvent,
    TextBlockStartEvent, TextBlockDeltaEvent, TextBlockEndEvent,
    ThinkingBlockStartEvent, ThinkingBlockDeltaEvent, ThinkingBlockEndEvent,
    ToolCallStartEvent, ToolCallDeltaEvent, ToolCallEndEvent,
    HintBlockEvent,
    RetryEvent,
)
from ...llm import LLMHandler, LLMError, TextDelta, ReasoningDelta, ToolInputDelta, ToolCall, StepFinish
from ._actions import Reasoning, TurnResult

if TYPE_CHECKING:
    from ...hooks import FtreCoreHookManager
    from ...memory import MemoryManager
    from ...tracing import TraceSpan
    from ._state import RunState

logger = logging.getLogger(__name__)


class ReasoningExecutor:
    """执行 Reasoning 动作：调用 LLM，流式产出事件，返回 TurnResult。"""

    def __init__(
        self,
        agent,
        state: "RunState",
        llm: LLMHandler,
        hook_manager: "FtreCoreHookManager",
    ):
        self.agent = agent
        self.state = state
        self.llm = llm
        self.hook_manager = hook_manager
        self.result: TurnResult | None = None

    async def stream(self, action: Reasoning) -> AsyncGenerator[AgentStreamEvent, None]:
        """执行 LLM 调用，yield 事件，结束后设置 self.result。"""
        reply_id = self.state.reply_id
        session_id = self.state.runtime_context.get("session_id", "")
        model_name = self.agent.model

        # hint 写入 memory（在 LLM 调用之前）
        if action.hint:
            self.agent.memory.add_raw({"role": "user", "content": action.hint})
            yield HintBlockEvent(
                reply_id=reply_id,
                block_id=uuid.uuid4().hex[:16],
                source="system",
                hint=action.hint,
                metadata={"hide": True, "internal": True, "reason": "finalization_retry"},
            )

        messages = self.agent.memory.get_messages()
        tools = None if action.force_no_tools else self.agent.tool_registry.to_openai_tools() or None

        max_attempts = 1 + self.agent.max_retries
        turn_start_ts = time.perf_counter()
        first_token_logged = False
        ttft_ms: float | None = None

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        finish_reason = "unknown"
        usage: dict | None = None

        for attempt in range(max_attempts):
            llm_span = None
            if self.state.trace_span:
                llm_span = self.state.trace_span.child(
                    "llm", RunType_LLM,
                    inputs={"messages": messages, "tools": tools},
                    metadata={
                        "model": model_name,
                        "api_type": self.agent.api_type,
                        "iteration": self.state.iteration,
                        "attempt": attempt + 1,
                    },
                )

            try:
                yield ModelCallStartEvent(reply_id=reply_id, model_name=model_name)

                text_block_id: str | None = None
                thinking_block_id: str | None = None
                tool_call_started: set[str] = set()

                async for event in self.llm.stream(messages, tools):
                    if not first_token_logged:
                        first_token_logged = True
                        ttft_ms = (time.perf_counter() - turn_start_ts) * 1000
                        logger.info(f"[react] 第 {self.state.iteration} 轮 TTFT {ttft_ms:.0f}ms")
                        if llm_span and not llm_span.ended:
                            llm_span.add_event("ttft", {"ms": round(ttft_ms)})

                    if isinstance(event, TextDelta):
                        text_parts.append(event.text)
                        if text_block_id is None:
                            text_block_id = uuid.uuid4().hex[:16]
                            yield TextBlockStartEvent(reply_id=reply_id, block_id=text_block_id)
                        yield TextBlockDeltaEvent(reply_id=reply_id, block_id=text_block_id, delta=event.text)

                    elif isinstance(event, ReasoningDelta):
                        reasoning_parts.append(event.text)
                        if thinking_block_id is None:
                            thinking_block_id = uuid.uuid4().hex[:16]
                            yield ThinkingBlockStartEvent(reply_id=reply_id, block_id=thinking_block_id)
                        yield ThinkingBlockDeltaEvent(reply_id=reply_id, block_id=thinking_block_id, delta=event.text)

                    elif isinstance(event, ToolInputDelta):
                        if event.id not in tool_call_started:
                            tool_call_started.add(event.id)
                            yield ToolCallStartEvent(reply_id=reply_id, tool_call_id=event.id, tool_call_name=event.name or "")
                        yield ToolCallDeltaEvent(reply_id=reply_id, tool_call_id=event.id, delta=event.text)

                    elif isinstance(event, ToolCall):
                        if event.id not in tool_call_started:
                            tool_call_started.add(event.id)
                            yield ToolCallStartEvent(reply_id=reply_id, tool_call_id=event.id, tool_call_name=event.name)
                        yield ToolCallEndEvent(reply_id=reply_id, tool_call_id=event.id)
                        tool_calls.append(event)

                    elif isinstance(event, StepFinish):
                        finish_reason = event.finish_reason
                        response_metadata = event.response_metadata
                        if event.usage:
                            usage = event.usage
                            self.state.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                            self.state.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                            self.state.token_usage["cached_tokens"] += usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                            self.state.token_usage["llm_calls"] += 1

                        if text_block_id is not None:
                            yield TextBlockEndEvent(reply_id=reply_id, block_id=text_block_id)
                            text_block_id = None
                        if thinking_block_id is not None:
                            yield ThinkingBlockEndEvent(reply_id=reply_id, block_id=thinking_block_id)
                            thinking_block_id = None

                        input_tokens = usage.get("prompt_tokens", 0) if usage else 0
                        output_tokens = usage.get("completion_tokens", 0) if usage else 0
                        yield ModelCallEndEvent(
                            reply_id=reply_id,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            finished_reason=finish_reason,
                        )

                # 关闭 LLM span
                full_text = "".join(text_parts)
                full_reasoning = "".join(reasoning_parts)
                if llm_span and not llm_span.ended:
                    llm_span.end(outputs={
                        "text": full_text,
                        "reasoning": full_reasoning,
                        "finish_reason": finish_reason,
                        "has_tool_calls": bool(tool_calls),
                        "usage": usage,
                    })

                # 写入 memory
                self.agent.memory.add_assistant(full_text, reasoning=full_reasoning or None)

                # 成功完成
                self.result = TurnResult(
                    text=full_text,
                    reasoning=full_reasoning,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                    usage=usage,
                )
                return

            except Exception as exc:
                # 关闭 LLM span
                if llm_span and not llm_span.ended:
                    if isinstance(exc, asyncio.CancelledError):
                        llm_span.end(status=TraceRunStatus_CANCELLED)
                    else:
                        llm_span.end(error=exc)

                # 收尾 open blocks
                if text_block_id is not None:
                    yield TextBlockEndEvent(reply_id=reply_id, block_id=text_block_id)
                    text_block_id = None
                if thinking_block_id is not None:
                    yield ThinkingBlockEndEvent(reply_id=reply_id, block_id=thinking_block_id)
                    thinking_block_id = None

                # 写入半截文本到 memory
                _full_text = "".join(text_parts)
                _full_reasoning = "".join(reasoning_parts)
                if _full_text.strip():
                    self.agent.memory.add_assistant(_full_text, reasoning=_full_reasoning or None)

                # CancelledError 直接传播（不走重试）
                if isinstance(exc, asyncio.CancelledError):
                    raise

                err = exc if isinstance(exc, LLMError) else LLMError.classify(exc)
                is_last = attempt >= max_attempts - 1

                logger.warning(
                    "LLM 调用失败 [%s] %s (第 %d/%d 次尝试)",
                    err.code, err.message[:200], attempt + 1, max_attempts,
                )

                if err.code in LLMError.UNRETRYABLE_CODES or is_last:
                    # 重试耗尽或不可重试 → 返回 error
                    self.result = TurnResult(
                        text="",
                        reasoning="",
                        tool_calls=[],
                        finish_reason="error",
                        error=err,
                    )
                    return

                # 可重试 → 发出 RetryEvent，等待后继续
                yield RetryEvent(
                    reply_id=reply_id,
                    code=err.code,
                    message=err.message,
                    attempt=attempt + 1,
                    max_attempts=max_attempts - 1,
                )
                await asyncio.sleep(self.agent.retry_delay)
                # 重新读取 messages
                messages = self.agent.memory.get_messages()
                # 重置收集器
                text_parts = []
                reasoning_parts = []
                tool_calls = []
                finish_reason = "unknown"
                usage = None
```

Note: The implementation references `RunType_LLM` and `TraceRunStatus_CANCELLED` which need to be imported. Add these imports at the top:

```python
from ...tracing import RunType, RunStatus as TraceRunStatus
```

And use `RunType.LLM` and `TraceRunStatus.CANCELLED` in the code. Fix the placeholder names.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd E:\ftre-agent-core && python -m pytest tests/test_execute_reasoning.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
cd E:\ftre-agent-core && git add src/ftre_agent_core/agent/runner/_execute_reasoning.py tests/test_execute_reasoning.py && git commit -m "feat(runner): add _execute_reasoning.py with ReasoningExecutor"
```

---

### Task 5: 创建 `_execute_acting.py` — 工具执行器

**Files:**
- Create: `src/ftre_agent_core/agent/runner/_execute_acting.py`
- Test: `tests/test_execute_acting.py`

**Interfaces:**
- Consumes: `Acting` from `_actions.py`, `RunState` from `_state.py`, `ToolHandler` from `tool_handler.py`
- Produces: `ActingExecutor` class with `async def stream(action) -> AsyncGenerator[AgentStreamEvent, None]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_execute_acting.py
"""ActingExecutor 单元测试。"""
import pytest
from ftre_agent_core.agent.runner._execute_acting import ActingExecutor
from ftre_agent_core.agent.runner._actions import Acting
from ftre_agent_core.agent.runner._state import RunState
from ftre_agent_core.agent.react import ReActAgent
from ftre_agent_core.llm import ToolCall
from ftre_agent_core.tool import tool, ToolRegistry
from ftre_agent_core.event import EventType


def make_agent():
    @tool(description="Echo text")
    def echo(text: str) -> str:
        return f"echo:{text}"

    registry = ToolRegistry()
    registry.register(echo)
    return ReActAgent(
        model="fake", api_key="fake", system_prompt="test",
        tool_registry=registry, max_iterations=5,
    )


def make_state():
    s = RunState()
    s.runtime_context = {"session_id": "s1", "max_iterations": 5}
    s.start()
    s.reply_id = "r1"
    return s


@pytest.mark.asyncio
async def test_single_tool_execution():
    agent = make_agent()
    state = make_state()

    # 先写入 assistant 消息（含 tool_calls）到 memory
    from ftre_agent_core.agent.runner.tool_handler import ToolHandler
    tc = ToolCall(id="c1", name="echo", input={"text": "hello"})
    agent.memory.add_raw(
        ToolHandler.build_assistant_message(tool_calls=[tc], content="let me echo")
    )

    executor = ActingExecutor(agent, state, agent.runner.tool_handler)
    events = [e async for e in executor.stream(Acting(tool_calls=[tc]))]

    # 验证事件
    types = [e.type for e in events]
    assert EventType.TOOL_RESULT_START in types
    assert EventType.TOOL_RESULT_END in types

    # 验证 memory 写入（assistant tool_calls + tool result）
    msgs = agent.memory.messages
    # 最后一条应该是 tool result
    assert "echo:hello" in str(msgs[-1].get("content", ""))


@pytest.mark.asyncio
async def test_multiple_tool_execution():
    agent = make_agent()
    state = make_state()

    from ftre_agent_core.agent.runner.tool_handler import ToolHandler
    tc1 = ToolCall(id="c1", name="echo", input={"text": "a"})
    tc2 = ToolCall(id="c2", name="echo", input={"text": "b"})

    executor = ActingExecutor(agent, state, agent.runner.tool_handler)
    events = [e async for e in executor.stream(Acting(tool_calls=[tc1, tc2]))]

    tool_end_events = [e for e in events if e.type == EventType.TOOL_RESULT_END]
    assert len(tool_end_events) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd E:\ftre-agent-core && python -m pytest tests/test_execute_acting.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ftre_agent_core/agent/runner/_execute_acting.py
"""工具执行器：并发执行工具 + 成组写入 Memory + 产出事件。"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, AsyncGenerator

from ...event import (
    AgentStreamEvent,
    ToolResultStartEvent, ToolResultTextDeltaEvent, ToolResultEndEvent,
    HintBlockEvent,
)
from ...message import ToolResultState
from ._actions import Acting
from ._state import CancelledError
from .tool_handler import ToolHandler

if TYPE_CHECKING:
    from ._state import RunState


class ActingExecutor:
    """执行 Acting 动作：并发执行工具，成组写入 memory，产出事件。"""

    def __init__(self, agent, state: "RunState", tool_handler: ToolHandler):
        self.agent = agent
        self.state = state
        self.tool_handler = tool_handler

    async def stream(self, action: Acting) -> AsyncGenerator[AgentStreamEvent, None]:
        """执行工具调用，yield 事件。"""
        reply_id = self.state.reply_id
        tool_calls = action.tool_calls

        # spawn 所有工具任务
        tool_tasks: dict[str, asyncio.Task] = {}
        for call in tool_calls:
            tool_tasks[call.id] = self.tool_handler.spawn(
                call,
                self.state,
                parent_span=self.state.trace_span,
            )

        # 等待全部完成
        results, cancelled = await self.tool_handler.gather_results(
            tool_calls, tool_tasks, self.state,
        )

        # 成组写入 memory：assistant(tool_calls) → tool(result_1) → ...
        self.agent.memory.add_raw(
            self.tool_handler.build_assistant_message(
                tool_calls=tool_calls,
            )
        )

        # 收集 pending hints（延后追加）
        pending_hints: list[AgentStreamEvent] = []

        for tc, result in zip(tool_calls, results):
            self.agent.memory.add_tool_result(
                tc.id, result.result or f"[{tc.name}] 已完成"
            )

            yield ToolResultStartEvent(
                reply_id=reply_id, tool_call_id=tc.id, tool_call_name=tc.name,
            )
            if result.result:
                yield ToolResultTextDeltaEvent(
                    reply_id=reply_id, tool_call_id=tc.id, delta=result.result,
                )
            state = ToolResultState.SUCCESS if not result.error else ToolResultState.ERROR
            yield ToolResultEndEvent(
                reply_id=reply_id, tool_call_id=tc.id,
                state=state, metadata=result.metadata or {},
            )

            if result.event is not None:
                pending_hints.append(result.event)

        # 统一追加 hints（确保在 tool_result 之后）
        for ev in pending_hints:
            if isinstance(ev, HintBlockEvent):
                content = ev.hint if isinstance(ev.hint, str) else str(ev.hint)
            else:
                content = str(ev)
            self.agent.memory.add_raw({"role": "user", "content": content})
            yield HintBlockEvent(
                reply_id=reply_id,
                block_id=uuid.uuid4().hex[:16],
                source="tool",
                hint=content,
            )

        if cancelled:
            raise CancelledError()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd E:\ftre-agent-core && python -m pytest tests/test_execute_acting.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
cd E:\ftre-agent-core && git add src/ftre_agent_core/agent/runner/_execute_acting.py tests/test_execute_acting.py && git commit -m "feat(runner): add _execute_acting.py with ActingExecutor"
```

---

### Task 6: 创建 `_execute_exit.py` — 退出执行器

**Files:**
- Create: `src/ftre_agent_core/agent/runner/_execute_exit.py`
- Test: `tests/test_execute_exit.py`

**Interfaces:**
- Consumes: `Exit, ExitOutcome` from `_actions.py`, `RunState` from `_state.py`, `FtreCoreHookManager` from `hooks.py`
- Produces: `ExitExecutor` class with `async def stream(action) -> AsyncGenerator[AgentStreamEvent, None]` and `.outcome: ExitOutcome` attribute

- [ ] **Step 1: Write the failing test**

```python
# tests/test_execute_exit.py
"""ExitExecutor 单元测试。"""
import pytest
from ftre_agent_core.agent.runner._execute_exit import ExitExecutor
from ftre_agent_core.agent.runner._actions import Exit, ExitOutcome
from ftre_agent_core.agent.runner._state import RunState
from ftre_agent_core.agent.react import ReActAgent
from ftre_agent_core.event import EventType
from ftre_agent_core.types import ReplyFinishedReason


def make_agent():
    return ReActAgent(
        model="fake", api_key="fake", system_prompt="test",
        max_iterations=5,
    )


def make_state():
    s = RunState()
    s.runtime_context = {"session_id": "s1", "max_iterations": 5}
    s.start()
    s.reply_id = "r1"
    return s


@pytest.mark.asyncio
async def test_completed_exit_yields_reply_end():
    agent = make_agent()
    state = make_state()

    executor = ExitExecutor(agent, state, agent.hook_manager)
    events = [e async for e in executor.stream(Exit(finished_reason=ReplyFinishedReason.COMPLETED))]

    types = [e.type for e in events]
    assert EventType.REPLY_END in types
    assert executor.outcome.should_continue is False
    assert state.done_reason == ReplyFinishedReason.COMPLETED


@pytest.mark.asyncio
async def test_error_exit_no_on_stop_hook():
    agent = make_agent()
    state = make_state()

    executor = ExitExecutor(agent, state, agent.hook_manager)
    events = [e async for e in executor.stream(
        Exit(finished_reason=ReplyFinishedReason.ERROR, error="boom", error_code="test")
    )]

    assert executor.outcome.should_continue is False
    assert state.done_reason == ReplyFinishedReason.ERROR
    assert state.error == "boom"
    assert state.error_code == "test"


@pytest.mark.asyncio
async def test_on_stop_block_returns_continue():
    from ftre_agent_core.hooks import ON_STOP, StopInput, HookOutput

    agent = make_agent()
    state = make_state()

    def block_hook(inp: StopInput):
        return HookOutput(decision="block", reason="keep working")

    agent.hook_manager.register(ON_STOP, block_hook)

    executor = ExitExecutor(agent, state, agent.hook_manager)
    events = [e async for e in executor.stream(Exit(finished_reason=ReplyFinishedReason.COMPLETED))]

    assert executor.outcome.should_continue is True
    assert executor.outcome.continue_hint == "keep working"
    # 不应该有 ReplyEndEvent
    types = [e.type for e in events]
    assert EventType.REPLY_END not in types
    # 应该有 HintBlockEvent
    assert EventType.HINT_BLOCK in types
    # memory 应该有 continuation prompt
    msgs = agent.memory.messages
    assert any("keep working" in str(m.get("content", "")) for m in msgs)


@pytest.mark.asyncio
async def test_on_stop_allow_exits_normally():
    from ftre_agent_core.hooks import ON_STOP, StopInput, HookOutput

    agent = make_agent()
    state = make_state()

    def allow_hook(inp: StopInput):
        return HookOutput(decision="allow")

    agent.hook_manager.register(ON_STOP, allow_hook)

    executor = ExitExecutor(agent, state, agent.hook_manager)
    events = [e async for e in executor.stream(Exit(finished_reason=ReplyFinishedReason.COMPLETED))]

    assert executor.outcome.should_continue is False
    assert state.done_reason == ReplyFinishedReason.COMPLETED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd E:\ftre-agent-core && python -m pytest tests/test_execute_exit.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ftre_agent_core/agent/runner/_execute_exit.py
"""退出执行器：ON_STOP Hook + 产出 ReplyEndEvent + 设置终态。"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, AsyncGenerator

from ...event import AgentStreamEvent, ReplyEndEvent, HintBlockEvent
from ...types import ReplyFinishedReason
from ._actions import Exit, ExitOutcome
from ._state import RunStatus

if TYPE_CHECKING:
    from ...hooks import FtreCoreHookManager
    from ._state import RunState


class ExitExecutor:
    """执行 Exit 动作：ON_STOP Hook 检查 + 产出 ReplyEnd + 设置终态。"""

    def __init__(self, agent, state: "RunState", hook_manager: "FtreCoreHookManager"):
        self.agent = agent
        self.state = state
        self.hook_manager = hook_manager
        self.outcome: ExitOutcome = ExitOutcome()

    async def stream(self, action: Exit) -> AsyncGenerator[AgentStreamEvent, None]:
        """执行退出逻辑，yield 事件。"""
        session_id = self.state.runtime_context.get("session_id", "")
        reply_id = self.state.reply_id

        # 仅 COMPLETED 触发 ON_STOP
        if action.finished_reason == ReplyFinishedReason.COMPLETED:
            from ...hooks import ON_STOP, StopInput

            stop_output = await self.hook_manager.trigger(
                ON_STOP,
                lambda: StopInput(
                    session_id=session_id,
                    turn_id=self.state.turn_id,
                    iteration=self.state.iteration,
                    runtime_context=self.state.runtime_context,
                ),
            )

            if stop_output is not None and stop_output.decision == "block":
                # ON_STOP block → 不退出，返回 continue
                hint = stop_output.reason or "继续工作。"
                self.agent.memory.add_raw({"role": "user", "content": hint})
                yield HintBlockEvent(
                    reply_id=reply_id,
                    block_id=uuid.uuid4().hex[:16],
                    source="system",
                    hint=hint,
                    metadata={"hide": True, "internal": True, "reason": "stop_hook_block"},
                )
                self.outcome = ExitOutcome(should_continue=True, continue_hint=hint)
                return

        # 正常退出 → 设置终态 + yield ReplyEndEvent
        self._finalize(action.finished_reason, action.error, action.error_code)

        yield ReplyEndEvent(
            session_id=session_id,
            reply_id=reply_id,
            finished_reason=action.finished_reason,
            error={"message": action.error, "code": action.error_code} if action.error else None,
        )
        self.outcome = ExitOutcome()

    def _finalize(self, reason: ReplyFinishedReason, error: str | None, error_code: str | None) -> None:
        """设置终态。"""
        self.state.done_reason = reason
        self.state.status = (
            RunStatus.CANCELLED if reason == ReplyFinishedReason.INTERRUPTED
            else RunStatus.ERROR if reason == ReplyFinishedReason.ERROR
            else RunStatus.COMPLETED
        )
        if error:
            self.state.error = error
        if error_code:
            self.state.error_code = error_code
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd E:\ftre-agent-core && python -m pytest tests/test_execute_exit.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
cd E:\ftre-agent-core && git add src/ftre_agent_core/agent/runner/_execute_exit.py tests/test_execute_exit.py && git commit -m "feat(runner): add _execute_exit.py with ExitExecutor"
```

---

### Task 7: 重写 `react_runner.py` — 主 Runner

**Files:**
- Modify: `src/ftre_agent_core/agent/runner/react_runner.py` (完全重写)
- Test: `tests/test_react_runner_new.py`

**Interfaces:**
- Consumes: All previous tasks
- Produces: `ReActRunner` with `run()`, `cancel_nowait()`, `_loop()`, `_finalize()`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_react_runner_new.py
"""ReActRunner 主循环集成测试。"""
import asyncio
import pytest
from ftre_agent_core.agent import ReActAgent
from ftre_agent_core.event import EventType
from ftre_agent_core.llm import TextDelta, ToolCall, StepFinish, LLMError
from ftre_agent_core.tool import tool, ToolRegistry
from ftre_agent_core.types import ReplyFinishedReason


def make_agent(tools=None, max_iterations=5, max_retries=1):
    registry = ToolRegistry()
    for t in (tools or []):
        registry.register(t)
    return ReActAgent(
        model="fake", api_key="fake", system_prompt="test",
        tool_registry=registry, max_iterations=max_iterations,
        max_retries=max_retries, retry_delay=0.01,
    )


@pytest.mark.asyncio
async def test_simple_text_reply():
    agent = make_agent()

    async def fake_stream(messages, tools=None):
        yield TextDelta(text="hello world")
        yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream
    events = [e async for e in agent.run("hi")]

    types = [e.type for e in events]
    # 只有一对 ReplyStart/ReplyEnd
    assert types.count(EventType.REPLY_START) == 1
    assert types.count(EventType.REPLY_END) == 1
    assert agent.state.done_reason == ReplyFinishedReason.COMPLETED


@pytest.mark.asyncio
async def test_tool_call_then_text():
    @tool(description="Echo")
    def echo(text: str) -> str:
        return f"echo:{text}"

    agent = make_agent(tools=[echo])
    call_count = 0

    async def fake_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield ToolCall(id="c1", name="echo", input={"text": "hi"})
            yield StepFinish(finish_reason="tool_calls")
        else:
            yield TextDelta(text="done")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream
    events = [e async for e in agent.run("echo hi")]

    assert call_count == 2
    types = [e.type for e in events]
    assert types.count(EventType.REPLY_START) == 1
    assert types.count(EventType.REPLY_END) == 1
    assert EventType.TOOL_RESULT_END in types
    assert agent.state.done_reason == ReplyFinishedReason.COMPLETED


@pytest.mark.asyncio
async def test_max_iterations():
    agent = make_agent(max_iterations=2)

    async def fake_stream(messages, tools=None):
        yield ToolCall(id=f"c{id(messages)}", name="nonexistent", input={})
        yield StepFinish(finish_reason="tool_calls")

    agent.runner.llm.stream = fake_stream
    events = [e async for e in agent.run("loop")]

    assert agent.state.done_reason == ReplyFinishedReason.EXCEED_MAX_ITERS


@pytest.mark.asyncio
async def test_cancel_via_task_cancel():
    agent = make_agent()

    async def fake_stream(messages, tools=None):
        await asyncio.sleep(10)  # 永远不返回
        yield TextDelta(text="never")
        yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    task = asyncio.create_task(_collect_events(agent.run("hi")))

    await asyncio.sleep(0.1)
    agent.cancel_nowait()

    events = await task
    assert agent.state.done_reason == ReplyFinishedReason.INTERRUPTED
    types = [e.type for e in events]
    assert EventType.REPLY_END in types


async def _collect_events(gen):
    return [e async for e in gen]


@pytest.mark.asyncio
async def test_concurrent_run_raises():
    agent = make_agent()

    async def fake_stream(messages, tools=None):
        await asyncio.sleep(10)
        yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    task1 = asyncio.create_task(_collect_events(agent.run("first")))
    await asyncio.sleep(0.05)

    with pytest.raises(RuntimeError, match="already running"):
        async for _ in agent.run("second"):
            pass

    task1.cancel()
    try:
        await task1
    except asyncio.CancelledError:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd E:\ftre-agent-core && python -m pytest tests/test_react_runner_new.py -v`
Expected: FAIL (tests reference new behavior)

- [ ] **Step 3: Write minimal implementation**

```python
# src/ftre_agent_core/agent/runner/react_runner.py
"""ReActRunner — ReAct Agent 的核心执行引擎（状态机重构版）。

职责：
  - 驱动 Reason → Act → Observe 循环（通过 _decide + 执行器）
  - 管理运行锁（同一 Agent 禁止并发 run）
  - 取消入口（cancel_nowait → Task.cancel）
  - Tracing 根 span 生命周期
  - 统一终态写入（_finalize）
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, AsyncGenerator

from ...event import AgentStreamEvent, ReplyStartEvent, ReplyEndEvent
from ...tracing import RunStatus as TraceRunStatus, RunType
from ...types import ReplyFinishedReason
from ._actions import Reasoning, Acting, Exit, TurnResult
from ._decide import decide
from ._execute_acting import ActingExecutor
from ._execute_exit import ExitExecutor
from ._execute_reasoning import ReasoningExecutor
from ._state import RunState, RunStatus, CancelledError

if TYPE_CHECKING:
    from ..react import ReActAgent

logger = logging.getLogger(__name__)

_REASON_TO_STATUS = {
    ReplyFinishedReason.COMPLETED: RunStatus.COMPLETED,
    ReplyFinishedReason.INTERRUPTED: RunStatus.CANCELLED,
    ReplyFinishedReason.ERROR: RunStatus.ERROR,
    ReplyFinishedReason.EXCEED_MAX_ITERS: RunStatus.COMPLETED,
}

_TRACE_STATUS = {
    RunStatus.COMPLETED: TraceRunStatus.COMPLETED,
    RunStatus.CANCELLED: TraceRunStatus.CANCELLED,
    RunStatus.ERROR: TraceRunStatus.ERROR,
}


class ReActRunner:
    """ReAct Agent 的核心执行引擎。"""

    def __init__(self, agent: "ReActAgent"):
        self.agent = agent
        self.state = RunState()
        self._run_task: asyncio.Task | None = None

    async def run(
        self,
        message,
        runtime_context: dict | None = None,
    ) -> AsyncGenerator[AgentStreamEvent, None]:
        """启动一次完整的 ReAct 执行。"""
        # 并发锁
        if self._run_task is not None and not self._run_task.done():
            raise RuntimeError("Agent is already running")

        self._run_task = asyncio.current_task()
        self.state.runtime_context = runtime_context or {}
        self.state.runtime_context.setdefault(
            "max_iterations", self.agent.max_iterations,
        )
        self.state.start()

        # 准备 tracing
        trace_metadata = self.state.runtime_context.get("trace_metadata") or {}
        if not isinstance(trace_metadata, dict):
            trace_metadata = {"value": trace_metadata}
        trace_metadata = {
            "model": self.agent.model,
            "api_type": self.agent.api_type,
            **trace_metadata,
        }
        trace_tags = self.state.runtime_context.get("trace_tags") or []
        if isinstance(trace_tags, str):
            trace_tags = [trace_tags]

        self.state.trace_span = self.agent.tracer.start_run(
            str(self.state.runtime_context.get("trace_name") or "react_agent"),
            RunType.AGENT,
            inputs={"message": message},
            metadata=trace_metadata,
            tags=list(trace_tags),
        )

        # 写入用户消息到 memory
        if isinstance(message, str):
            self.agent.memory.add_user(message)
        else:
            for msg in message:
                self.agent.memory.add_raw(msg)

        # ReplyStart（只产一次）
        reply_id = uuid.uuid4().hex[:16]
        self.state.reply_id = reply_id
        session_id = self.state.runtime_context.get("session_id", "")
        model_name = self.agent.model

        yield ReplyStartEvent(
            session_id=session_id, reply_id=reply_id, name=model_name,
        )

        # 主循环
        try:
            async for event in self._loop():
                yield event
        except asyncio.CancelledError:
            self._finalize(ReplyFinishedReason.INTERRUPTED)
            yield ReplyEndEvent(
                session_id=session_id, reply_id=reply_id,
                finished_reason=ReplyFinishedReason.INTERRUPTED,
            )
        except Exception:
            self._finalize(ReplyFinishedReason.ERROR)
            yield ReplyEndEvent(
                session_id=session_id, reply_id=reply_id,
                finished_reason=ReplyFinishedReason.ERROR,
                error={"message": str(self.state.error or "Unknown error")},
            )
            raise
        finally:
            # Tracing 收尾
            if self.state.trace_span and not self.state.trace_span.ended:
                self.state.trace_span.end(
                    status=_TRACE_STATUS.get(self.state.status, TraceRunStatus.ERROR),
                    outputs={
                        "success": self.state.status == RunStatus.COMPLETED,
                        "done_reason": self.state.done_reason,
                        "iterations": self.state.iteration,
                    },
                    error=self.state.error if self.state.status == RunStatus.ERROR else None,
                )
            self._run_task = None

    def cancel_nowait(self) -> None:
        """外部调用：取消当前执行。"""
        if self._run_task is not None and not self._run_task.done():
            self._run_task.cancel()

    async def _loop(self) -> AsyncGenerator[AgentStreamEvent, None]:
        """ReAct 主循环。"""
        prev: TurnResult | None = None

        max_iters = self.agent.max_iterations
        reasoning_executor = ReasoningExecutor(
            self.agent, self.state, self.agent.runner.llm, self.agent.hook_manager,
        )
        # 需要访问 LLMHandler — 通过 self.agent.runner.llm 获取
        # 但 runner 就是 self，所以直接用 self.llm
        reasoning_executor.llm = self.llm

        acting_executor = ActingExecutor(
            self.agent, self.state, self.tool_handler,
        )

        exit_executor = ExitExecutor(
            self.agent, self.state, self.agent.hook_manager,
        )

        try:
            while max_iters is None or self.state.iteration < max_iters:
                self.state.iteration += 1

                # on_turn_start hook
                await self._trigger_on_turn_start()

                action = decide(self.state, prev)

                if isinstance(action, Reasoning):
                    async for event in reasoning_executor.stream(action):
                        yield event
                    prev = reasoning_executor.result

                elif isinstance(action, Acting):
                    async for event in acting_executor.stream(action):
                        yield event
                    prev = None

                elif isinstance(action, Exit):
                    async for event in exit_executor.stream(action):
                        yield event
                    if exit_executor.outcome.should_continue:
                        prev = None
                        continue
                    return

            # 达到 max_iterations
            if not self.state.is_done:
                self._finalize(ReplyFinishedReason.EXCEED_MAX_ITERS)
                yield ReplyEndEvent(
                    session_id=self.state.runtime_context.get("session_id", ""),
                    reply_id=self.state.reply_id,
                    finished_reason=ReplyFinishedReason.EXCEED_MAX_ITERS,
                )

        except CancelledError:
            self._finalize(ReplyFinishedReason.INTERRUPTED)
            yield ReplyEndEvent(
                session_id=self.state.runtime_context.get("session_id", ""),
                reply_id=self.state.reply_id,
                finished_reason=ReplyFinishedReason.INTERRUPTED,
            )

    def _finalize(self, reason: ReplyFinishedReason) -> None:
        """统一终态写入。"""
        self.state.done_reason = reason
        self.state.status = _REASON_TO_STATUS.get(reason, RunStatus.ERROR)

    async def _trigger_on_turn_start(self) -> None:
        """触发 on_turn_start hook。"""
        from ...hooks import ON_TURN_START, TurnStartInput

        ts_output = await self.agent.hook_manager.trigger(
            ON_TURN_START,
            lambda: TurnStartInput(
                session_id=self.state.runtime_context.get("session_id", ""),
                turn_id=self.state.turn_id,
                iteration=self.state.iteration,
                messages=self.agent.memory.get_messages(),
                runtime_context=self.state.runtime_context,
            ),
        )
        if ts_output is not None:
            from ...hooks import TurnStartOutput
            if isinstance(ts_output, TurnStartOutput):
                for msg in ts_output.inject_messages:
                    self.agent.memory.add_raw(msg)

    @property
    def llm(self):
        """LLMHandler 实例。"""
        return self._llm

    @property
    def tool_handler(self):
        """ToolHandler 实例。"""
        return self._tool_handler
```

Note: The `ReActRunner.__init__` needs to create `LLMHandler` and `ToolHandler` (same as the old code). Add these to `__init__`:

```python
def __init__(self, agent: "ReActAgent"):
    self.agent = agent
    self.state = RunState()
    self._run_task: asyncio.Task | None = None
    self._llm = LLMHandler(
        agent.model, agent.api_key, agent.api_base, agent.api_type,
        max_tokens=agent.max_tokens, reasoning_effort=agent.reasoning_effort,
    )
    self._tool_handler = ToolHandler(agent.tool_registry, agent.hook_manager)
```

And add the necessary imports at the top:
```python
from ...llm import LLMHandler
from .tool_handler import ToolHandler
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd E:\ftre-agent-core && python -m pytest tests/test_react_runner_new.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
cd E:\ftre-agent-core && git add src/ftre_agent_core/agent/runner/react_runner.py tests/test_react_runner_new.py && git commit -m "feat(runner): rewrite react_runner.py with state machine architecture"
```

---

### Task 8: 更新 `__init__.py` 导出

**Files:**
- Modify: `src/ftre_agent_core/agent/runner/__init__.py`
- Modify: `src/ftre_agent_core/agent/__init__.py`

**Interfaces:**
- Consumes: All previous tasks
- Produces: Updated public API exports

- [ ] **Step 1: Update runner/__init__.py**

```python
# src/ftre_agent_core/agent/runner/__init__.py
from ._state import RunState, RunStatus, CancelledError
from ._actions import Reasoning, Acting, Exit, TurnResult, ExitOutcome
from ._decide import decide
from .react_runner import ReActRunner
from .tool_handler import ToolHandler, ToolResult

__all__ = [
    "RunState",
    "RunStatus",
    "CancelledError",
    "Reasoning",
    "Acting",
    "Exit",
    "TurnResult",
    "ExitOutcome",
    "decide",
    "ReActRunner",
    "ToolHandler",
    "ToolResult",
]
```

- [ ] **Step 2: Update agent/__init__.py**

Add the new exports to the existing `__all__`:

```python
# In the from .runner import (...) block, add:
    "Reasoning",
    "Acting",
    "Exit",
    "TurnResult",
    "ExitOutcome",
    "decide",
```

- [ ] **Step 3: Run existing tests to check for import breakage**

Run: `cd E:\ftre-agent-core && python -m pytest tests/ -v --tb=short -x`
Expected: Some old tests may fail due to behavior changes (length continuation, ReplyEnd per turn). Note which fail.

- [ ] **Step 4: Commit**

```bash
cd E:\ftre-agent-core && git add src/ftre_agent_core/agent/runner/__init__.py src/ftre_agent_core/agent/__init__.py && git commit -m "refactor(runner): update __init__.py exports for new state machine"
```

---

### Task 9: 适配测试 `test_react_runner_continuation.py`

**Files:**
- Modify: `tests/test_react_runner_continuation.py`

**Interfaces:**
- Consumes: New ReActRunner behavior

- [ ] **Step 1: Update tests**

The following changes are needed:
1. Remove `test_length_finish_adds_hidden_user_continuation` — length continuation feature removed
2. Update other tests: no more per-turn `ReplyEndEvent` — only one at the end
3. Update empty response tests: `ReplyEndEvent` only at the end, not per turn
4. Verify `on_turn_end` behavior unchanged (only on COMPLETED)

Replace the entire test file with:

```python
"""
ReActRunner continuation / retry 逻辑测试（状态机重构版）。

length 截断续写已删除。空响应重试和 on_stop hook 由新状态机处理。
"""
import logging

import pytest

from ftre_agent_core.agent import EventType, ReActAgent
from ftre_agent_core.event import ReplyFinishedReason
from ftre_agent_core.llm import ReasoningDelta, StepFinish, TextDelta, ToolCall
from ftre_agent_core.tool import tool, ToolRegistry


def make_agent(tools=None, max_iterations=3):
    registry = ToolRegistry()
    for t in (tools or []):
        registry.register(t)
    return ReActAgent(
        model="fake",
        api_key="fake",
        system_prompt="test",
        tool_registry=registry,
        max_iterations=max_iterations,
    )


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def test_step_finish_defaults_to_unknown():
    assert StepFinish().finish_reason == "unknown"


@pytest.mark.asyncio
async def test_reasoning_only_turn_is_treated_as_empty_response_retry():
    agent = make_agent()
    calls = 0

    async def fake_stream(messages, tools=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield ReasoningDelta(text="need more work")
            yield StepFinish(finish_reason="stop")
        else:
            yield TextDelta(text="done")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    events = [event async for event in agent.run("start")]

    assert calls == 2
    assert agent.state.done_reason == ReplyFinishedReason.COMPLETED


@pytest.mark.asyncio
async def test_empty_response_retries_then_requests_finalization_without_tools():
    @tool(description="Echo text")
    def echo(text: str) -> str:
        return f"echo:{text}"

    agent = make_agent(tools=[echo], max_iterations=5)
    calls = 0

    async def fake_stream(messages, tools=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            assert tools is not None
            yield StepFinish(finish_reason="stop")
        elif calls == 2:
            assert tools is not None
            yield TextDelta(text=" \n ")
            yield StepFinish(finish_reason="stop")
        else:
            assert tools is None
            assert messages[-1]["role"] == "user"
            assert "直接给出回复用户的最终内容" in _content_text(
                messages[-1]["content"]
            )
            yield TextDelta(text="final")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    events = [event async for event in agent.run("start")]

    assert calls == 3
    assert agent.state.done_reason == ReplyFinishedReason.COMPLETED


@pytest.mark.asyncio
async def test_unknown_finish_with_text_continues(caplog):
    agent = make_agent(max_iterations=3)
    calls = 0

    async def fake_stream(messages, tools=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield TextDelta(text="hello")
            yield StepFinish(finish_reason="unknown")
        else:
            yield TextDelta(text="done")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    with caplog.at_level(logging.INFO):
        events = [event async for event in agent.run("start")]

    assert calls == 2
    assert agent.state.done_reason == ReplyFinishedReason.COMPLETED


@pytest.mark.asyncio
async def test_unknown_finish_with_empty_response_continues_without_finalization():
    agent = make_agent(max_iterations=5)
    calls = 0

    async def fake_stream(messages, tools=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield StepFinish(finish_reason="unknown")
        elif calls == 2:
            yield TextDelta(text="done")
            yield StepFinish(finish_reason="stop")
        else:
            pytest.fail("should not reach 3rd call")

    agent.runner.llm.stream = fake_stream

    events = [event async for event in agent.run("start")]

    assert calls == 2


@pytest.mark.asyncio
async def test_tool_call_turn_produces_tool_result_events():
    @tool(description="Echo text")
    def echo(text: str) -> str:
        return f"echo:{text}"

    agent = make_agent(tools=[echo])
    calls = 0

    async def fake_stream(messages, tools=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield ToolCall(id="call_echo", name="echo", input={"text": "x"})
            yield StepFinish(finish_reason="tool_calls")
        else:
            yield TextDelta(text="finished")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    events = [event async for event in agent.run("start")]

    assert calls == 2
    tool_results = [e for e in events if e.type == EventType.TOOL_RESULT_END]
    assert len(tool_results) == 1
    assert agent.state.done_reason == ReplyFinishedReason.COMPLETED


@pytest.mark.asyncio
async def test_multi_tool_call_events_are_emitted_before_results():
    @tool(description="Echo text")
    def echo(text: str) -> str:
        return f"echo:{text}"

    @tool(description="Uppercase text")
    def upper(text: str) -> str:
        return text.upper()

    agent = make_agent(tools=[echo, upper])
    calls = 0

    async def fake_stream(messages, tools=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield ToolCall(id="call_echo", name="echo", input={"text": "x"})
            yield ToolCall(id="call_upper", name="upper", input={"text": "y"})
            yield StepFinish(finish_reason="tool_calls")
        else:
            yield TextDelta(text="finished")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    events = [event async for event in agent.run("start")]

    assert calls == 2
    tool_results = [e for e in events if e.type == EventType.TOOL_RESULT_END]
    assert len(tool_results) == 2


@pytest.mark.asyncio
async def test_single_reply_start_and_reply_end():
    """验证一次 run() 只产一对 ReplyStart/ReplyEnd。"""
    agent = make_agent(max_iterations=3)
    calls = 0

    async def fake_stream(messages, tools=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield ToolCall(id="c1", name="nonexistent", input={})
            yield StepFinish(finish_reason="tool_calls")
        else:
            yield TextDelta(text="done")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    events = [event async for event in agent.run("start")]

    types = [e.type for e in events]
    assert types.count(EventType.REPLY_START) == 1
    assert types.count(EventType.REPLY_END) == 1
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd E:\ftre-agent-core && python -m pytest tests/test_react_runner_continuation.py -v`
Expected: PASS (7 tests)

- [ ] **Step 3: Run full test suite**

Run: `cd E:\ftre-agent-core && python -m pytest tests/ -v --tb=short`
Expected: All tests pass. If any fail, investigate and fix.

- [ ] **Step 4: Commit**

```bash
cd E:\ftre-agent-core && git add tests/test_react_runner_continuation.py && git commit -m "test(runner): adapt continuation tests for state machine refactor"
```

---

### Task 10: 清理旧代码与最终验证

**Files:**
- Modify: `src/ftre_agent_core/agent/runner/react_runner.py` (remove any dead code)
- Modify: `src/ftre_agent_core/agent/react.py` (verify compatibility)

- [ ] **Step 1: Verify react.py compatibility**

Check that `ReActAgent` still works with the new runner. The key properties used:
- `agent.runner` → returns ReActRunner
- `agent.runner.llm` → returns LLMHandler (now `agent.runner._llm`)
- `agent.runner.tool_handler` → returns ToolHandler (now `agent.runner._tool_handler`)
- `agent.runner.state` → returns RunState
- `agent.cancel_nowait()` → calls `self._runner.cancel()`

Verify `react.py` doesn't reference removed attributes like `state.reply_msg`, `state.cancel_token.cancel()`, `state.force_no_tools_once`, etc.

Run: `cd E:\ftre-agent-core && python -c "from ftre_agent_core.agent import ReActAgent; a = ReActAgent(model='test', api_key='test'); print('OK')"`

- [ ] **Step 2: Run full test suite**

Run: `cd E:\ftre-agent-core && python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 3: Check for leftover references to old patterns**

Run: `cd E:\ftre-agent-core && findstr /S /N "force_no_tools_once\|finalization_retrying\|reply_msg\|LENGTH_CONTINUATION_PROMPT" src\ftre_agent_core\*.py`
Expected: No matches in source code (test files may still reference old names if not fully updated)

- [ ] **Step 4: Commit**

```bash
cd E:\ftre-agent-core && git add -A && git commit -m "chore(runner): final cleanup and verification"
```
