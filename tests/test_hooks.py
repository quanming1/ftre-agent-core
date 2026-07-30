"""
Core Hook 系统测试。

覆盖：
  - FtreCoreHookManager 基础操作（register/unregister/has_hooks/clear）
  - trigger 链式执行（多 hook 顺序、block 终止、modify 传递、异常跳过）
  - on_stop hook 集成到 ReActRunner（阻止停止 → continuation → 最终停止）
  - on_turn_start hook 集成（注入消息）
  - on_turn_end hook 集成（只读观察）
"""
import asyncio
import pytest

from ftre_agent_core.agent import ReActAgent
from ftre_agent_core.event import ReplyFinishedReason
from ftre_agent_core.hooks import (
    FtreCoreHookManager,
    ON_TURN_START,
    ON_STOP,
    ON_TURN_END,
    HookInput,
    HookOutput,
    TurnStartInput,
    TurnStartOutput,
    StopInput,
    TurnEndInput,
)
from ftre_agent_core.llm import TextDelta, StepFinish, ToolCall


# ── 测试辅助 ──────────────────────────────────────────────────────

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

def make_agent(max_iterations=5, hook_manager=None):
    return ReActAgent(
        model="fake",
        api_key="fake",
        system_prompt="test",
        max_iterations=max_iterations,
        hook_manager=hook_manager,
    )


# ═══════════════════════════════════════════════════════════════════
# Part 1: FtreCoreHookManager 单元测试
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_trigger_no_hooks_returns_none():
    mgr = FtreCoreHookManager()
    result = await mgr.trigger(ON_STOP, StopInput())
    assert result is None


@pytest.mark.asyncio
async def test_single_hook_allow():
    mgr = FtreCoreHookManager()

    async def hook(inp: StopInput) -> HookOutput:
        return HookOutput(decision="allow")

    mgr.register(ON_STOP, hook)
    result = await mgr.trigger(ON_STOP, StopInput())
    assert result is not None
    assert result.decision == "allow"


@pytest.mark.asyncio
async def test_single_hook_block():
    mgr = FtreCoreHookManager()

    async def hook(inp: StopInput) -> HookOutput:
        return HookOutput(decision="block", reason="not done yet")

    mgr.register(ON_STOP, hook)
    result = await mgr.trigger(ON_STOP, StopInput())
    assert result is not None
    assert result.decision == "block"
    assert result.reason == "not done yet"


@pytest.mark.asyncio
async def test_multiple_hooks_block_terminates_chain():
    mgr = FtreCoreHookManager()
    call_order = []

    async def hook1(inp: StopInput) -> HookOutput:
        call_order.append("hook1")
        return HookOutput(decision="allow")

    async def hook2(inp: StopInput) -> HookOutput:
        call_order.append("hook2")
        return HookOutput(decision="block", reason="stop here")

    async def hook3(inp: StopInput) -> HookOutput:
        call_order.append("hook3")
        return HookOutput(decision="allow")

    mgr.register(ON_STOP, hook1)
    mgr.register(ON_STOP, hook2)
    mgr.register(ON_STOP, hook3)

    result = await mgr.trigger(ON_STOP, StopInput())
    assert call_order == ["hook1", "hook2"]
    assert result.decision == "block"
    assert result.reason == "stop here"


@pytest.mark.asyncio
async def test_hook_returning_none_is_allow():
    mgr = FtreCoreHookManager()
    called = []

    async def hook1(inp: StopInput) -> None:
        called.append("hook1")

    async def hook2(inp: StopInput) -> HookOutput:
        called.append("hook2")
        return HookOutput(decision="block", reason="hook2 blocks")

    mgr.register(ON_STOP, hook1)
    mgr.register(ON_STOP, hook2)
    result = await mgr.trigger(ON_STOP, StopInput())
    assert called == ["hook1", "hook2"]
    assert result.decision == "block"


@pytest.mark.asyncio
async def test_hook_exception_skipped():
    mgr = FtreCoreHookManager()

    async def bad_hook(inp: StopInput) -> HookOutput:
        raise RuntimeError("boom")

    async def good_hook(inp: StopInput) -> HookOutput:
        return HookOutput(decision="block", reason="after bad hook")

    mgr.register(ON_STOP, bad_hook)
    mgr.register(ON_STOP, good_hook)
    result = await mgr.trigger(ON_STOP, StopInput())
    assert result is not None
    assert result.decision == "block"
    assert result.reason == "after bad hook"


@pytest.mark.asyncio
async def test_sync_hook_supported():
    mgr = FtreCoreHookManager()

    def sync_hook(inp: StopInput) -> HookOutput:
        return HookOutput(decision="block", reason="sync block")

    mgr.register(ON_STOP, sync_hook)
    result = await mgr.trigger(ON_STOP, StopInput())
    assert result.decision == "block"
    assert result.reason == "sync block"


@pytest.mark.asyncio
async def test_unregister():
    mgr = FtreCoreHookManager()

    async def hook(inp: StopInput) -> HookOutput:
        return HookOutput(decision="block")

    mgr.register(ON_STOP, hook)
    assert mgr.has_hooks(ON_STOP)

    removed = mgr.unregister(ON_STOP, hook)
    assert removed
    assert not mgr.has_hooks(ON_STOP)

    result = await mgr.trigger(ON_STOP, StopInput())
    assert result is None


@pytest.mark.asyncio
async def test_clear_specific_point():
    mgr = FtreCoreHookManager()

    async def hook(inp: StopInput) -> HookOutput:
        return HookOutput(decision="block")

    mgr.register(ON_STOP, hook)
    mgr.register(ON_TURN_START, hook)
    assert mgr.has_hooks(ON_STOP)
    assert mgr.has_hooks(ON_TURN_START)

    mgr.clear(ON_STOP)
    assert not mgr.has_hooks(ON_STOP)
    assert mgr.has_hooks(ON_TURN_START)


@pytest.mark.asyncio
async def test_clear_all():
    mgr = FtreCoreHookManager()

    async def hook(inp: StopInput) -> HookOutput:
        return HookOutput(decision="block")

    mgr.register(ON_STOP, hook)
    mgr.register(ON_TURN_START, hook)
    mgr.clear()
    assert not mgr.has_hooks(ON_STOP)
    assert not mgr.has_hooks(ON_TURN_START)


@pytest.mark.asyncio
async def test_register_non_callable_raises():
    mgr = FtreCoreHookManager()
    with pytest.raises(TypeError):
        mgr.register(ON_STOP, "not a function")


@pytest.mark.asyncio
async def test_trigger_factory_not_called_when_no_hooks():
    """没有 hook 时，factory 不应被调用（避免无谓的 input 构造开销）。"""
    mgr = FtreCoreHookManager()
    factory_called = False

    def factory() -> StopInput:
        nonlocal factory_called
        factory_called = True
        return StopInput(last_assistant_text="expensive to build")

    result = await mgr.trigger(ON_STOP, factory)
    assert result is None
    assert factory_called is False


@pytest.mark.asyncio
async def test_trigger_factory_called_when_hooks_exist():
    """有 hook 时，factory 被调用一次，hook 收到构造好的 input。"""
    mgr = FtreCoreHookManager()
    factory_called = False
    received_text = ""

    async def hook(inp: StopInput) -> HookOutput:
        nonlocal received_text
        received_text = inp.last_assistant_text
        return HookOutput(decision="allow")

    mgr.register(ON_STOP, hook)

    def factory() -> StopInput:
        nonlocal factory_called
        factory_called = True
        return StopInput(last_assistant_text="built on demand")

    result = await mgr.trigger(ON_STOP, factory)
    assert factory_called is True
    assert received_text == "built on demand"
    assert result is not None
    assert result.decision == "allow"


# ═══════════════════════════════════════════════════════════════════
# Part 2: on_stop hook 集成测试
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_on_stop_hook_blocks_and_continues():
    """on_stop hook 返回 block → Agent 被阻止停止，注入 continuation prompt 后继续。"""
    mgr = FtreCoreHookManager()
    agent = make_agent(max_iterations=5, hook_manager=mgr)
    call_count = 0
    hook_calls = 0

    async def fake_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield TextDelta(text="working on it")
            yield StepFinish(finish_reason="stop")
        else:
            # 第二轮：continuation prompt 已注入，Agent 应该能看到它
            assert messages[-1]["role"] == "user"
            assert "keep going" in _content_text(messages[-1]["content"])
            yield TextDelta(text="done")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    async def stop_hook(inp: StopInput) -> HookOutput:
        nonlocal hook_calls
        hook_calls += 1
        if hook_calls == 1:
            return HookOutput(decision="block", reason="keep going")
        return HookOutput(decision="allow")

    mgr.register(ON_STOP, stop_hook)

    events = [e async for e in agent.run("start")]

    assert call_count == 2
    assert hook_calls == 2

    # memory 中应该有：user(start) → assistant(working) → user(continuation) → assistant(done)
    msgs = agent.messages
    assert msgs[0]["role"] == "user"
    assert _content_text(msgs[0]["content"]) == "start"
    assert msgs[1]["role"] == "assistant"
    assert msgs[2]["role"] == "user"
    assert "keep going" in _content_text(msgs[2]["content"])
    assert msgs[3]["role"] == "assistant"

    # 验证最终状态（agent-core 不再产出 Step 事件，通过 state 检查）
    assert agent.run_state.done_reason == ReplyFinishedReason.COMPLETED


@pytest.mark.asyncio
async def test_on_stop_hook_allow_lets_agent_stop():
    """on_stop hook 返回 allow → Agent 正常停止。"""
    mgr = FtreCoreHookManager()
    agent = make_agent(max_iterations=3, hook_manager=mgr)
    hook_calls = 0

    async def fake_stream(messages, tools=None):
        yield TextDelta(text="final answer")
        yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    async def stop_hook(inp: StopInput) -> HookOutput:
        nonlocal hook_calls
        hook_calls += 1
        assert inp.last_assistant_text == "final answer"
        return HookOutput(decision="allow")

    mgr.register(ON_STOP, stop_hook)
    events = [e async for e in agent.run("start")]

    assert hook_calls == 1
    assert agent.run_state.done_reason == ReplyFinishedReason.COMPLETED


@pytest.mark.asyncio
async def test_on_stop_hook_not_called_when_tool_calls_present():
    """有工具调用时 on_stop 不应触发（Agent 还没想停下）。"""
    from ftre_agent_core.tool import tool

    @tool(description="echo")
    def echo(text: str) -> str:
        return f"echo:{text}"

    mgr = FtreCoreHookManager()
    agent = make_agent(max_iterations=3, hook_manager=mgr)
    agent._registry.register(echo)
    call_count = 0
    hook_calls = 0

    async def fake_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield ToolCall(id="c1", name="echo", input={"text": "hi"})
            yield StepFinish(finish_reason="tool_calls")
        else:
            yield TextDelta(text="final")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    async def stop_hook(inp: StopInput) -> HookOutput:
        nonlocal hook_calls
        hook_calls += 1
        return HookOutput(decision="allow")

    mgr.register(ON_STOP, stop_hook)
    events = [e async for e in agent.run("start")]

    assert call_count == 2
    assert hook_calls == 1  # 只在第二轮（无工具调用）时触发


@pytest.mark.asyncio
async def test_on_stop_hook_system_message():
    """block 时 system_message 可以同时传递（用于 UI 通知）。"""
    mgr = FtreCoreHookManager()
    agent = make_agent(max_iterations=5, hook_manager=mgr)
    call_count = 0

    async def fake_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield TextDelta(text="first")
            yield StepFinish(finish_reason="stop")
        else:
            yield TextDelta(text="final")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    async def stop_hook(inp: StopInput) -> HookOutput:
        if call_count == 1:
            return HookOutput(
                decision="block",
                reason="continue",
                system_message="Goal still active",
            )
        return HookOutput(decision="allow")

    mgr.register(ON_STOP, stop_hook)
    events = [e async for e in agent.run("start")]

    assert call_count == 2


# ═══════════════════════════════════════════════════════════════════
# Part 3: on_turn_start hook 集成测试
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_on_turn_start_injects_message():
    """on_turn_start hook 注入消息 → Agent 在本轮迭代中能看到。"""
    mgr = FtreCoreHookManager()
    agent = make_agent(max_iterations=3, hook_manager=mgr)
    call_count = 0
    seen_messages = []

    async def fake_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        # 捕获本轮 LLM 看到的消息
        seen_messages.append([dict(m) for m in messages])
        if call_count == 1:
            # 第一轮：应该能看到注入的消息
            roles = [m["role"] for m in messages]
            assert "system" in roles  # system_prompt
            assert messages[-1]["role"] == "user"  # injected message
            yield TextDelta(text="ack")
            yield StepFinish(finish_reason="stop")
        else:
            yield TextDelta(text="done")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    async def turn_start_hook(inp: TurnStartInput) -> TurnStartOutput:
        return TurnStartOutput(
            inject_messages=[{"role": "user", "content": "reminder: check tests"}]
        )

    mgr.register(ON_TURN_START, turn_start_hook)
    events = [e async for e in agent.run("start")]

    assert call_count == 1
    # 注入的消息应该在 memory 中
    contents = [
        _content_text(m.get("content"))
        for m in agent.messages
        if m["role"] == "user"
    ]
    assert "reminder: check tests" in contents


@pytest.mark.asyncio
async def test_on_turn_start_no_injection_when_no_hook():
    """没有 on_turn_start hook 时，行为与原来完全一致。"""
    mgr = FtreCoreHookManager()
    agent = make_agent(max_iterations=3, hook_manager=mgr)

    async def fake_stream(messages, tools=None):
        yield TextDelta(text="hello")
        yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream
    events = [e async for e in agent.run("start")]

    # 只有一条用户消息（原始输入）
    user_msgs = [m for m in agent.messages if m["role"] == "user"]
    assert len(user_msgs) == 1
    assert _content_text(user_msgs[0]["content"]) == "start"


# ═══════════════════════════════════════════════════════════════════
# Part 4: on_turn_end hook 集成测试
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_on_turn_end_called_on_completion():
    """Agent 正常完成时 on_turn_end 被调用，携带 done_reason。"""
    mgr = FtreCoreHookManager()
    agent = make_agent(max_iterations=3, hook_manager=mgr)
    turn_end_calls = []

    async def fake_stream(messages, tools=None):
        yield TextDelta(text="done")
        yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    async def turn_end_hook(inp: TurnEndInput) -> HookOutput:
        turn_end_calls.append(inp)
        return None  # on_turn_end 的返回值被忽略

    mgr.register(ON_TURN_END, turn_end_hook)
    events = [e async for e in agent.run("start")]

    assert len(turn_end_calls) == 1
    assert turn_end_calls[0].done_reason == ReplyFinishedReason.COMPLETED
    assert turn_end_calls[0].iteration == 1


@pytest.mark.asyncio
async def test_on_turn_end_called_after_tool_iteration():
    """工具迭代完成后 on_turn_end 也被调用。"""
    from ftre_agent_core.tool import tool

    @tool(description="echo")
    def echo(text: str) -> str:
        return f"echo:{text}"

    mgr = FtreCoreHookManager()
    agent = make_agent(max_iterations=3, hook_manager=mgr)
    agent._registry.register(echo)
    call_count = 0
    turn_end_calls = []

    async def fake_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield ToolCall(id="c1", name="echo", input={"text": "hi"})
            yield StepFinish(finish_reason="tool_calls")
        else:
            yield TextDelta(text="final")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    async def turn_end_hook(inp: TurnEndInput) -> HookOutput:
        turn_end_calls.append(inp)
        return None

    mgr.register(ON_TURN_END, turn_end_hook)
    events = [e async for e in agent.run("start")]

    assert call_count == 2
    assert len(turn_end_calls) == 1
    assert turn_end_calls[0].done_reason == ReplyFinishedReason.COMPLETED
    assert turn_end_calls[0].iteration == 2


# ═══════════════════════════════════════════════════════════════════
# Part 5: 综合 — 多 hook 叠加 + goal 模拟
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_multiple_stop_hooks_chain():
    """两个 on_stop hook：第一个 allow，第二个 block → 最终 block。"""
    mgr = FtreCoreHookManager()
    agent = make_agent(max_iterations=5, hook_manager=mgr)
    call_count = 0

    async def fake_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield TextDelta(text="attempt 1")
            yield StepFinish(finish_reason="stop")
        elif call_count == 2:
            yield TextDelta(text="attempt 2")
            yield StepFinish(finish_reason="stop")
        else:
            yield TextDelta(text="done")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    async def logger_hook(inp: StopInput) -> HookOutput:
        # 第一个 hook 只记录，不阻止
        return HookOutput(decision="allow")

    async def goal_hook(inp: StopInput) -> HookOutput:
        # 第二个 hook 前 2 次阻止，第 3 次放行
        if call_count < 3:
            return HookOutput(decision="block", reason=f"continue (attempt {call_count})")
        return HookOutput(decision="allow")

    mgr.register(ON_STOP, logger_hook)
    mgr.register(ON_STOP, goal_hook)

    events = [e async for e in agent.run("start")]

    assert call_count == 3
    # memory 中应该有 2 条 continuation prompt
    user_msgs = [m for m in agent.messages if m["role"] == "user"]
    # 1 个原始 + 2 个 continuation
    assert len(user_msgs) == 3
    assert "continue (attempt 1)" in _content_text(user_msgs[1]["content"])
    assert "continue (attempt 2)" in _content_text(user_msgs[2]["content"])


@pytest.mark.asyncio
async def test_goal_simulation_block_twice_then_allow():
    """模拟 /goal：block 2 次后第 3 次 allow，验证迭代计数和最终状态。"""
    mgr = FtreCoreHookManager()
    agent = make_agent(max_iterations=10, hook_manager=mgr)
    call_count = 0
    hook_iterations = []

    async def fake_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        texts = ["still building", "added tests", "verified working"]
        yield TextDelta(text=texts[min(call_count - 1, 2)])
        yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    async def goal_hook(inp: StopInput) -> HookOutput:
        hook_iterations.append(inp.iteration)
        if inp.iteration < 3:
            return HookOutput(
                decision="block",
                reason="Goal not yet achieved, keep working.",
            )
        return HookOutput(decision="allow")

    mgr.register(ON_STOP, goal_hook)
    events = [e async for e in agent.run("build a blog")]

    assert call_count == 3
    assert hook_iterations == [1, 2, 3]

    assert agent.run_state.done_reason == ReplyFinishedReason.COMPLETED
    assert agent.run_state.iteration == 3


# ═══════════════════════════════════════════════════════════════════
# Part 6: on_pre_tool hook 集成测试
# ═══════════════════════════════════════════════════════════════════

from ftre_agent_core.hooks import (
    ON_PRE_TOOL,
    ON_POST_TOOL,
    PreToolInput,
    PreToolOutput,
    PostToolInput,
    PostToolOutput,
)
from ftre_agent_core.event import ToolResultEndEvent, ToolResultTextDeltaEvent


def _tool_result_text(events) -> str:
    return "".join(
        event.delta
        for event in events
        if isinstance(event, ToolResultTextDeltaEvent)
    )
from ftre_agent_core.tool import tool


@pytest.mark.asyncio
async def test_on_pre_tool_block_prevents_execution():
    """on_pre_tool block → 工具不执行，reason 作为 tool_result 返回给 Agent。"""
    mgr = FtreCoreHookManager()
    agent = make_agent(max_iterations=3, hook_manager=mgr)
    execution_count = 0

    @tool(description="echo")
    def echo(text: str) -> str:
        nonlocal execution_count
        execution_count += 1
        return f"echo:{text}"

    agent._registry.register(echo)
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

    async def block_hook(inp: PreToolInput) -> PreToolOutput:
        return PreToolOutput(decision="block", reason="禁止执行此工具")

    mgr.register(ON_PRE_TOOL, block_hook)
    events = [e async for e in agent.run("start")]

    # 工具没被执行
    assert execution_count == 0

    # Agent 应该看到拦截 reason 作为 tool_result
    tool_results = [e for e in events if isinstance(e, ToolResultEndEvent)]
    assert len(tool_results) == 1
    assert "禁止执行此工具" in _tool_result_text(events)
    assert tool_results[0].state == "error"


@pytest.mark.asyncio
async def test_on_pre_tool_modify_args():
    """on_pre_tool modify → 替换参数后执行。"""
    mgr = FtreCoreHookManager()
    agent = make_agent(max_iterations=3, hook_manager=mgr)

    @tool(description="echo")
    def echo(text: str) -> str:
        return f"echo:{text}"

    agent._registry.register(echo)
    call_count = 0

    async def fake_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield ToolCall(id="c1", name="echo", input={"text": "original"})
            yield StepFinish(finish_reason="tool_calls")
        else:
            yield TextDelta(text="done")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    async def modify_hook(inp: PreToolInput) -> PreToolOutput:
        return PreToolOutput(decision="modify", modified_args={"text": "modified"})

    mgr.register(ON_PRE_TOOL, modify_hook)
    events = [e async for e in agent.run("start")]

    tool_results = [e for e in events if isinstance(e, ToolResultEndEvent)]
    assert len(tool_results) == 1
    assert _tool_result_text(events) == "echo:modified"


@pytest.mark.asyncio
async def test_on_pre_tool_allow_executes_normally():
    """on_pre_tool allow → 工具正常执行。"""
    mgr = FtreCoreHookManager()
    agent = make_agent(max_iterations=3, hook_manager=mgr)

    @tool(description="echo")
    def echo(text: str) -> str:
        return f"echo:{text}"

    agent._registry.register(echo)
    call_count = 0

    async def fake_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield ToolCall(id="c1", name="echo", input={"text": "hello"})
            yield StepFinish(finish_reason="tool_calls")
        else:
            yield TextDelta(text="done")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    async def allow_hook(inp: PreToolInput) -> PreToolOutput:
        assert inp.tool_name == "echo"
        assert inp.tool_args == {"text": "hello"}
        return PreToolOutput(decision="allow")

    mgr.register(ON_PRE_TOOL, allow_hook)
    events = [e async for e in agent.run("start")]

    tool_results = [e for e in events if isinstance(e, ToolResultEndEvent)]
    assert len(tool_results) == 1
    assert _tool_result_text(events) == "echo:hello"


# ═══════════════════════════════════════════════════════════════════
# Part 7: on_post_tool hook 集成测试
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_on_post_tool_modify_result():
    """on_post_tool modify → 替换工具返回的结果字符串。"""
    mgr = FtreCoreHookManager()
    agent = make_agent(max_iterations=3, hook_manager=mgr)

    @tool(description="echo")
    def echo(text: str) -> str:
        return f"echo:{text}"

    agent._registry.register(echo)
    call_count = 0

    async def fake_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield ToolCall(id="c1", name="echo", input={"text": "secret"})
            yield StepFinish(finish_reason="tool_calls")
        else:
            yield TextDelta(text="done")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    async def redact_hook(inp: PostToolInput) -> PostToolOutput:
        assert inp.tool_name == "echo"
        assert inp.result == "echo:secret"
        return PostToolOutput(
            decision="modify",
            modified_result="echo:***REDACTED***",
        )

    mgr.register(ON_POST_TOOL, redact_hook)
    events = [e async for e in agent.run("start")]

    tool_results = [e for e in events if isinstance(e, ToolResultEndEvent)]
    assert len(tool_results) == 1
    assert _tool_result_text(events) == "echo:***REDACTED***"


@pytest.mark.asyncio
async def test_on_post_tool_allow_keeps_original():
    """on_post_tool allow → 结果原样返回。"""
    mgr = FtreCoreHookManager()
    agent = make_agent(max_iterations=3, hook_manager=mgr)

    @tool(description="echo")
    def echo(text: str) -> str:
        return "original_result"

    agent._registry.register(echo)
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

    async def observe_hook(inp: PostToolInput) -> PostToolOutput:
        # 只观察，不改
        assert inp.result == "original_result"
        assert inp.status == "completed"
        assert inp.error is None
        return PostToolOutput(decision="allow")

    mgr.register(ON_POST_TOOL, observe_hook)
    events = [e async for e in agent.run("start")]

    tool_results = [e for e in events if isinstance(e, ToolResultEndEvent)]
    assert len(tool_results) == 1
    assert _tool_result_text(events) == "original_result"


@pytest.mark.asyncio
async def test_on_post_tool_sees_error_on_failure():
    """工具执行失败时 on_post_tool 也能看到 error 和 status。"""
    mgr = FtreCoreHookManager()
    agent = make_agent(max_iterations=3, hook_manager=mgr)

    @tool(description="always fails")
    def fail_tool() -> str:
        raise RuntimeError("boom")

    agent._registry.register(fail_tool)
    call_count = 0

    async def fake_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield ToolCall(id="c1", name="fail_tool", input={})
            yield StepFinish(finish_reason="tool_calls")
        else:
            yield TextDelta(text="done")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    observed = []

    async def error_hook(inp: PostToolInput) -> PostToolOutput:
        observed.append(inp)
        return PostToolOutput(decision="allow")

    mgr.register(ON_POST_TOOL, error_hook)
    events = [e async for e in agent.run("start")]

    assert len(observed) == 1
    assert observed[0].status == "failed"
    assert observed[0].error is not None
    assert "boom" in observed[0].error


@pytest.mark.asyncio
async def test_pre_and_post_tool_both_registered():
    """pre 和 post 同时注册，按各自挂点独立链式执行。"""
    mgr = FtreCoreHookManager()
    agent = make_agent(max_iterations=3, hook_manager=mgr)

    @tool(description="echo")
    def echo(text: str) -> str:
        return f"echo:{text}"

    agent._registry.register(echo)
    call_count = 0

    async def fake_stream(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield ToolCall(id="c1", name="echo", input={"text": "original"})
            yield StepFinish(finish_reason="tool_calls")
        else:
            yield TextDelta(text="done")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    pre_seen = []
    post_seen = []

    async def pre_hook(inp: PreToolInput) -> PreToolOutput:
        pre_seen.append(inp.tool_args)
        return PreToolOutput(decision="modify", modified_args={"text": "from_pre"})

    mgr.register(ON_PRE_TOOL, pre_hook)

    async def post_hook(inp: PostToolInput) -> PostToolOutput:
        post_seen.append(inp.result)
        return PostToolOutput(decision="modify", modified_result="from_post")

    mgr.register(ON_POST_TOOL, post_hook)

    events = [e async for e in agent.run("start")]

    # pre 看到原始参数，改成了 from_pre
    assert pre_seen == [{"text": "original"}]

    # post 看到工具执行后的结果（echo:from_pre），改成了 from_post
    assert post_seen == ["echo:from_pre"]

    # 最终 Agent 拿到的是 post 修改后的结果
    tool_results = [e for e in events if isinstance(e, ToolResultEndEvent)]
    assert _tool_result_text(events) == "from_post"
