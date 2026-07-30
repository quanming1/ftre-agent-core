# -*- coding: utf-8 -*-
"""权限系统接线的集成测试：Acting 分流、挂起、恢复、拒绝、挂起期间拒绝新消息。"""
import pytest

from ftre_agent_core.agent.react import ReActAgent
from ftre_agent_core.agent.runner._execute_acting import ActingExecutor
from ftre_agent_core.agent.runner._state import Acting, RunState
from ftre_agent_core.event import (
    EventType,
    RequireUserConfirmEvent,
    UserConfirmResultEvent,
)
from ftre_agent_core.llm import ToolCall
from ftre_agent_core.message import ToolCallState, ToolResultState
from ftre_agent_core.message_context import MessageContext
from ftre_agent_core.permission import PermissionBehavior, PermissionEngine, PermissionRule
from ftre_agent_core.tool import tool, ToolRegistry


# ── 构造带 echo 工具的 agent，permission_context 由参数注入 ──
def make_agent(rules=None, default_behavior=None, engine=True):
    @tool(description="Echo text")
    def echo(text: str) -> str:
        return f"echo:{text}"

    registry = ToolRegistry()
    registry.register(echo)

    from ftre_agent_core.state import AgentState

    permission_context = {}
    if rules is not None:
        permission_context["permission_rules"] = [r.model_dump() for r in rules]
    if default_behavior is not None:
        permission_context["default_behavior"] = default_behavior.value
    state = AgentState(permission_context=permission_context)

    return ReActAgent(
        model="fake", api_key="fake", system_prompt="test",
        tool_registry=registry, max_iterations=5, state=state,
        permission_engine=PermissionEngine() if engine else None,
    )


def make_state():
    s = RunState()
    s.runtime_context = {"session_id": "s1", "max_iterations": 5}
    s.start()
    s.reply_id = "r1"
    return s


def _executor(agent, state):
    return ActingExecutor(
        agent, state, agent.runner.tool_handler, agent.permission_engine
    )


@pytest.mark.asyncio
async def test_all_allow_executes_normally():
    """全 ALLOW（规则放行）→ 正常执行，产出工具结果，无确认事件。"""
    rules = [PermissionRule(id="allow-echo", tool_name="echo", behavior=PermissionBehavior.ALLOW)]
    agent = make_agent(rules=rules, default_behavior=PermissionBehavior.ASK)
    state = make_state()

    tc = ToolCall(id="c1", name="echo", input={"text": "hi"})
    events = [e async for e in _executor(agent, state).stream(Acting(tool_calls=[tc]))]

    types = [e.type for e in events]
    assert EventType.TOOL_RESULT_END in types
    assert EventType.REQUIRE_USER_CONFIRM not in types
    assert "echo:hi" in str(agent.messages[-1].get("content", ""))


@pytest.mark.asyncio
async def test_ask_pauses_and_emits_confirm_event():
    """命中 ASK → 挂起：产出 RequireUserConfirmEvent，不执行工具，tool_call 置 ASKING。"""
    rules = [PermissionRule(id="ask-echo", tool_name="echo", behavior=PermissionBehavior.ASK)]
    agent = make_agent(rules=rules)
    state = make_state()

    tc = ToolCall(id="c1", name="echo", input={"text": "hi"})
    events = [e async for e in _executor(agent, state).stream(Acting(tool_calls=[tc]))]

    confirms = [e for e in events if e.type == EventType.REQUIRE_USER_CONFIRM]
    assert len(confirms) == 1
    assert confirms[0].tool_call_id == "c1"
    assert confirms[0].rule_id == "ask-echo"
    # 未执行：没有工具结果事件
    assert EventType.TOOL_RESULT_END not in [e.type for e in events]
    # tool_call 状态置 ASKING
    asking = MessageContext.tool_calls_in_state(agent.state.context, ToolCallState.ASKING)
    assert [b.id for b in asking] == ["c1"]


@pytest.mark.asyncio
async def test_resume_execute_after_approval():
    """挂起后确认放行 → resume_execute 执行工具，写 SUCCESS 结果。"""
    rules = [PermissionRule(id="ask-echo", tool_name="echo", behavior=PermissionBehavior.ASK)]
    agent = make_agent(rules=rules)
    state = make_state()
    executor = _executor(agent, state)

    action = Acting(tool_calls=[ToolCall(id="c1", name="echo", input={"text": "hi"})])
    [e async for e in executor.stream(action)]  # 挂起

    # 用户确认放行 → 置 ALLOWED（模拟 runner._resume 的动作）
    MessageContext.set_tool_call_state(agent.state.context, "c1", ToolCallState.ALLOWED)
    events = [e async for e in executor.resume_execute()]

    assert EventType.TOOL_RESULT_END in [e.type for e in events]
    assert "echo:hi" in str(agent.messages[-1].get("content", ""))


@pytest.mark.asyncio
async def test_resume_execute_after_rejection_writes_denied():
    """挂起后拒绝 → resume_execute 写 DENIED 结果，不执行工具。"""
    rules = [PermissionRule(id="ask-echo", tool_name="echo", behavior=PermissionBehavior.ASK)]
    agent = make_agent(rules=rules)
    state = make_state()
    executor = _executor(agent, state)

    action = Acting(tool_calls=[ToolCall(id="c1", name="echo", input={"text": "hi"})])
    [e async for e in executor.stream(action)]  # 挂起

    # 用户拒绝 → 置 FINISHED
    MessageContext.set_tool_call_state(agent.state.context, "c1", ToolCallState.FINISHED)
    events = [e async for e in executor.resume_execute()]

    end = [e for e in events if e.type == EventType.TOOL_RESULT_END]
    assert len(end) == 1
    assert end[0].state == ToolResultState.DENIED
    # 未执行：结果里没有 echo 输出
    assert "echo:hi" not in str(agent.messages[-1].get("content", ""))


@pytest.mark.asyncio
async def test_deny_pauses_with_ask_batch():
    """DENY 与 ASK 同批 → 整批挂起；恢复时 DENY 写 DENIED、ASK 确认后执行。"""
    rules = [
        PermissionRule(id="ask-echo", tool_name="echo", behavior=PermissionBehavior.ASK),
        PermissionRule(id="deny-echo2", tool_name="echo2", behavior=PermissionBehavior.DENY),
    ]

    @tool(description="Echo2")
    def echo2(text: str) -> str:
        return f"echo2:{text}"

    agent = make_agent(rules=rules)
    agent.tool_registry.register(echo2)
    state = make_state()
    executor = _executor(agent, state)

    action = Acting(tool_calls=[
        ToolCall(id="c1", name="echo", input={"text": "a"}),
        ToolCall(id="c2", name="echo2", input={"text": "b"}),
    ])
    events = [e async for e in executor.stream(action)]

    # 只有 ASK 的 c1 产出确认事件；DENY 的 c2 不产
    confirms = [e for e in events if e.type == EventType.REQUIRE_USER_CONFIRM]
    assert [e.tool_call_id for e in confirms] == ["c1"]

    # 确认放行 c1；c2 保持 PENDING（DENY）
    MessageContext.set_tool_call_state(agent.state.context, "c1", ToolCallState.ALLOWED)
    resume_events = [e async for e in executor.resume_execute()]

    ends = {e.tool_call_id: e for e in resume_events if e.type == EventType.TOOL_RESULT_END}
    assert ends["c1"].state == ToolResultState.SUCCESS
    assert ends["c2"].state == ToolResultState.DENIED


@pytest.mark.asyncio
async def test_new_message_rejected_while_awaiting_confirmation():
    """挂起期间收到普通消息（非确认）→ run() 拒绝。"""
    rules = [PermissionRule(id="ask-echo", tool_name="echo", behavior=PermissionBehavior.ASK)]
    agent = make_agent(rules=rules)

    # 手动往 context 塞一个 ASKING 的 tool_call（模拟已挂起）
    MessageContext.add_raw(
        agent.state.context,
        agent.runner.tool_handler.build_assistant_message(
            tool_calls=[ToolCall(id="c1", name="echo", input={"text": "hi"})]
        ),
    )
    MessageContext.set_tool_call_state(agent.state.context, "c1", ToolCallState.ASKING)

    with pytest.raises(RuntimeError, match="awaiting permission confirmation"):
        async for _ in agent.run("another message"):
            pass


@pytest.mark.asyncio
async def test_confirm_event_rejected_when_not_awaiting():
    """未挂起时收到确认事件 → run() 拒绝。"""
    agent = make_agent(engine=False)

    with pytest.raises(RuntimeError, match="not awaiting confirmation"):
        async for _ in agent.run(
            UserConfirmResultEvent(reply_id="r1", tool_call_id="c1", approved=True)
        ):
            pass


@pytest.mark.asyncio
async def test_no_engine_executes_without_permission_check():
    """无权限引擎 → 工具直接执行，行为与接线前一致。"""
    agent = make_agent(engine=False)
    state = make_state()

    tc = ToolCall(id="c1", name="echo", input={"text": "x"})
    events = [e async for e in _executor(agent, state).stream(Acting(tool_calls=[tc]))]

    assert EventType.TOOL_RESULT_END in [e.type for e in events]
    assert EventType.REQUIRE_USER_CONFIRM not in [e.type for e in events]


@pytest.mark.asyncio
async def test_append_event_marks_tool_call_asking():
    """RequireUserConfirmEvent 经 append_event 把对应 ToolCallBlock 置 ASKING。

    这是持久化/前端渲染的关键：ASKING 状态靠该事件写进 Msg 快照。
    """
    from ftre_agent_core.message import Msg, MsgName, ToolCallBlock

    msg = Msg(
        name=MsgName.DEFAULT,
        role="assistant",
        id="r1",
        content=[ToolCallBlock(id="c1", name="bash", arguments={"cmd": "ls"})],
    )
    assert msg.content[0].state == ToolCallState.PENDING

    msg.append_event(
        RequireUserConfirmEvent(
            reply_id="r1",
            tool_call_id="c1",
            tool_call_name="bash",
            arguments={"cmd": "ls"},
            reason="需要确认",
        )
    )
    assert msg.content[0].state == ToolCallState.ASKING


@pytest.mark.asyncio
async def test_resume_execute_rebuilds_from_context_only():
    """resume_execute 完全从 context 重建待收尾清单，不依赖任何传入动作或实例内存。

    模拟"持久化往返后恢复"：手动构造一条 ASKING 的 assistant 消息（如同从
    state.json 加载而来），把它置 ALLOWED，然后仅调 resume_execute() 就能执行。
    """
    agent = make_agent(rules=[
        PermissionRule(id="ask-echo", tool_name="echo", behavior=PermissionBehavior.ASK)
    ])
    state = make_state()

    # 直接往 context 塞一条带 tool_call 的 assistant 消息（模拟持久化加载）
    MessageContext.add_raw(
        agent.state.context,
        agent.runner.tool_handler.build_assistant_message(
            tool_calls=[ToolCall(id="c1", name="echo", input={"text": "hi"})]
        ),
    )
    # 模拟：挂起时置 ASKING，用户确认后置 ALLOWED
    MessageContext.set_tool_call_state(agent.state.context, "c1", ToolCallState.ALLOWED)

    # 仅凭 context 恢复，不传任何 action
    events = [e async for e in _executor(agent, state).resume_execute()]

    assert EventType.TOOL_RESULT_END in [e.type for e in events]
    assert "echo:hi" in str(agent.messages[-1].get("content", ""))
