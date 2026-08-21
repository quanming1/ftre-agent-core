# tests/test_execute_acting.py
"""ActingExecutor 单元测试。"""
import pytest

from ftre_agent_core.agent.react import ReActAgent
from ftre_agent_core.agent.runner._execute_acting import ActingExecutor
from ftre_agent_core.agent.runner._state import Acting, RunState
from ftre_agent_core.event import EventType
from ftre_agent_core.llm import ToolCall
from ftre_agent_core.tool import ToolRegistry, tool


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

    tc = ToolCall(id="c1", name="echo", input={"text": "hello"})
    executor = ActingExecutor(agent, state, agent.runner.tool_handler, agent.permission_engine)
    events = [e async for e in executor.stream(Acting(tool_calls=[tc]))]

    types = [e.type for e in events]
    assert EventType.TOOL_RESULT_START in types
    assert EventType.TOOL_RESULT_END in types

    # 验证 memory 写入（assistant tool_calls + tool result）
    msgs = agent.messages
    last_msg = msgs[-1]
    # 最后一条应该是 tool result
    assert "echo:hello" in str(last_msg.get("content", ""))


@pytest.mark.asyncio
async def test_multiple_tool_execution():
    agent = make_agent()
    state = make_state()

    tc1 = ToolCall(id="c1", name="echo", input={"text": "a"})
    tc2 = ToolCall(id="c2", name="echo", input={"text": "b"})

    executor = ActingExecutor(agent, state, agent.runner.tool_handler, agent.permission_engine)
    events = [e async for e in executor.stream(Acting(tool_calls=[tc1, tc2]))]

    tool_end_events = [e for e in events if e.type == EventType.TOOL_RESULT_END]
    assert len(tool_end_events) == 2
