import pytest

from ftre_agent_core.agent import EventType, ReActAgent
from ftre_agent_core.llm import ReasoningDelta, StepFinish, TextDelta, ToolCall
from ftre_agent_core.tool import tool


def make_agent(tools=None, max_iterations=3):
    return ReActAgent(
        model="fake",
        api_key="fake",
        system_prompt="test",
        tools=tools or [],
        max_iterations=max_iterations,
    )


@pytest.mark.asyncio
async def test_reasoning_only_turn_is_persisted_and_continues():
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
    types = [event.type for event in events]

    assert calls == 2
    assert types.count(EventType.DONE) == 1
    assert agent.memory.messages[1] == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "need more work",
    }


@pytest.mark.asyncio
async def test_tool_call_turn_does_not_emit_done_before_followup_turn():
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
    types = [event.type for event in events]

    assert calls == 2
    assert EventType.TOOL_CALL in types
    assert EventType.TOOL_RESULT in types
    assert types.index(EventType.DONE) > types.index(EventType.TOOL_RESULT)
