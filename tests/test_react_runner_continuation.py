import logging

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
    types = [event.type for event in events]

    assert calls == 2
    assert types.count(EventType.DONE) == 1
    assert agent.memory.messages[1] == {"role": "assistant", "content": "done"}


@pytest.mark.asyncio
async def test_empty_response_retries_then_requests_finalization_without_tools():
    @tool(description="Echo text")
    def echo(text: str) -> str:
        return f"echo:{text}"

    agent = make_agent(tools=[echo], max_iterations=4)
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
            assert "直接给出回复用户的最终内容" in messages[-1]["content"]
            yield TextDelta(text="final")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    events = [event async for event in agent.run("start")]
    types = [event.type for event in events]

    assert calls == 3
    assert types.count(EventType.USER_MESSAGE) == 1
    assert types.count(EventType.DONE) == 1
    assert agent.memory.messages[1]["role"] == "user"
    assert "直接给出回复用户的最终内容" in agent.memory.messages[1]["content"]
    assert agent.memory.messages[2] == {"role": "assistant", "content": "final"}


@pytest.mark.asyncio
async def test_unknown_finish_with_text_logs_and_continues(caplog):
    agent = make_agent(max_iterations=3)
    calls = 0

    async def fake_stream(messages, tools=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield TextDelta(text="partial but eof")
            yield StepFinish(finish_reason="unknown")
        else:
            assert messages[-1] == {"role": "assistant", "content": "partial but eof"}
            yield TextDelta(text="final")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    with caplog.at_level(logging.WARNING, logger="ftre_agent_core.agent.runner.react_runner"):
        events = [event async for event in agent.run("start")]

    types = [event.type for event in events]

    assert calls == 2
    assert types.count(EventType.DONE) == 1
    assert agent.memory.messages[1] == {"role": "assistant", "content": "partial but eof"}
    assert agent.memory.messages[2] == {"role": "assistant", "content": "final"}
    assert "provider 未返回明确 finish_reason" in caplog.text


@pytest.mark.asyncio
async def test_unknown_finish_with_empty_response_continues_without_finalization(caplog):
    @tool(description="Echo text")
    def echo(text: str) -> str:
        return f"echo:{text}"

    agent = make_agent(tools=[echo], max_iterations=3)
    calls = 0

    async def fake_stream(messages, tools=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield StepFinish(finish_reason="unknown")
        else:
            assert tools is not None
            yield TextDelta(text="final")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    with caplog.at_level(logging.WARNING, logger="ftre_agent_core.agent.runner.react_runner"):
        events = [event async for event in agent.run("start")]

    types = [event.type for event in events]

    assert calls == 2
    assert types.count(EventType.USER_MESSAGE) == 0
    assert types.count(EventType.DONE) == 1
    assert agent.memory.messages[1] == {"role": "assistant", "content": "final"}
    assert "provider 未返回明确 finish_reason" in caplog.text


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


@pytest.mark.asyncio
async def test_length_finish_adds_hidden_user_continuation():
    agent = make_agent()
    calls = 0

    async def fake_stream(messages, tools=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield TextDelta(text="partial")
            yield StepFinish(finish_reason="length")
        else:
            assert messages[-1]["role"] == "user"
            assert "从刚才中断的位置继续" in messages[-1]["content"]
            yield TextDelta(text=" done")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    events = [event async for event in agent.run("start")]
    types = [event.type for event in events]

    assert calls == 2
    assert types.count(EventType.DONE) == 1
    assert agent.memory.messages[1]["content"] == "partial"
    assert agent.memory.messages[2]["role"] == "user"
    assert "从刚才中断的位置继续" in agent.memory.messages[2]["content"]


@pytest.mark.asyncio
async def test_pending_user_message_continues_after_final_looking_response():
    agent = make_agent()
    calls = 0
    pending = ["Please keep going."]

    async def fake_stream(messages, tools=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield TextDelta(text="intermediate")
            yield StepFinish(finish_reason="stop")
        else:
            assert messages[-1] == {"role": "user", "content": "Please keep going."}
            yield TextDelta(text="final")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    events = [
        event
        async for event in agent.run(
            "start",
            runtime_context={"pending_user_messages": pending},
        )
    ]
    types = [event.type for event in events]

    assert calls == 2
    assert pending == []
    assert types.count(EventType.USER_MESSAGE) == 1
    assert types.count(EventType.DONE) == 1
    assert agent.memory.messages[2] == {"role": "user", "content": "Please keep going."}


@pytest.mark.asyncio
async def test_active_continuation_continues_after_final_looking_response():
    agent = make_agent()
    calls = 0

    def active_once():
        return calls == 1

    async def fake_stream(messages, tools=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield TextDelta(text="still working")
            yield StepFinish(finish_reason="stop")
        else:
            assert messages[-1]["role"] == "user"
            assert messages[-1]["content"] == "custom continue"
            yield TextDelta(text="final")
            yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream

    events = [
        event
        async for event in agent.run(
            "start",
            runtime_context={
                "continuation_active": active_once,
                "continuation_message": "custom continue",
            },
        )
    ]
    types = [event.type for event in events]

    assert calls == 2
    assert types.count(EventType.USER_MESSAGE) == 1
    assert types.count(EventType.DONE) == 1
    assert agent.memory.messages[2] == {"role": "user", "content": "custom continue"}
