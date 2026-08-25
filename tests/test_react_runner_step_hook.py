"""agent/before-reasoning 的真实 ReAct 边界回归测试。"""
from __future__ import annotations

import asyncio

import pytest
from fake_llm import StepFinish, TextDelta, ToolCall, seq

from ftre_agent_core.agent import ReActAgent
from ftre_agent_core.hooks import (
    AGENT_BEFORE_REASONING_SPEC,
    AGENT_STOP_DECISION_SPEC,
    BeforeReasoningResult,
    ContinueTurn,
    StopTurn,
)
from ftre_agent_core.tool import Tool


class SequenceLLM:
    """按调用次数返回 ToolCall → Text 或 Text → Text 的最小适配器。"""

    model = "fake"

    def __init__(self, sequences: list[list]) -> None:
        self.sequences = sequences
        self.calls: list[list[dict]] = []

    async def stream(self, messages, tools=None):
        self.calls.append(messages)
        sequence = self.sequences[min(len(self.calls) - 1, len(self.sequences) - 1)]
        for chunk in sequence:
            yield chunk


class BeforeReasoningDispatcher:
    def __init__(self, *, continue_once: bool = False) -> None:
        self.calls = []
        self.continue_once = continue_once
        self._continued = False

    async def dispatch(self, spec, payload, *, context=None):
        self.calls.append((spec, payload))
        if spec is AGENT_BEFORE_REASONING_SPEC:
            return BeforeReasoningResult(({
                "role": "user",
                "content": f"steer-{payload.iteration}",
            },))
        if spec is AGENT_STOP_DECISION_SPEC:
            if self.continue_once and not self._continued:
                self._continued = True
                return ContinueTurn("继续推理", source="test")
            return StopTurn()
        return await spec.default(payload)


def _text_sequence(text: str) -> list:
    return seq(TextDelta(text=text), StepFinish(finish_reason="stop"))


@pytest.mark.asyncio
async def test_hook_runs_before_first_reasoning_and_after_tool():
    dispatcher = BeforeReasoningDispatcher()
    agent = ReActAgent(
        model="fake", api_key="fake", hooks=dispatcher, max_iterations=3,
    )
    agent.tool_registry.register(
        Tool(name="echo", func=lambda value: value),
    )
    agent.runner._llm = SequenceLLM([
        seq(ToolCall(id="call-1", name="echo", input={"value": "ok"}),
            StepFinish(finish_reason="tool_calls")),
        _text_sequence("完成"),
    ])

    events = [
        event async for event in agent.run(
            "开始", runtime_context={"session_id": "session-1"},
        )
    ]

    before_calls = [
        payload for spec, payload in dispatcher.calls
        if spec is AGENT_BEFORE_REASONING_SPEC
    ]
    assert [payload.iteration for payload in before_calls] == [1, 2]
    llm = agent.runner._llm
    assert len([
        message for message in llm.calls[0]
        if "steer-1" in str(message.get("content"))
    ]) == 1
    assert len([
        message for message in llm.calls[1]
        if "steer-2" in str(message.get("content"))
    ]) == 1
    assert any(getattr(event, "finished_reason", None) == "completed" for event in events)


@pytest.mark.asyncio
async def test_hook_runs_again_after_turn_continuation():
    dispatcher = BeforeReasoningDispatcher(continue_once=True)
    agent = ReActAgent(
        model="fake", api_key="fake", hooks=dispatcher, max_iterations=3,
    )
    agent.runner._llm = SequenceLLM([_text_sequence("第一轮"), _text_sequence("第二轮")])

    [
        event async for event in agent.run(
            "开始", runtime_context={"session_id": "session-1"},
        )
    ]

    before_calls = [
        payload for spec, payload in dispatcher.calls
        if spec is AGENT_BEFORE_REASONING_SPEC
    ]
    assert [payload.iteration for payload in before_calls] == [1, 2]


@pytest.mark.asyncio
async def test_user_message_creates_new_assistant_message_id_at_reasoning_boundary():
    """正式 UserMessage 让同一 reply 形成 A→User→B，Tool 状态不被搬运。"""
    class BoundaryDispatcher:
        async def dispatch(self, spec, payload, *, context=None):
            del context
            if spec is AGENT_BEFORE_REASONING_SPEC and payload.iteration == 2:
                return BeforeReasoningResult(({
                    "id": "user-steer-1",
                    "role": "user",
                    "content": "请继续，但改用中文",
                    "metadata": {"request_id": "request-steer-1"},
                },))
            if spec is AGENT_STOP_DECISION_SPEC:
                return StopTurn()
            return await spec.default(payload)

    agent = ReActAgent(
        model="fake", api_key="fake", hooks=BoundaryDispatcher(), max_iterations=3,
    )
    agent.runner._llm = SequenceLLM([
        seq(ToolCall(id="call-1", name="echo", input={"value": "ok"}),
            StepFinish(finish_reason="tool_calls")),
        _text_sequence("完成"),
    ])
    agent.tool_registry.register(Tool(name="echo", func=lambda value: value))

    events = [event async for event in agent.run("开始")]
    assistant = [message for message in agent.state.context if message.role == "assistant"]
    users = [message for message in agent.state.context if message.role == "user"]
    assert len(assistant) == 2
    reply_start = next(event for event in events if event.type == "REPLY_START")
    model_starts = [
        event.message_id for event in events if event.type == "MODEL_CALL_START"
    ]
    assert model_starts == [assistant[0].id, assistant[1].id]
    assert assistant[0].id == reply_start.message_id
    assert assistant[0].id != assistant[1].id
    assert [message.id for message in users if message.id == "user-steer-1"] == ["user-steer-1"]
    assert [event.reply_id for event in events if getattr(event, "reply_id", None)]
    assert len({event.reply_id for event in events if getattr(event, "reply_id", None)}) == 1
    assert assistant[0].get_content_blocks("tool_result")


@pytest.mark.asyncio
async def test_cancelled_context_does_not_dispatch_or_inject():
    dispatcher = BeforeReasoningDispatcher()
    agent = ReActAgent(model="fake", api_key="fake", hooks=dispatcher)
    cancellation = asyncio.Event()
    cancellation.set()
    agent.runner.state.runtime_context = {
        "session_id": "session-1",
        "cancellation": cancellation,
    }
    agent.runner.state.start()

    with pytest.raises(asyncio.CancelledError):
        await agent.runner._dispatch_before_reasoning()
    assert dispatcher.calls == []
    assert agent.messages == []


def test_before_reasoning_result_freezes_messages():
    result = BeforeReasoningResult(({"role": "user", "content": "hello"},))
    assert result.messages[0]["content"] == "hello"
    with pytest.raises(TypeError):
        result.messages[0]["content"] = "changed"
