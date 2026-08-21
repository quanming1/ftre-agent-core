"""Direct Core HookDispatcher integration for Tool and LLM boundaries."""
from __future__ import annotations

import asyncio

import pytest

from ftre_agent_core.agent import ReActAgent
from ftre_agent_core.agent.runner._execute_reasoning import ReasoningExecutor
from ftre_agent_core.agent.runner._state import Reasoning, RunState
from ftre_agent_core.agent.runner.tool_handler import ToolHandler
from ftre_agent_core.hooks import (
    LLM_STREAM_SPEC,
    TOOLS_EXECUTE_SPEC,
    TOOLS_POST_EXECUTE_SPEC,
    TOOLS_PRE_EXECUTE_SPEC,
    TOOLS_RESULT_SPEC,
    ToolArguments,
    ToolExecutionResult,
)
from ftre_agent_core.llm import (
    BlockEnd,
    BlockStart,
    FinishChunk,
    FinishReason,
    TextDeltaChunk,
    UsageChunk,
)
from ftre_agent_core.tool import Tool, ToolRegistry


class PipelineDispatcher:
    def __init__(self) -> None:
        self.names: list[str] = []

    async def dispatch(self, spec, payload, *, context=None):
        self.names.append(spec.name)
        if spec is TOOLS_PRE_EXECUTE_SPEC:
            return ToolArguments({"value": "changed"})
        if spec is TOOLS_EXECUTE_SPEC:
            raw = await payload.invoke()
            return ToolExecutionResult(output=raw.output.upper(), value=raw.value)
        if spec is TOOLS_POST_EXECUTE_SPEC:
            return ToolExecutionResult(output=payload.result.output + "!")
        if spec is TOOLS_RESULT_SPEC:
            return None
        if spec is LLM_STREAM_SPEC:
            stream = payload.invoke()

            async def wrapped():
                async for chunk in stream:
                    yield chunk

            return wrapped()
        return spec.default(payload) if spec.default else None


def make_state() -> RunState:
    state = RunState()
    state.runtime_context = {
        "session_id": "session-1",
        "cancellation": asyncio.Event(),
        "max_iterations": 3,
    }
    state.start()
    state.reply_id = "reply-1"
    return state


@pytest.mark.asyncio
async def test_tool_handler_dispatches_all_four_core_hooks():
    dispatcher = PipelineDispatcher()
    registry = ToolRegistry()
    registry.register(Tool(name="echo", func=lambda value: value))
    result = await ToolHandler(registry, dispatcher).run_one(
        "call-1", "echo", {"value": "original"}, make_state()
    )
    assert result.result == "CHANGED!"
    assert dispatcher.names == [
        "tools/pre-execute",
        "tools/execute",
        "tools/post-execute",
        "tools/result",
    ]


@pytest.mark.asyncio
async def test_reasoning_executor_dispatches_llm_stream_without_adapter():
    dispatcher = PipelineDispatcher()
    agent = ReActAgent(model="fake", api_key="fake", hooks=dispatcher)
    state = make_state()

    async def provider(messages, tools=None):
        yield BlockStart(block_type="text")
        yield TextDeltaChunk(text="hello")
        yield BlockEnd(block={"type": "text", "text": "hello"})
        yield UsageChunk(usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
        yield FinishChunk(
            reason=FinishReason(kind="stop", raw="stop")
        )

    agent.runner.llm.stream = provider
    executor = ReasoningExecutor(agent, state, agent.runner.llm, dispatcher)
    _events = [event async for event in executor.stream(Reasoning())]
    assert executor.result is not None
    assert executor.result.text == "hello"
    assert dispatcher.names == ["llm/stream"]
