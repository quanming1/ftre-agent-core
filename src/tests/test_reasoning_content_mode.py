from __future__ import annotations

from ftre_agent_core.agent.runner.tool_handler import ToolHandler
from ftre_agent_core.llm import ToolCall
from ftre_agent_core.message_context import MessageContext
from ftre_agent_core.state import AgentState
from ftre_agent_core.tool import ToolRegistry


def test_message_context_keeps_reasoning_separate_from_content():
    state = AgentState()

    MessageContext.add_assistant(state.context, "answer", reasoning="thinking")

    assert MessageContext.messages(state.context) == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "answer"},
            ],
            "reasoning_content": "thinking",
        }
    ]


def test_tool_call_message_preserves_reasoning_content():
    handler = ToolHandler(ToolRegistry())
    msg = handler.build_assistant_message(
        [ToolCall(id="call_1", name="bash", input={"command": "pwd"})],
        reasoning="thinking",
    )

    assert msg["content"] == ""
    assert msg["reasoning_content"] == "thinking"
    assert msg["tool_calls"][0]["id"] == "call_1"


def test_tool_call_message_preserves_visible_content_as_parts():
    handler = ToolHandler(ToolRegistry())
    msg = handler.build_assistant_message(
        [ToolCall(id="call_1", name="bash", input={"command": "pwd"})],
        content="visible answer",
        reasoning="thinking",
    )

    assert msg["content"] == [{"type": "text", "text": "visible answer"}]
    assert msg["reasoning_content"] == "thinking"
    assert msg["tool_calls"][0]["id"] == "call_1"


def test_tool_call_message_with_visible_content_without_reasoning():
    handler = ToolHandler(ToolRegistry())
    msg = handler.build_assistant_message(
        [ToolCall(id="call_1", name="bash", input={"command": "pwd"})],
        content="visible answer",
    )

    assert msg["content"] == [{"type": "text", "text": "visible answer"}]
    assert msg["reasoning_content"] == ""
    assert msg["tool_calls"][0]["id"] == "call_1"
