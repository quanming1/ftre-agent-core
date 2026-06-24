from __future__ import annotations

from ftre_agent_core.agent.runner.tool_handler import ToolHandler
from ftre_agent_core.llm import ToolCall
from ftre_agent_core.memory import MemoryManager
from ftre_agent_core.tool import ToolRegistry


def test_memory_merges_reasoning_into_content():
    memory = MemoryManager()

    memory.add_assistant("answer", reasoning="thinking")

    assert memory.messages == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "thinking"},
                {"type": "text", "text": "answer"},
            ],
            "reasoning_content": "",
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
