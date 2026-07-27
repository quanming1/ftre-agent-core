"""Regression test: assistant_message_complete should not accumulate across turns."""
import pytest

from ftre_agent_core.agent.react import ReActAgent
from ftre_agent_core.event import AssistantMessageCompleteEvent
from ftre_agent_core.llm.completion import (
    TextDelta,
    ReasoningDelta,
    ToolInputDelta,
    ToolCall,
    StepFinish,
)
from ftre_agent_core.memory import MemoryManager
from ftre_agent_core.agent.runner.tool_handler import ToolResult
from ftre_agent_core.tool import ToolRegistry


class FakeLLM:
    def __init__(self, turns):
        self._turns = turns
        self._turn_idx = 0

    async def stream(self, messages, tools=None):
        if self._turn_idx >= len(self._turns):
            return
        turn = self._turns[self._turn_idx]
        self._turn_idx += 1
        for ev in turn:
            yield ev

    def cancel(self):
        pass


class FakeToolHandler:
    @staticmethod
    def build_assistant_message(tool_calls, content=None, reasoning=None):
        return {
            "role": "assistant",
            "content": content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": "{}"}}
                for tc in tool_calls
            ],
        }

    def spawn(self, tool_call, state, parent_span=None):
        task = type("Task", (), {})()
        task.result = lambda: ToolResult(
            call_id=tool_call.id, name=tool_call.name, result="ok", event=None
        )
        return task

    async def drain(self, tasks):
        pass

    async def gather_results(self, tool_calls, tasks, state, cancel_token=None):
        return [tasks[tc.id].result() for tc in tool_calls], set()


class FakeMemory(MemoryManager):
    def __init__(self):
        self._messages = []
        self.system_prompt = ""

    def get_messages(self):
        return list(self._messages)

    def add_assistant(self, content, reasoning=None, tool_calls=None):
        self._messages.append({"role": "assistant", "content": content or ""})

    def add_tool_results(self, results):
        for r in results:
            self._messages.append({"role": "tool", "content": getattr(r, "output", "ok"), "tool_call_id": getattr(r, "tool_call_id", "tc1")})


@pytest.mark.asyncio
async def test_amc_does_not_accumulate_across_turns():
    turns = [
        # Turn 1: one tool call
        [
            ToolInputDelta(id="tc1", name="read", text='{"path": "a.py"}'),
            ToolCall(id="tc1", name="read", input={"path": "a.py"}),
            StepFinish(finish_reason="tool_calls", usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
        ],
        # Turn 2: reasoning + text + two tool calls
        [
            ReasoningDelta(text="Let me think"),
            TextDelta(text="I will read more"),
            ToolInputDelta(id="tc2", name="read", text='{"path": "b.py"}'),
            ToolInputDelta(id="tc3", name="read", text='{"path": "c.py"}'),
            ToolCall(id="tc2", name="read", input={"path": "b.py"}),
            ToolCall(id="tc3", name="read", input={"path": "c.py"}),
            StepFinish(finish_reason="tool_calls", usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}),
        ],
        # Turn 3: final text
        [
            TextDelta(text="Done"),
            StepFinish(finish_reason="stop", usage={"prompt_tokens": 30, "completion_tokens": 2, "total_tokens": 32}),
        ],
    ]

    agent = ReActAgent(
        model="fake",
        api_key="fake",
        memory=FakeMemory(),
        tool_registry=ToolRegistry(),
        max_iterations=5,
        max_retries=0,
    )
    # monkey-patch the runner's llm and tool_handler
    agent._runner.llm = FakeLLM(turns)
    agent._runner.tool_handler = FakeToolHandler()

    events = []
    async for ev in agent._runner.run("start"):
        events.append(ev)

    amcs = [
        (ev.content, ev.metadata.get("kind"))
        for ev in events
        if isinstance(ev, AssistantMessageCompleteEvent)
    ]

    print("AMC contents:")
    for i, (content, kind) in enumerate(amcs):
        print(f"  AMC {i}: kind={kind} blocks={[(b.get('type'), b.get('name') or b.get('text', '')[:20]) for b in content]}")

    assert len(amcs) == 3, f"Expected 3 AMC, got {len(amcs)}"

    # Turn 1 AMC: only tc1
    assert len(amcs[0][0]) == 1
    assert amcs[0][0][0]["type"] == "toolCall"
    assert amcs[0][0][0]["id"] == "tc1"

    # Turn 2 AMC: reasoning + text + tc2 + tc3 (NOT tc1)
    assert len(amcs[1][0]) == 4
    block_types = [b["type"] for b in amcs[1][0]]
    assert block_types == ["thinking", "text", "toolCall", "toolCall"]
    tc_ids = [b["id"] for b in amcs[1][0] if b["type"] == "toolCall"]
    assert tc_ids == ["tc2", "tc3"]

    # Turn 3 AMC: only text
    assert len(amcs[2][0]) == 1
    assert amcs[2][0][0]["type"] == "text"
