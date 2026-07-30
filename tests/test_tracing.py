import json

import pytest

from ftre_agent_core.agent import ReActAgent
from ftre_agent_core.event import ReplyFinishedReason
from ftre_agent_core.llm import StepFinish, TextDelta, ToolCall
from ftre_agent_core.tool import tool
from ftre_agent_core.tracing import (
    InMemoryTraceExporter,
    JsonlTraceExporter,
    RunStatus,
    RunType,
    Tracer,
)


@pytest.mark.asyncio
async def test_agent_trace_records_llm_and_tool_run_tree():
    @tool(description="Echo text")
    def echo(text: str) -> str:
        return f"echo:{text}"

    exporter = InMemoryTraceExporter()
    agent = ReActAgent(
        model="requested-model",
        api_key="fake",
        max_iterations=3,
        tracer=Tracer([exporter]),
    )
    agent._registry.register(echo)
    calls = 0

    async def fake_stream(messages, tools=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield ToolCall(id="call_echo", name="echo", input={"text": "x"})
            yield StepFinish(
                finish_reason="tool_calls",
                usage={"total_tokens": 10},
                response_metadata={"model": "routed-model"},
            )
        else:
            yield TextDelta(text="done")
            yield StepFinish(
                finish_reason="stop",
                usage={"total_tokens": 4},
                response_metadata={"model": "routed-model"},
            )

    agent.runner.llm.stream = fake_stream
    events = [event async for event in agent.run("start")]

    assert agent.run_state.done_reason == ReplyFinishedReason.COMPLETED
    root = next(run for run in exporter.runs.values() if run.run_type == RunType.AGENT)
    runs = exporter.get_trace(root.trace_id)
    llm_runs = [run for run in runs if run.run_type == RunType.LLM]
    tool_runs = [run for run in runs if run.run_type == RunType.TOOL]

    assert len(runs) == 4
    assert len(llm_runs) == 2
    assert len(tool_runs) == 1
    assert all(run.parent_run_id == root.id for run in llm_runs + tool_runs)
    assert root.outputs == {"success": True, "done_reason": "completed", "iterations": 2}
    assert llm_runs[0].outputs["finish_reason"] == "tool_calls"
    assert llm_runs[0].outputs["response_metadata"]["model"] == "routed-model"
    assert llm_runs[1].outputs["finish_reason"] == "stop"
    assert llm_runs[1].outputs["has_tool_calls"] is False
    assert tool_runs[0].outputs["result"] == "echo:x"


@pytest.mark.asyncio
async def test_process_text_with_stop_is_observable_as_legal_completion():
    exporter = InMemoryTraceExporter()
    agent = ReActAgent(
        model="fake",
        api_key="fake",
        tracer=Tracer([exporter]),
    )

    async def fake_stream(messages, tools=None):
        yield TextDelta(text="我现在开始执行。")
        yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream
    events = [event async for event in agent.run("start")]

    assert agent.run_state.done_reason == ReplyFinishedReason.COMPLETED
    llm_run = next(run for run in exporter.runs.values() if run.run_type == RunType.LLM)
    assert llm_run.status == RunStatus.COMPLETED
    assert llm_run.outputs["text"] == "我现在开始执行。"
    assert llm_run.outputs["finish_reason"] == "stop"
    assert llm_run.outputs["has_tool_calls"] is False


def test_jsonl_exporter_writes_start_and_end_records(tmp_path):
    path = tmp_path / "traces.jsonl"
    tracer = Tracer([JsonlTraceExporter(path)])
    span = tracer.start_run("test", RunType.AGENT, inputs={"value": 1})
    span.add_event("checkpoint", {"pending": 2})
    span.end(outputs={"ok": True})

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["phase"] for record in records] == ["start", "end"]
    assert records[1]["run"]["events"][0]["name"] == "checkpoint"
    assert records[1]["run"]["outputs"] == {"ok": True}


@pytest.mark.asyncio
async def test_exporter_failure_does_not_break_agent_run():
    class BrokenExporter:
        def on_run_start(self, run):
            raise RuntimeError("offline")

        def on_run_end(self, run):
            raise RuntimeError("offline")

    agent = ReActAgent(
        model="fake",
        api_key="fake",
        tracer=Tracer([BrokenExporter()]),
    )

    async def fake_stream(messages, tools=None):
        yield TextDelta(text="done")
        yield StepFinish(finish_reason="stop")

    agent.runner.llm.stream = fake_stream
    events = [event async for event in agent.run("start")]
    assert agent.run_state.done_reason == ReplyFinishedReason.COMPLETED
