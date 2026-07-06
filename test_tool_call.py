"""Test tool call flow with Minimax to debug 'tool call result does not follow tool call' error."""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ftre_agent_core.llm.completion import LLMHandler
from ftre_agent_core.memory import MemoryManager
from ftre_agent_core.reasoning import format_assistant_message

# Minimax config
API_KEY = "sk-REDACTED"  # truncated, read full from config
API_BASE = "https://api.minimaxi.com/v1"
MODEL = "MiniMax-M3"

# Read full API key
import json as _json
_cfg = _json.load(open(r"C:\Users\蒋全明\.ftre\config.json", encoding="utf-8"))
API_KEY = _cfg["providers"]["MiniMax拼车"]["api_key"]


async def main():
    print(f"Model: {MODEL}")
    print(f"API Base: {API_BASE}")
    print(f"API Key: {API_KEY[:12]}...")
    print()

    llm = LLMHandler(model=MODEL, api_key=API_KEY, api_base=API_BASE)
    memory = MemoryManager()
    memory._system_prompt = "You are a helpful assistant. Use tools when needed."

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            },
        }
    ]

    # --- Round 1: User asks about weather ---
    memory.add_user("What's the weather in Beijing, Shanghai, and Guangzhou? Check them one by one.")
    messages = memory.get_messages()
    print("=== Round 1: Sending to LLM ===")
    for m in messages:
        print(f"  {m['role']}: {str(m.get('content', ''))[:60]}")

    text_parts = []
    reasoning_parts = []
    tool_calls = []
    finish_reason = "unknown"

    async for event in llm.stream(messages, tools):
        etype = type(event).__name__
        if etype == "TextDelta":
            text_parts.append(event.text)
        elif etype == "ReasoningDelta":
            reasoning_parts.append(event.text)
        elif etype == "ToolInputDelta":
            pass  # streaming tool args
        elif etype == "ToolCall":
            tool_calls.append(event)
            print(f"  Got ToolCall: id={event.id}, name={event.name}, input={event.input}")
        elif etype == "StepFinish":
            finish_reason = event.finish_reason
            print(f"  StepFinish: finish_reason={finish_reason}")

    full_text = "".join(text_parts)
    full_reasoning = "".join(reasoning_parts)
    print(f"  Text: {full_text[:80]}")
    print(f"  Reasoning: {full_reasoning[:80]}")
    print(f"  Tool calls: {len(tool_calls)}")

    # --- Write assistant message to memory (same as react_runner) ---
    if tool_calls:
        from ftre_agent_core.agent.runner.tool_handler import ToolHandler
        assistant_msg = ToolHandler.build_assistant_message(
            tool_calls=tool_calls,
            content=full_text or None,
            reasoning=full_reasoning or None,
        )
    else:
        assistant_msg = format_assistant_message(
            content=full_text,
            reasoning=full_reasoning,
        )

    memory.add_raw(assistant_msg)
    print()
    print("=== Memory after assistant (Round 1) ===")
    for m in memory._messages:
        print(f"  {m['role']}: content={str(m.get('content', ''))[:40]} "
              f"reasoning={str(m.get('reasoning_content', ''))[:20]} "
              f"tool_calls={'yes' if m.get('tool_calls') else 'no'}")

    # --- Simulate tool result ---
    for tc in tool_calls:
        memory.add_tool_result(tc.id, json.dumps({"weather": "sunny, 25C"}))
    print()
    print("=== Memory after tool_result ===")
    for m in memory._messages:
        print(f"  {m['role']}: content={str(m.get('content', ''))[:40]} "
              f"tool_call_id={m.get('tool_call_id', '-')} "
              f"tool_calls={'yes' if m.get('tool_calls') else 'no'}")

    # --- Run multiple rounds of tool calls ---
    for round_num in range(2, 5):
        if not tool_calls:
            print(f"\n=== Round {round_num}: No tool calls, skipping ===")
            break

        print(f"\n=== Round {round_num}: Sending to LLM ===")
        messages2 = memory.get_messages()
        print(f"  Total messages: {len(messages2)}")
        # Print full JSON of last 4 messages to see exact format
        print("  Last 4 messages JSON:")
        for i, m in enumerate(messages2[-4:]):
            idx = len(messages2) - 4 + i
            print(f"  [{idx}] {json.dumps(m, ensure_ascii=False, default=str)[:200]}")

        text_parts = []
        reasoning_parts = []
        tool_calls = []
        finish_reason = "unknown"

        try:
            async for event in llm.stream(messages2, tools):
                etype = type(event).__name__
                if etype == "TextDelta":
                    text_parts.append(event.text)
                elif etype == "ReasoningDelta":
                    reasoning_parts.append(event.text)
                elif etype == "ToolCall":
                    tool_calls.append(event)
                    print(f"  ToolCall: {event.id} {event.name}")
                elif etype == "StepFinish":
                    finish_reason = event.finish_reason
                    print(f"  StepFinish: {finish_reason}")

            full_text = "".join(text_parts)
            full_reasoning = "".join(reasoning_parts)
            print(f"  Text: {full_text[:80]}")
            print(f"  Tool calls: {len(tool_calls)}")

            # Write to memory
            if tool_calls:
                assistant_msg = ToolHandler.build_assistant_message(
                    tool_calls=tool_calls,
                    content=full_text or None,
                    reasoning=full_reasoning or None,
                )
            else:
                assistant_msg = format_assistant_message(
                    content=full_text,
                    reasoning=full_reasoning,
                )
            memory.add_raw(assistant_msg)

            # Simulate tool results
            for tc in tool_calls:
                memory.add_tool_result(tc.id, json.dumps({"result": "ok"}))

            print(f"=== Round {round_num} SUCCESS ===")

        except Exception as e:
            print(f"\n=== Round {round_num} FAILED: {e} ===")
            # Print full message sequence on failure
            print("\n=== Full messages on failure ===")
            print(json.dumps(memory.get_messages(), indent=2, ensure_ascii=False, default=str))
            break


if __name__ == "__main__":
    asyncio.run(main())

