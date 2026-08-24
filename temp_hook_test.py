"""临时脚本：真实 API 验证 Core Hook 管线（Tool / LLM / stop-decision）。"""
from __future__ import annotations

import asyncio
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ftre_agent_core.agent import ReActAgent
from ftre_agent_core.hooks import (
    AGENT_STOP_DECISION_SPEC,
    LLM_STREAM_SPEC,
    TOOL_AFTER_SPEC,
    TOOL_BEFORE_SPEC,
    ContinueTurn,
    HookMode,
    StopTurn,
)
from ftre_agent_core.tool import ToolRegistry, tool


class DemoHookDispatcher:
    """极简 dispatcher：WATERFALL 走 next() 链，EMIT 直接广播。"""

    def __init__(self) -> None:
        self.listeners: dict[Any, list] = defaultdict(list)

    def on(self, spec, listener) -> None:
        self.listeners[spec].append(listener)

    async def dispatch(self, spec, payload, *, context=None) -> Any:
        del context
        spec.validate_payload(payload)
        listeners = tuple(self.listeners.get(spec, ()))

        if spec.mode is HookMode.WATERFALL:
            async def invoke(index):
                if index == len(listeners):
                    return await spec.default(payload) if spec.default else None
                called = False

                async def next_():
                    nonlocal called
                    if called:
                        raise RuntimeError(f"{spec.name}: next() called twice")
                    called = True
                    return await invoke(index + 1)

                return await listeners[index](payload, next_)

            result = await invoke(0)
        elif spec.mode is HookMode.EMIT:
            for listener in listeners:
                await listener(payload)
            result = await spec.default(payload) if spec.default else None
        else:
            raise NotImplementedError(str(spec.mode))

        spec.validate_result(result)
        return result


def add_numbers(a: int, b: int) -> str:
    return str(a + b)


def build_dispatcher() -> DemoHookDispatcher:
    dispatcher = DemoHookDispatcher()

    async def llm_stream(payload, next_):
        print(
            f"[hook] {LLM_STREAM_SPEC.name}: model={payload.model} "
            f"messages={len(payload.messages)} tools={len(payload.tools)}"
        )
        stream = await next_()

        async def observed_stream():
            chunk_count = 0
            async for chunk in stream:
                chunk_count += 1
                yield chunk
            print(f"[hook] {LLM_STREAM_SPEC.name}: chunks={chunk_count}")

        return observed_stream()

    async def stop_decision(payload, next_):
        decision = await next_()
        if isinstance(decision, StopTurn):
            print(
                f"[hook] {AGENT_STOP_DECISION_SPEC.name}: stop 被拦截 → 继续 "
                f"cont={payload.continuation_count}/{payload.max_continuations}"
            )
            return ContinueTurn(
                prompt="继续工作，检查你刚才的回答是否完整、是否需要补充。",
                reason="block-stop-for-demo",
            )
        print(f"[hook] {AGENT_STOP_DECISION_SPEC.name}: continue={decision.reason!r}")
        return decision

    async def tool_before(payload, next_):
        print(
            f"[hook] {TOOL_BEFORE_SPEC.name}: "
            f"{payload.call.name} args={dict(payload.arguments)}"
        )
        return await next_()

    async def tool_after(payload, next_):
        result = await next_()
        print(f"[hook] {TOOL_AFTER_SPEC.name}: status={result.status}")
        return result

    dispatcher.on(LLM_STREAM_SPEC, llm_stream)
    dispatcher.on(AGENT_STOP_DECISION_SPEC, stop_decision)
    dispatcher.on(TOOL_BEFORE_SPEC, tool_before)
    dispatcher.on(TOOL_AFTER_SPEC, tool_after)
    return dispatcher


async def main() -> None:
    message = os.getenv(
        "FTRE_MESSAGE", "请调用 add_numbers 计算 7 + 5，然后用一句中文告诉我结果。"
    )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 OPENAI_API_KEY；请通过环境变量提供真实 API Key。")
    registry = ToolRegistry()
    registry.register(
        tool(name="add_numbers", description="计算两个整数的和。需要调用这个工具来得到准确结果。")(
            add_numbers
        )
    )
    agent = ReActAgent(
        model=os.getenv("FTRE_MODEL", "muse-spark-1.2-contributor"),
        api_key=api_key,
        api_base=os.getenv("OPENAI_BASE_URL", "https://opencode.ai/zen/go/v1"),
        api_type=os.getenv("FTRE_API_TYPE", "responses"),
        system_prompt="你是一个简洁的中文助手，需要精确计算时必须使用可用工具。",
        tool_registry=registry,
        hooks=build_dispatcher(),
        hook_context="temp-hook-demo",
        max_iterations=4,
        max_tokens=512,
        reasoning_effort=os.getenv("FTRE_REASONING_EFFORT", "medium"),
        max_retries=1,
        retry_delay=1.0,
    )

    print(f"message={message}")
    print("--- agent stream ---")
    async for event in agent.run(
        message,
        runtime_context={
            "session_id": "temp-hook-session",
            "agent_id": "temp-hook-agent",
            "request_id": "temp-hook-request",
            "max_continuations": 2,
        },
    ):
        event_type = str(event.type)
        if event_type == "TEXT_BLOCK_DELTA":
            print(getattr(event, "delta", ""), end="", flush=True)
        elif event_type in {"REPLY_START", "REPLY_END", "RETRY", "REQUIRE_USER_CONFIRM"}:
            print(f"\n[event] {event_type}: {event.model_dump(mode='json')}")
    print("\n--- done ---")


if __name__ == "__main__":
    asyncio.run(main())
