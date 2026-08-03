from types import SimpleNamespace

import pytest

from ftre_agent_core.llm import LLMHandler, StepFinish


class _FakeStream:
    def __init__(self):
        self._chunks = iter([
            {
                "choices": [{
                    "delta": {"content": "ok"},
                    "finish_reason": "stop",
                }],
            },
        ])

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    def close(self):
        return None


@pytest.mark.asyncio
async def test_llm_handler_passes_configured_max_tokens(monkeypatch):
    handler = LLMHandler("test-model", "test-key", max_tokens=8192)
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return _FakeStream()

    monkeypatch.setattr(
        handler._client,
        "chat",
        SimpleNamespace(completions=SimpleNamespace(create=create)),
    )

    events = [event async for event in handler.stream([{"role": "user", "content": "hi"}])]

    assert captured["max_tokens"] == 8192
    assert captured["stream"] is True
    assert captured["tool_choice"] == "auto"
    assert isinstance(events[-1], StepFinish)
    assert events[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_responses_api_uses_auto_tool_choice(monkeypatch):
    handler = LLMHandler("test-model", "test-key", api_type="responses")
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return _FakeStream()

    monkeypatch.setattr(
        handler._client,
        "responses",
        SimpleNamespace(create=create),
    )

    events = [event async for event in handler.stream([{"role": "user", "content": "hi"}])]

    assert captured["tool_choice"] == "auto"
    assert isinstance(events[-1], StepFinish)


@pytest.mark.asyncio
async def test_chat_completions_normalizes_empty_assistant_content(monkeypatch):
    """真实请求中历史 tool-call 和当前 ReAct 消息都可能没有可见正文。"""
    handler = LLMHandler("test-model", "test-key")
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return _FakeStream()

    monkeypatch.setattr(
        handler._client,
        "chat",
        SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    messages = [
        # 22.json[152]：历史 assistant tool-call，content 字段已被 converter 省略。
        {
            "role": "assistant",
            "tool_calls": [{
                "id": "call_history",
                "type": "function",
                "function": {"name": "bash", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "call_history", "content": "ok"},
        # 22.json[214]：当前 ReAct memory 写入 content="" + reasoning + tool_calls。
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "need tools",
            "tool_calls": [{
                "id": "call_live",
                "type": "function",
                "function": {"name": "read", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "call_live", "content": "ok"},
        # 22.json[158]：只有 reasoning_content，没有正文或工具调用；
        # OpenAI-compatible provider 不把它视为合法 assistant payload。
        {"role": "assistant", "reasoning_content": "internal reasoning"},
        {"role": "user", "content": "continue"},
    ]

    await _collect(handler, messages)

    assistant_messages = [m for m in captured["messages"] if m["role"] == "assistant"]
    assert assistant_messages[0]["content"] == ""
    assert assistant_messages[1]["content"] == ""
    assert len(assistant_messages) == 2
    # 请求边界规范化不能污染 memory/history 原对象。
    assert "content" not in messages[0]
    assert messages[2]["content"] == ""
    assert "content" not in messages[4]


async def _collect(handler, messages):
    return [event async for event in handler.stream(messages)]
