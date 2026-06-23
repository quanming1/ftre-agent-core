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
    assert isinstance(events[-1], StepFinish)
    assert events[-1].finish_reason == "stop"
