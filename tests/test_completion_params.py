"""适配器请求参数组装测试（PRD-B2：迁移自 test_completion_params）。

LLMHandler 单类 → create_llm_handler 工厂 + 双适配器后的参数断言：
- completions：max_tokens / stream / tool_choice 透传；空 assistant 归一化
- responses：tool_choice=auto；instructions 提取
"""

from types import SimpleNamespace

import pytest

from ftre_agent_core.llm import FinishChunk, create_llm_handler


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
async def test_completions_adapter_passes_configured_max_tokens(monkeypatch):
    handler = create_llm_handler("completions", model="test-model", api_key="test-key", max_tokens=8192)
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return _FakeStream()

    monkeypatch.setattr(
        handler._client,
        "chat",
        SimpleNamespace(completions=SimpleNamespace(create=create)),
    )

    chunks = [c async for c in handler.stream([{"role": "user", "content": "hi"}])]

    assert captured["max_tokens"] == 8192
    assert captured["stream"] is True
    assert captured["tool_choice"] == "auto"
    assert isinstance(chunks[-1], FinishChunk)
    assert chunks[-1].reason.kind == "stop"


@pytest.mark.asyncio
async def test_responses_adapter_uses_auto_tool_choice(monkeypatch):
    handler = create_llm_handler("responses", model="test-model", api_key="test-key")
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return _FakeStream()

    monkeypatch.setattr(
        handler._client,
        "responses",
        SimpleNamespace(create=create),
    )

    [c async for c in handler.stream([{"role": "user", "content": "hi"}])]

    assert captured["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_completions_adapter_normalizes_empty_assistant_content(monkeypatch):
    """真实请求中历史 tool-call 和当前 ReAct 消息都可能没有可见正文。"""
    handler = create_llm_handler("completions", model="test-model", api_key="test-key")
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

    [c async for c in handler.stream(messages)]

    assistant_messages = [m for m in captured["messages"] if m["role"] == "assistant"]
    assert assistant_messages[0]["content"] == ""
    assert assistant_messages[1]["content"] == ""
    assert len(assistant_messages) == 2
    # 请求边界规范化不能污染 memory/history 原对象。
    assert "content" not in messages[0]
    assert messages[2]["content"] == ""
    assert "content" not in messages[4]


@pytest.mark.asyncio
async def test_completions_adapter_deepseek_thinking_extra_body(monkeypatch):
    """deepseek 模型 + reasoning_effort → extra_body.thinking.enabled 特判。"""
    handler = create_llm_handler(
        "completions", model="deepseek-v4-flash", api_key="k", reasoning_effort="high",
    )
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return _FakeStream()

    monkeypatch.setattr(
        handler._client,
        "chat",
        SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    [c async for c in handler.stream([{"role": "user", "content": "hi"}])]

    assert captured["reasoning_effort"] == "high"
    assert captured["extra_body"] == {"thinking": {"type": "enabled"}}


@pytest.mark.asyncio
async def test_responses_adapter_reasoning_effort_param(monkeypatch):
    """responses 适配器：reasoning_effort → reasoning.effort（Muse 档位生效路径）。"""
    handler = create_llm_handler(
        "responses", model="muse-spark-1.2", api_key="k", reasoning_effort="xhigh",
    )
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return _FakeStream()

    monkeypatch.setattr(
        handler._client,
        "responses",
        SimpleNamespace(create=create),
    )
    [c async for c in handler.stream([{"role": "user", "content": "hi"}])]

    assert captured["reasoning"] == {"effort": "xhigh"}
