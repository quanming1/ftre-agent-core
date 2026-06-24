"""Helpers for formatting assistant messages in OpenAI-compatible history."""

from __future__ import annotations

from typing import Any


def content_parts(content: Any) -> list[dict[str, str]]:
    if isinstance(content, list):
        return [
            item if isinstance(item, dict) else {"type": "text", "text": str(item)}
            for item in content
        ]
    if content in (None, ""):
        return []
    return [{"type": "text", "text": str(content)}]


def format_assistant_message(
    *,
    content: Any = None,
    reasoning: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """尽量把同一轮模型输出还原为一条 assistant 消息。

    推理内容只写入 reasoning_content，正文只写入 content，工具调用只写入
    tool_calls。这样 DeepSeek / BigModel 这类原生推理网关可以继续识别真实的
    reasoning_content，同时正文不会和推理内容混在一起。
    """
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": content_parts(content) or "",
        "reasoning_content": reasoning or "",
    }
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return msg
