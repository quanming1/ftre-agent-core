"""Helpers for formatting reasoning in OpenAI-compatible messages."""

from __future__ import annotations

from typing import Any


def content_with_reasoning(content: Any, reasoning: str) -> list[dict[str, str]]:
    parts: list[dict[str, str]] = [{"type": "text", "text": reasoning}]
    parts.extend(content_parts(content))
    return parts


def content_parts(content: Any) -> list[dict[str, str]]:
    if isinstance(content, list):
        return [
            item if isinstance(item, dict) else {"type": "text", "text": str(item)}
            for item in content
        ]
    if content in (None, ""):
        return []
    return [{"type": "text", "text": str(content)}]


def merge_reasoning_into_content(msg: dict[str, Any], reasoning: str | None) -> None:
    if not reasoning:
        return
    msg["content"] = content_with_reasoning(msg.get("content"), reasoning)
    msg["reasoning_content"] = ""


def preserve_tool_call_reasoning(msg: dict[str, Any], reasoning: str | None) -> None:
    content = msg.get("content")
    if content in (None, ""):
        msg["content"] = ""
    else:
        msg["content"] = content_parts(content)
    if reasoning:
        msg["reasoning_content"] = reasoning
