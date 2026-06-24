"""Helpers for formatting reasoning in OpenAI-compatible messages."""

from __future__ import annotations

from typing import Any


def content_with_reasoning(content: Any, reasoning: str) -> list[dict[str, str]]:
    parts: list[dict[str, str]] = [{"type": "text", "text": reasoning}]
    if isinstance(content, list):
        for item in content:
            parts.append(item if isinstance(item, dict) else {"type": "text", "text": str(item)})
    elif content not in (None, ""):
        parts.append({"type": "text", "text": str(content)})
    return parts


def merge_reasoning_into_content(msg: dict[str, Any], reasoning: str | None) -> None:
    if not reasoning:
        return
    msg["content"] = content_with_reasoning(msg.get("content"), reasoning)
    msg["reasoning_content"] = ""


def preserve_tool_call_reasoning(msg: dict[str, Any], reasoning: str | None) -> None:
    if msg.get("content") is None:
        msg["content"] = ""
    if reasoning:
        msg["reasoning_content"] = reasoning
