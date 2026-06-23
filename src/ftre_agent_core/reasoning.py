"""Helpers for formatting historical reasoning in assistant messages."""

from __future__ import annotations

from typing import Any


def content_with_reasoning(content: Any, reasoning: str) -> list[dict[str, str]]:
    parts: list[dict[str, str]] = [{"type": "text", "text": reasoning}]
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                parts.append(item)
            else:
                parts.append({"type": "text", "text": str(item)})
    elif content not in (None, ""):
        parts.append({"type": "text", "text": str(content)})
    return parts


def attach_reasoning(msg: dict[str, Any], reasoning: str | None) -> None:
    if not reasoning:
        return
    msg["content"] = content_with_reasoning(msg.get("content"), reasoning)
    msg["reasoning_content"] = ""
