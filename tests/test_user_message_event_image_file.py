import base64
import os
import tempfile

from ftre_agent_core.agent.event import UserMessageEvent


def _make_temp_image(raw: bytes, suffix: str = ".png") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return path


def test_to_openai_message_converts_image_file_to_image_url():
    raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    path = _make_temp_image(raw, ".png")

    ev = UserMessageEvent(
        content=[{"type": "image_file", "path": path, "mime_type": "image/png"}],
    )

    msg = ev.to_openai_message()

    assert msg["role"] == "user"
    content = msg["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")

    # 验证 base64 内容正确
    b64_part = content[0]["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(b64_part) == raw


def test_to_openai_message_handles_missing_file():
    ev = UserMessageEvent(
        content=[{"type": "image_file", "path": "/nonexistent/image.png", "mime_type": "image/png"}],
    )

    msg = ev.to_openai_message()

    assert msg["role"] == "user"
    content = msg["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "图片加载失败" in content[0]["text"]


def test_to_openai_message_preserves_image_url_type():
    """已有的 image_url 类型不受影响。"""
    ev = UserMessageEvent(
        content=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}],
    )

    msg = ev.to_openai_message()

    assert msg["content"][0] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}


def test_to_openai_message_preserves_text_type():
    ev = UserMessageEvent(content="hello world")

    msg = ev.to_openai_message()

    assert msg == {"role": "user", "content": "hello world"}


def test_to_openai_message_mixed_content():
    raw = b"\x00" * 50
    path = _make_temp_image(raw, ".png")

    ev = UserMessageEvent(
        content=[
            {"type": "text", "text": "看这张图"},
            {"type": "image_file", "path": path, "mime_type": "image/png"},
        ],
    )

    msg = ev.to_openai_message()

    content = msg["content"]
    assert content[0] == {"type": "text", "text": "看这张图"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
