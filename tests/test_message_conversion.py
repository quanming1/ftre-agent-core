from ftre_agent_core.message import (
    TextBlock,
    ThinkingBlock,
    to_openai_message,
)


def test_thinking_block_stays_out_of_openai_content_parts():
    message = to_openai_message(
        [
            ThinkingBlock(thinking="internal reasoning"),
            TextBlock(text="visible answer"),
        ],
        role="assistant",
    )

    assert message == {
        "role": "assistant",
        "content": [{"type": "text", "text": "visible answer"}],
        "reasoning_content": "internal reasoning",
    }


def test_thinking_only_message_uses_reasoning_content():
    message = to_openai_message(
        [ThinkingBlock(thinking="internal reasoning")],
        role="assistant",
    )

    assert message == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "internal reasoning",
    }
