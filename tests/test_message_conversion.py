from ftre_agent_core.message import (
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
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


def test_tool_result_text_blocks_are_flattened_for_provider():
    message = to_openai_message(
        [
            ToolResultBlock(
                id="call-1",
                name="bash",
                output=[{"type": "text", "text": "plain output"}],
                state="success",
            )
        ],
        role="tool",
    )

    assert message == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "plain output",
    }
