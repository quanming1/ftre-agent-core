"""Responses 历史 Item 与 Vision 输入契约回归。"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from ftre_agent_core.event import ModelCallEndEvent
from ftre_agent_core.llm import FinishChunk, OpenAIResponsesAdapter
from ftre_agent_core.llm.adapters.openai_responses import (
    _convert_messages_to_responses_input,
)
from ftre_agent_core.message import AssistantMsg, ThinkingBlock
from ftre_agent_core.message_context import MessageContext


@pytest.mark.parametrize("raw_status", ["completed", "in_progress"])
def test_reasoning_input_never_invents_return_only_status(raw_status):
    """请求 input 不能携带 API 返回态 status。"""
    _, input_items = _convert_messages_to_responses_input(
        [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "先检查配置。",
                "responses_output_items": [
                    {
                        "type": "reasoning",
                        "id": "rs-original",
                        "summary": [],
                        "content": [{"type": "reasoning_text", "text": "先检查配置。"}],
                        "status": raw_status,
                    }
                ],
            }
        ],
        include_reasoning=True,
    )

    assert input_items == [
        {
            "type": "reasoning",
            "id": "rs-original",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "先检查配置。"}],
        }
    ]


def test_legacy_reasoning_replay_is_status_free():
    """旧会话只有 reasoning_content 时，降级也不能拼接 status。"""
    _, input_items = _convert_messages_to_responses_input(
        [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "旧会话思考",
            }
        ],
        include_reasoning=True,
    )

    assert input_items[0]["type"] == "reasoning"
    assert "status" not in input_items[0]


def test_responses_image_input_supports_url_data_url_and_file_id():
    _, input_items = _convert_messages_to_responses_input(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "看这两张图"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,AAAA",
                            "detail": "high",
                        },
                    },
                    {"type": "input_image", "file_id": "file-123"},
                ],
            }
        ]
    )

    assert input_items == [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "看这两张图"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AAAA",
                    "detail": "high",
                },
                {"type": "input_image", "file_id": "file-123"},
            ],
        }
    ]


@dataclass
class _Event:
    item: object | None = None
    response: object | None = None


class _ResponsesStream:
    def __init__(self, events):
        self.events = list(events)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.events:
            raise StopAsyncIteration
        event_type, event = self.events.pop(0)
        event.__class__ = type(event_type, (type(event),), {"__name__": event_type})
        return event

    def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_output_item_done_is_exposed_as_json_safe_metadata():
    item = SimpleNamespace(
        id="rs-original",
        type="reasoning",
        summary=[],
        content=[{"type": "reasoning_text", "text": "原始思考"}],
        status="completed",
    )
    response = SimpleNamespace(
        id="resp-1",
        model="test-model",
        created_at=1,
        status="completed",
        incomplete_details=None,
        usage=None,
    )
    stream = _ResponsesStream(
        [
            ("ResponseOutputItemDoneEvent", _Event(item=item)),
            ("ResponseCompletedEvent", _Event(response=response)),
        ]
    )
    adapter = OpenAIResponsesAdapter(model="test-model", api_key="k")

    async def create(**kwargs):
        return stream

    adapter._client.responses.create = create  # type: ignore[attr-defined]
    chunks = [chunk async for chunk in adapter.stream([{"role": "user", "content": "hi"}])]

    finish = next(chunk for chunk in chunks if isinstance(chunk, FinishChunk))
    output_items = finish.reason.response_metadata["output_items"]
    assert output_items[0]["id"] == "rs-original"
    assert output_items[0]["status"] == "completed"
    assert output_items[0]["content"][0]["type"] == "reasoning_text"
    assert stream.closed


def test_msg_metadata_round_trips_raw_output_item_to_provider_message():
    message = AssistantMsg(content=[ThinkingBlock(thinking="原始思考")], id="msg-1")
    message.append_event(
        ModelCallEndEvent(
            reply_id="msg-1",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            response_metadata={
                "output_items": [
                    {
                        "type": "reasoning",
                        "id": "rs-1",
                        "content": [{"type": "reasoning_text", "text": "原始思考"}],
                    }
                ]
            },
        )
    )

    provider_messages = MessageContext.messages([message])
    assert provider_messages[0]["responses_output_items"][0]["id"] == "rs-1"
