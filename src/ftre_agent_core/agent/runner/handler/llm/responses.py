"""
Responses API 适配器
"""
import litellm
from typing import Generator

from .base import StreamAdapter
from .types import StreamDelta, LLMResponse, ToolCallWrapper
from .utils import dump_llm_input


class ResponsesAdapter(StreamAdapter):
    """
    Responses API 适配器
    
    使用 litellm.responses() 进行流式调用。
    """

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None
    ) -> Generator[StreamDelta | LLMResponse, None, None]:
        dump_llm_input(messages, tools, self.model)

        response = litellm.responses(
            model=self.model,
            input=messages,
            tools=tools if tools else None,
            api_key=self.api_key,
            api_base=self.api_base,
            stream=True,
        )

        content_buffer: list[str] = []
        tool_calls_buffer: list[dict] = []
        usage = None

        try:
            for event in response:
                if self.is_cancelled:
                    break

                event_type = getattr(event, "type", None)

                if event_type == "response.output_text.delta":
                    delta_text = getattr(event, "delta", "")
                    if delta_text:
                        content_buffer.append(delta_text)
                        yield StreamDelta(content=delta_text)

                elif event_type == "response.completed":
                    completed_response = getattr(event, "response", None)
                    if completed_response:
                        usage = getattr(completed_response, "usage", None)
                        for item in getattr(completed_response, "output", []):
                            if getattr(item, "type", None) == "function_call":
                                tool_calls_buffer.append({
                                    "id": getattr(item, "call_id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": getattr(item, "name", ""),
                                        "arguments": getattr(item, "arguments", "")
                                    }
                                })

            if tool_calls_buffer:
                yield LLMResponse(
                    content="".join(content_buffer) if content_buffer else None,
                    tool_calls=[ToolCallWrapper(tc) for tc in tool_calls_buffer],
                    usage=usage,
                )
            else:
                if usage:
                    yield StreamDelta(usage=usage)
        finally:
            pass
