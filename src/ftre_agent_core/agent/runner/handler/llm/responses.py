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
    
    格式转换：
    - tools: completion 嵌套格式 → responses 扁平格式
    - messages: role=tool 的工具结果 → type=function_call_output
    """

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """
        将 completion 格式的 tools 转换为 responses 格式
        
        completion: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        responses:  {"type": "function", "name": ..., "description": ..., "parameters": ...}
        """
        converted = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                converted.append({
                    "type": "function",
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                })
            else:
                converted.append(tool)
        return converted

    def _convert_messages(self, messages: list[dict]) -> list[dict]:
        """
        将 completion 格式的 messages 转换为 responses 格式
        
        主要处理工具结果：
        completion: {"role": "tool", "tool_call_id": "...", "content": "..."}
        responses:  {"type": "function_call_output", "call_id": "...", "output": "..."}
        """
        converted = []
        for msg in messages:
            if msg.get("role") == "tool":
                converted.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": msg.get("content", ""),
                })
            else:
                converted.append(msg)
        return converted

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None
    ) -> Generator[StreamDelta | LLMResponse, None, None]:
        converted_messages = self._convert_messages(messages)
        converted_tools = self._convert_tools(tools) if tools else None
        
        dump_llm_input(converted_messages, converted_tools, self.model)

        response = litellm.responses(
            model=self.model,
            input=converted_messages,
            tools=converted_tools,
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
