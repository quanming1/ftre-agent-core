"""
Responses API 适配器
"""
import litellm
from typing import Generator, Callable

from .completion import StreamAdapter, StreamDelta, LLMResponse, ToolCallWrapper, normalize_usage
from .utils import LLMLogger


class ResponsesAdapter(StreamAdapter):
    """
    Responses API 适配器

    使用 litellm.responses() 进行流式调用。

    格式转换：
    - tools: completion 嵌套格式 → responses 扁平格式
    - messages: role=tool 的工具结果 → type=function_call_output
    
    工具调用上下文：
    - 保存每次响应的 response.id
    - 回传工具结果时带上 previous_response_id
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str | None = None,
        cancelled_check: Callable[[], bool] | None = None,
    ):
        super().__init__(model, api_key, api_base, cancelled_check)
        self._last_response_id: str | None = None

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

    def _has_tool_results(self, messages: list[dict]) -> bool:
        """检查 messages 中是否有工具结果"""
        return any(msg.get("role") == "tool" for msg in messages)

    def _convert_messages(self, messages: list[dict], tool_results_only: bool = False) -> list[dict]:
        """
        将 completion 格式的 messages 转换为 responses 格式

        主要处理工具结果：
        completion: {"role": "tool", "tool_call_id": "...", "content": "..."}
        responses:  {"type": "function_call_output", "call_id": "...", "output": "..."}
        
        Args:
            messages: 原始消息列表
            tool_results_only: 如果为 True，只返回工具结果（用于 previous_response_id 场景）
        """
        converted = []
        for msg in messages:
            if msg.get("role") == "tool":
                converted.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": msg.get("content", ""),
                })
            elif not tool_results_only:
                converted.append(msg)
        return converted

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None
    ) -> Generator[StreamDelta | LLMResponse, None, None]:
        # 检查是否需要回传工具结果
        has_tool_results = self._has_tool_results(messages)
        use_previous_response = has_tool_results and self._last_response_id
        
        # 如果使用 previous_response_id，只传工具结果，不传完整 messages
        converted_messages = self._convert_messages(messages, tool_results_only=use_previous_response)
        converted_tools = self._convert_tools(tools) if tools else None

        llm_log = LLMLogger(self.model)
        llm_log.log_input(converted_messages, converted_tools)

        kwargs = {
            "model": self.model,
            "input": converted_messages,
            "tools": converted_tools,
            "api_key": self.api_key,
            "api_base": self.api_base,
            "stream": True,
        }
        
        if use_previous_response:
            kwargs["previous_response_id"] = self._last_response_id

        response = litellm.responses(**kwargs)
        self._active_response = response

        content_buffer: list[str] = []
        tool_calls_buffer: list[dict] = []
        usage = None

        try:
            for event in response:
                if self.is_cancelled:
                    break

                llm_log.log_chunk(event)

                event_type = getattr(event, "type", None)

                if event_type == "response.output_text.delta":
                    delta_text = getattr(event, "delta", "")
                    if delta_text:
                        content_buffer.append(delta_text)
                        yield StreamDelta(content=delta_text)

                elif event_type == "response.completed":
                    completed_response = getattr(event, "response", None)
                    if completed_response:
                        # 保存 response.id 供下次工具结果回传使用
                        self._last_response_id = getattr(completed_response, "id", None)
                        
                        usage = normalize_usage(getattr(completed_response, "usage", None))
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
            self._active_response = None
            llm_log.flush()
