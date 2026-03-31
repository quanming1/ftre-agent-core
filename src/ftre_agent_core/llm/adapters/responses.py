"""
OpenAI Responses API (/v1/responses) 适配器

将 Chat Completions 格式的请求/响应翻译为 Responses API 格式，
使上层代码（LLMHandler）无需感知底层协议差异。

协议差异要点（Chat Completions vs Responses API）：
  - role: system     → role: developer
  - messages 数组    → input 数组（混合 role-based 消息和 typed items）
  - assistant + tool_calls → 拆分为独立的 function_call items
  - role: tool       → type: function_call_output
  - tools[].function → 扁平化（name/description/parameters 提升到顶层）
  - 响应 output 数组中 message.content[].output_text → choices[0].message.content
  - 流式事件：output_item.added 携带 call_id/name，arguments.delta 携带增量
"""
from ..registry import register_adapter
from ..base import BaseProtocolAdapter
from ..types import (
    FakeUsage, FakeMessage, FakeChoice, FakeResponse,
    FakeDelta, FakeFunctionDelta, FakeToolCallDelta,
    FakeStreamChoice, FakeChunk,
)

@register_adapter("responses")
def _create_responses_client(api_key: str, base_url: str):
    return ResponsesAdapter(api_key, base_url)

class ResponsesAdapter(BaseProtocolAdapter):
    """OpenAI Responses API → Chat Completions 兼容层"""

    def _get_endpoint(self) -> str:
        return "/responses"

    # ================================================================
    # 请求翻译
    # ================================================================

    def _convert_messages(self, messages: list[dict]) -> list[dict]:
        """
        Chat Completions messages → Responses API input 数组。

        转换规则：
          system    → { role: "developer", content }
          user      → { role: "user", content }
          assistant → 纯文本时保持 role-based；带 tool_calls 时拆分为：
                      - (可选) { type: "message", role: "assistant", content }
                      - 每个 tool_call → { type: "function_call", call_id, name, arguments }
          tool      → { type: "function_call_output", call_id, output }
        """
        items = []
        for msg in messages:
            role = msg.get("role", "user")

            if role == "system":
                # Responses API 用 developer 替代 system
                items.append({
                    "role": "developer",
                    "content": msg.get("content", ""),
                })

            elif role == "assistant":
                content = msg.get("content")
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    # 有工具调用时，文本和调用拆分为独立 items
                    if content:
                        items.append({"type": "message", "role": "assistant", "content": content})
                    for tc in tool_calls:
                        fn = tc.get("function", tc) if isinstance(tc, dict) else tc
                        fn_dict = fn if isinstance(fn, dict) else {"name": fn.name, "arguments": fn.arguments}
                        items.append({
                            "type": "function_call",
                            "call_id": tc.get("id", "") if isinstance(tc, dict) else tc.id,
                            "name": fn_dict.get("name", ""),
                            "arguments": fn_dict.get("arguments", ""),
                        })
                elif content:
                    items.append({"role": "assistant", "content": content})

            elif role == "tool":
                # 工具执行结果 → function_call_output，call_id 必须匹配上方的 function_call
                items.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": msg.get("content", ""),
                })

            else:
                items.append({
                    "role": "user",
                    "content": msg.get("content", ""),
                })

        return items

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """
        Chat Completions tools → Responses API tools。

        Chat Completions 格式：{ type: "function", function: { name, description, parameters } }
        Responses API 格式：  { type: "function", name, description, parameters }（扁平化）
        """
        result = []
        for t in tools:
            if t.get("type") == "function":
                fn = t["function"]
                result.append({
                    "type": "function",
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": self._fix_schema(fn.get("parameters", {})),
                })
            else:
                result.append(t)
        return result

    def _build_request_body(self, model, converted_input, converted_tools, **kwargs):
        body = {"model": model, "input": converted_input}
        if converted_tools:
            body["tools"] = converted_tools
        return body

    # ================================================================
    # Schema 修复
    # ================================================================

    @staticmethod
    def _fix_schema(schema: dict) -> dict:
        """
        递归修复不完整的 JSON Schema，使其满足 Responses API 的严格校验。

        已知问题：
          - type: "array" 缺少 items 字段 → 补 items: {}
            （部分 MCP 工具的 schema 不规范，Chat Completions 容忍但 Responses API 拒绝）
        """
        if not isinstance(schema, dict):
            return schema

        schema = dict(schema)  # 浅拷贝，避免污染原始数据

        if schema.get("type") == "array" and "items" not in schema:
            schema["items"] = {}

        if "properties" in schema:
            schema["properties"] = {
                k: ResponsesAdapter._fix_schema(v)
                for k, v in schema["properties"].items()
            }

        if "items" in schema and isinstance(schema["items"], dict):
            schema["items"] = ResponsesAdapter._fix_schema(schema["items"])

        return schema

    # ================================================================
    # 同步响应翻译
    # ================================================================

    def _convert_response(self, data: dict) -> FakeResponse:
        """
        Responses API 响应 → FakeResponse（Chat Completions 兼容）。

        output 数组中：
          - type: "message" → 提取 content[].output_text 拼接为文本
          - type: "reasoning" → 提取 summary[].text 作为可见文本补充
          - type: "function_call" → 转为 tool_calls 列表
        """
        output = data.get("output", [])
        content = None
        tool_calls = []

        for item in output:
            item_type = item.get("type", "")
            if item_type == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        content = (content or "") + c.get("text", "")
            elif item_type == "reasoning":
                for s in item.get("summary", []):
                    text = s.get("text")
                    if text:
                        content = (content or "") + text
            elif item_type == "function_call":
                tool_calls.append({
                    "id": item.get("call_id", ""),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", ""),
                    },
                })

        usage_data = data.get("usage", {})
        usage = FakeUsage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        status = data.get("status")
        if tool_calls:
            finish_reason = "tool_calls"
        elif status == "incomplete":
            finish_reason = "length"
        else:
            finish_reason = "stop"

        message = FakeMessage(content=content, tool_calls=tool_calls or None)
        choice = FakeChoice(message=message, finish_reason=finish_reason)
        return FakeResponse(choices=[choice], usage=usage)

    # ================================================================
    # 流式事件翻译
    # ================================================================

    def _convert_stream_event(self, event: dict) -> FakeChunk | None:
        """
        Responses API SSE 事件 → FakeChunk（Chat Completions stream 兼容）。

        事件映射：
          response.output_text.delta              → content delta
          response.output_item.added (function_call) → tool_call 首帧（id + name）
          response.function_call_arguments.delta   → tool_call arguments 增量
          response.completed                       → 提取 usage，不产出 chunk
        """
        evt_type = event.get("type", "")

        # 文本增量
        if evt_type == "response.output_text.delta":
            delta = FakeDelta(content=event.get("delta", ""), tool_calls=None)
            return FakeChunk(choices=[FakeStreamChoice(delta=delta)], usage=None)

        # 工具调用开始：output_item.added 携带 call_id 和 name
        # （注意：arguments.delta/done 事件中没有这两个字段，只有 item_id）
        elif evt_type == "response.output_item.added":
            item = event.get("item", {})
            if item.get("type") != "function_call":
                return None
            tc = FakeToolCallDelta(
                index=event.get("output_index", 0),
                id=item.get("call_id", ""),
                function=FakeFunctionDelta(name=item.get("name", ""), arguments=""),
            )
            delta = FakeDelta(content=None, tool_calls=[tc])
            return FakeChunk(choices=[FakeStreamChoice(delta=delta)], usage=None)

        # 工具调用参数增量
        elif evt_type == "response.function_call_arguments.delta":
            tc = FakeToolCallDelta(
                index=event.get("output_index", 0),
                id=None,
                function=FakeFunctionDelta(name=None, arguments=event.get("delta", "")),
            )
            delta = FakeDelta(content=None, tool_calls=[tc])
            return FakeChunk(choices=[FakeStreamChoice(delta=delta)], usage=None)

        # 响应完成：提取 usage 统计（由 base._stream 在末尾 yield）
        elif evt_type == "response.completed":
            resp = event.get("response", {})
            usage_data = resp.get("usage", {})
            self._last_usage = FakeUsage(
                prompt_tokens=usage_data.get("input_tokens", 0),
                completion_tokens=usage_data.get("output_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )
            return None

        return None
