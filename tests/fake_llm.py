"""
FakeAdapter - 假的 LLM 适配器，用于本地测试，不需要真实 API。

支持几种场景：
  - 普通文本输出
  - 工具调用
  - 模拟 rate_limit / timeout / network 报错（用于测试重试逻辑）

用法：
    adapter = FakeAdapter.text("你好，我是假模型。")
    adapter = FakeAdapter.tool_call("get_weather", {"city": "北京"})
    adapter = FakeAdapter.error("rate_limit")          # 直接报错
    adapter = FakeAdapter.error_then_text(             # 先报错 N 次，再成功
        "rate_limit", retries=2, text="重试成功！"
    )

    # 替换到 LLMHandler 上：
    agent.runner.llm._adapter = adapter
"""
import json
import time
import litellm

from ftre_agent_core.llm.completion import (
    StreamAdapter,
    StreamDelta,
    LLMResponse,
    ToolCallWrapper,
    ToolCallDeltaChunk,
)


# ============================================================
# 工具函数
# ============================================================

def _chunk_text(text: str, size: int = 4) -> list[str]:
    """把文本按 size 字符切片，模拟流式输出。"""
    return [text[i:i + size] for i in range(0, len(text), size)]


def _make_error(code: str) -> Exception:
    """根据 code 构造对应的 litellm 异常。"""
    msg = f"[FakeAdapter] 模拟错误: {code}"
    mapping = {
        "rate_limit":     litellm.RateLimitError,
        "timeout":        litellm.Timeout,
        "network":        litellm.APIConnectionError,
        "api_error":      litellm.APIError,
    }
    cls = mapping.get(code)
    if cls is None:
        raise ValueError(f"不支持的错误类型: {code}，可选: {list(mapping)}")
    return cls(msg, llm_provider="fake", model="fake")


# ============================================================
# FakeAdapter
# ============================================================

class FakeAdapter(StreamAdapter):
    """假的 LLM 适配器，直接继承 StreamAdapter。"""

    def __init__(self, scenario: dict):
        """不调用父类 __init__，直接设置必要属性。"""
        self.model = "fake"
        self.api_key = "fake"
        self.api_base = None
        self._cancelled_check = lambda: False
        self._active_response = None
        self._scenario = scenario

    # --------------------------------------------------------
    # 工厂方法
    # --------------------------------------------------------

    @classmethod
    def text(cls, content: str, chunk_size: int = 4, delay: float = 0.0) -> "FakeAdapter":
        """普通文本输出场景。"""
        return cls({"type": "text", "content": content, "chunk_size": chunk_size, "delay": delay})

    @classmethod
    def tool_call(cls, name: str, arguments: dict, call_id: str = "fake_call_001") -> "FakeAdapter":
        """工具调用场景。"""
        return cls({"type": "tool_call", "name": name, "arguments": arguments, "call_id": call_id})

    @classmethod
    def error(cls, code: str) -> "FakeAdapter":
        """直接报错，不输出任何内容。"""
        return cls({"type": "error", "code": code})

    @classmethod
    def error_then_text(cls, code: str, retries: int, text: str) -> "FakeAdapter":
        """先连续报错 retries 次，然后成功输出 text。用于测试重试逻辑。"""
        return cls({"type": "error_then_text", "code": code, "retries": retries, "text": text, "_attempt": 0})

    # --------------------------------------------------------
    # stream 实现
    # --------------------------------------------------------

    def stream(self, messages: list[dict], tools: list[dict] | None = None):
        scenario = self._scenario
        t = scenario["type"]

        if t == "text":
            yield from self._stream_text(scenario["content"], scenario["chunk_size"], scenario["delay"])

        elif t == "tool_call":
            yield from self._stream_tool_call(scenario["call_id"], scenario["name"], scenario["arguments"])

        elif t == "error":
            raise _make_error(scenario["code"])

        elif t == "error_then_text":
            if scenario["_attempt"] < scenario["retries"]:
                scenario["_attempt"] += 1
                raise _make_error(scenario["code"])
            else:
                yield from self._stream_text(scenario["text"])

    # --------------------------------------------------------
    # 内部：文本流式输出
    # --------------------------------------------------------

    def _stream_text(self, content: str, chunk_size: int = 4, delay: float = 0.0):
        for chunk in _chunk_text(content, chunk_size):
            if delay > 0:
                time.sleep(delay)
            yield StreamDelta(content=chunk)
        yield StreamDelta(usage={"prompt_tokens": 10, "completion_tokens": len(content) // 4, "total_tokens": 10 + len(content) // 4})

    # --------------------------------------------------------
    # 内部：工具调用输出
    # --------------------------------------------------------

    def _stream_tool_call(self, call_id: str, name: str, arguments: dict):
        args_str = json.dumps(arguments, ensure_ascii=False)

        # 流式透出 tool_call delta（模拟真实 API 分批推送参数）
        yield StreamDelta(tool_calls=[ToolCallDeltaChunk(index=0, id=call_id, name=name, arguments_delta="")])
        for chunk in _chunk_text(args_str, size=8):
            yield StreamDelta(tool_calls=[ToolCallDeltaChunk(index=0, id=None, name=None, arguments_delta=chunk)])

        # 最后产出 LLMResponse，触发 _handle_tool_calls
        yield LLMResponse(
            content=None,
            tool_calls=[ToolCallWrapper({"id": call_id, "type": "function", "function": {"name": name, "arguments": args_str}})],
            usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        )
