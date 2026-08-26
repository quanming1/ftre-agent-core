"""
FakeAdapter - 假的 LLM 适配器，用于本地测试，不需要真实 API。

B2：产 StreamChunk 协议（七种 chunk + 配对契约），供 runner 测试与
契约测试复用。

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

    # 在运行前注入到 Runner：
    agent.runner.set_llm(adapter)
"""
import json
import time
from typing import ClassVar

import openai

from ftre_agent_core.llm import (
    BlockEnd,
    BlockStart,
    FinishChunk,
    FinishReason,
    LLMAdapter,
    LlmFailure,
    ReasoningDeltaChunk,
    StreamChunk,
    TextDeltaChunk,
    ToolCall,
    ToolCallDeltaChunk,
    UsageChunk,
)

# ============================================================
# 工具函数
# ============================================================

def _chunk_text(text: str, size: int = 4) -> list[str]:
    """把文本按 size 字符切片，模拟流式输出。"""
    return [text[i:i + size] for i in range(0, len(text), size)]


def legacy_event_sequence(*events) -> list[StreamChunk]:
    """把旧式事件序列（TextDelta/ReasoningDelta/ToolCall/StepFinish）转成
    合法的 StreamChunk 序列——测试迁移辅助（B2 FR7）。

    旧式测试 fake 的典型形态::

        yield TextDelta(text="hello")
        yield StepFinish(finish_reason="stop")

    迁移后::

        for chunk in legacy_event_sequence(
            TextDelta(text="hello"),
            StepFinish(finish_reason="stop"),
        ):
            yield chunk

    转换规则：
    - 旧 TextDelta(text=...)     → BlockStart(text) + TextDeltaChunk + BlockEnd
    - 旧 ReasoningDelta(text=..) → BlockStart(reasoning) + ReasoningDeltaChunk + BlockEnd
    - 旧 ToolCall(id,name,input) → BlockStart(tool-call) + BlockEnd（完整块）
    - 旧 StepFinish(finish_reason="stop"|"tool_calls"|"length"|其他)
      → UsageChunk（如带 usage）+ FinishChunk（kind 映射；未知 kind → error）
    """
    chunks: list[StreamChunk] = []
    index = 0
    kind_map = {"stop": "stop", "tool_calls": "tool-calls", "length": "max-tokens"}
    text_buf: str | None = None
    reasoning_buf: str | None = None

    def flush_text():
        nonlocal index, text_buf
        if text_buf is not None:
            chunks.append(BlockStart(index=index, block_type="text"))
            chunks.append(TextDeltaChunk(index=index, text=text_buf))
            chunks.append(BlockEnd(index=index, block={"type": "text", "text": text_buf}))
            index += 1
            text_buf = None

    def flush_reasoning():
        nonlocal index, reasoning_buf
        if reasoning_buf is not None:
            chunks.append(BlockStart(index=index, block_type="reasoning"))
            chunks.append(ReasoningDeltaChunk(index=index, text=reasoning_buf))
            chunks.append(BlockEnd(index=index, block={"type": "thinking", "thinking": reasoning_buf}))
            index += 1
            reasoning_buf = None

    for ev in events:
        ev_type = getattr(ev, "type", "")
        if ev_type == "text-delta":
            text_buf = (text_buf or "") + ev.text
        elif ev_type == "reasoning-delta":
            reasoning_buf = (reasoning_buf or "") + ev.text
        elif ev_type == "tool-call":
            # 旧 ToolCall（完整调用）→ 一个完整 tool-call 块
            flush_text()
            flush_reasoning()
            args = json.dumps(ev.input, ensure_ascii=False) if ev.input is not None else ""
            chunks.append(BlockStart(index=index, block_type="tool-call"))
            chunks.append(BlockEnd(index=index, block={
                "type": "tool-call", "id": ev.id, "name": ev.name, "arguments": args,
            }))
            index += 1
        elif ev_type == "step-finish":
            flush_text()
            flush_reasoning()
            if getattr(ev, "usage", None):
                chunks.append(UsageChunk(usage=ev.usage))
            raw = getattr(ev, "finish_reason", "unknown")
            kind = kind_map.get(raw)
            if kind is None:
                # 与适配器同语义：有内容产出时宽容映射 stop（内容完整可信），
                # 无产出时 error（接住 Muse 式 finish_reason: null）
                has_output = bool(text_buf or reasoning_buf) or any(
                    isinstance(c, BlockEnd) for c in chunks
                )
                if has_output:
                    kind = "stop"
            if kind is None:
                chunks.append(FinishChunk(reason=FinishReason(
                    kind="error",
                    failure=LlmFailure(message=f"model stopped: {raw}", code=str(raw).upper()),
                    raw=raw,
                    response_metadata=getattr(ev, "response_metadata", {}) or {},
                )))
            else:
                chunks.append(FinishChunk(reason=FinishReason(
                    kind=kind, raw=raw,
                    response_metadata=getattr(ev, "response_metadata", {}) or {},
                )))
    return chunks


def _make_error(code: str) -> Exception:
    """根据 code 构造对应的 openai SDK 异常（LLMError.classify 可识别）。"""
    msg = f"[FakeAdapter] 模拟错误: {code}"
    mapping = {
        "rate_limit":     openai.RateLimitError,
        "timeout":        openai.APITimeoutError,
        "network":        openai.APIConnectionError,
        "auth_error":     openai.AuthenticationError,
    }
    cls = mapping.get(code)
    if cls is None:
        raise ValueError(f"不支持的错误类型: {code}，可选: {list(mapping)}")
    if cls is openai.APITimeoutError:
        return cls(request=None)
    if cls is openai.APIConnectionError:
        return cls(request=None)
    # RateLimitError / AuthenticationError 需要 response 构造参数
    class _FakeResponse:
        status_code = 429 if cls is openai.RateLimitError else 401
        headers: ClassVar[dict] = {}
        def json(self):
            return {}
        def text(self):
            return msg
    return cls(message=msg, response=_FakeResponse(), body=None)


# ============================================================
# 旧事件形态 shim（测试迁移辅助，B2 FR7）
# ============================================================
# 旧测试大量使用 `yield TextDelta(text=...) + StepFinish(finish_reason=...)`
# 形态的 fake LLM。这里提供三个纯数据 shim + sequence_events() 组合器，
# 使旧式 fake 以最小改动迁移到 StreamChunk 协议：
#   for chunk in seq(TextDelta(text="hi"), StepFinish(finish_reason="stop")): yield chunk


from dataclasses import dataclass
from dataclasses import field as _dc_field

__all__ = ["StepFinish", "TextDelta", "ToolCall", "seq"]


@dataclass
class TextDelta:
    """旧事件 shim：正文文本增量（配合 legacy_event_sequence 使用）。"""
    type: str = _dc_field(default="text-delta", init=False)
    text: str = ""


@dataclass
class ReasoningDelta:
    """旧事件 shim：推理文本增量（配合 legacy_event_sequence 使用）。"""
    type: str = _dc_field(default="reasoning-delta", init=False)
    text: str = ""


@dataclass
class StepFinish:
    """旧事件 shim：一轮结束（配合 legacy_event_sequence 使用）。"""
    type: str = _dc_field(default="step-finish", init=False)
    finish_reason: str = "unknown"
    usage: dict | None = None
    response_metadata: dict = _dc_field(default_factory=dict)


def seq(*events) -> list[StreamChunk]:
    """旧事件序列 → StreamChunk 序列（legacy_event_sequence 的短名）。"""
    return legacy_event_sequence(*events)


# ============================================================
# FakeAdapter
# ============================================================

class FakeAdapter(LLMAdapter):
    """假的 LLM 适配器，实现 LLMAdapter 契约（StreamChunk 协议）。"""

    def __init__(self, scenario: dict):
        self.model = "fake"
        self._scenario = scenario

    def cancel(self) -> None:
        """fake 适配器不需要取消逻辑（同步产 chunk）。"""

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
    # 内部：文本流式输出（block 配对 + usage + finish）
    # --------------------------------------------------------

    def _stream_text(self, content: str, chunk_size: int = 4, delay: float = 0.0):
        yield BlockStart(index=0, block_type="text")
        for chunk in _chunk_text(content, chunk_size):
            if delay > 0:
                time.sleep(delay)
            yield TextDeltaChunk(index=0, text=chunk)
        yield BlockEnd(index=0, block={"type": "text", "text": content})
        yield UsageChunk(usage={
            "prompt_tokens": 10,
            "completion_tokens": len(content) // 4,
            "total_tokens": 10 + len(content) // 4,
        })
        yield FinishChunk(reason=FinishReason(kind="stop"))

    # --------------------------------------------------------
    # 内部：工具调用输出
    # --------------------------------------------------------

    def _stream_tool_call(self, call_id: str, name: str, arguments: dict):
        args_str = json.dumps(arguments, ensure_ascii=False)

        yield BlockStart(index=0, block_type="tool-call")
        # 流式透出 arguments delta（模拟真实 API 分批推送参数）
        for chunk in _chunk_text(args_str, size=8):
            yield ToolCallDeltaChunk(
                index=0, call_id=call_id, name=name, arguments_delta=chunk,
            )
        yield BlockEnd(index=0, block={
            "type": "tool-call",
            "id": call_id,
            "name": name,
            "arguments": args_str,
        })
        yield UsageChunk(usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30})
        yield FinishChunk(reason=FinishReason(kind="tool-calls"))
