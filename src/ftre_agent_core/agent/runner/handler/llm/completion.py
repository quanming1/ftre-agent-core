"""
Completions API 适配器
"""
import json
import logging
import litellm
from enum import Enum, auto
from typing import Generator

from .base import StreamAdapter
from .types import StreamDelta, LLMResponse, ToolCallDeltaChunk, ToolCallWrapper
from .utils import dump_llm_input

logger = logging.getLogger(__name__)


class _SplitReason(Enum):
    """index 复用拆分原因"""
    ID_CHANGED = auto()
    ARGS_COMPLETE = auto()
    NAME_CHANGED = auto()


class ToolCallAccumulator:
    """
    流式 tool_call delta 累积器。

    职责：
    - 按 index 累积 id / name / arguments
    - 检测 provider 复用 index 的异常情况，自动拆分到新的虚拟 slot
    - 最终输出去重后的完整 tool_call 列表
    """

    def __init__(self):
        self._buffer: dict[int, dict] = {}
        self._index_remap: dict[int, int] = {}

    def feed(self, tc) -> ToolCallDeltaChunk:
        """喂入一个 tool_call delta，返回 ToolCallDeltaChunk。"""
        raw_idx = tc.index
        idx = self._index_remap.get(raw_idx, raw_idx)
        existing = self._buffer.get(idx)

        split_reason = self._detect_split(existing, tc)
        if split_reason is not None:
            idx = self._allocate_new_slot(raw_idx, tc.id, split_reason)
            existing = None

        if not existing:
            self._buffer[idx] = {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }
        self._merge_delta(idx, tc)

        return ToolCallDeltaChunk(
            index=idx,
            id=tc.id or None,
            name=tc.function.name if tc.function and tc.function.name else None,
            arguments_delta=tc.function.arguments if tc.function and tc.function.arguments else None,
        )

    @property
    def has_data(self) -> bool:
        return len(self._buffer) > 0

    def build(self) -> list[ToolCallWrapper]:
        """构建最终的 tool_call 对象列表（按 index 排序）。"""
        return [
            ToolCallWrapper(tc_data)
            for _, tc_data in sorted(self._buffer.items())
        ]

    def _detect_split(self, existing: dict | None, tc) -> _SplitReason | None:
        if not existing:
            return None

        if tc.id and existing["id"] and tc.id != existing["id"]:
            return _SplitReason.ID_CHANGED

        if tc.function and tc.function.arguments:
            cur_args = existing["function"]["arguments"]
            if cur_args:
                try:
                    json.loads(cur_args)
                    return _SplitReason.ARGS_COMPLETE
                except json.JSONDecodeError:
                    pass

        if tc.function and tc.function.name:
            cur_name = existing["function"]["name"]
            if cur_name and cur_name != tc.function.name:
                return _SplitReason.NAME_CHANGED

        return None

    def _allocate_new_slot(self, raw_idx: int, tc_id: str | None, reason: _SplitReason) -> int:
        new_idx = max(self._buffer.keys()) + 1 if self._buffer else 0
        self._index_remap[raw_idx] = new_idx
        logger.warning(
            "tool_call index 复用，拆分到新 slot: raw_idx=%d -> idx=%d, id=%s, reason=%s",
            raw_idx, new_idx, tc_id, reason.name,
        )
        return new_idx

    def _merge_delta(self, idx: int, tc) -> None:
        entry = self._buffer[idx]
        if tc.id:
            entry["id"] = tc.id
        if tc.function:
            if tc.function.name:
                entry["function"]["name"] = tc.function.name
            if tc.function.arguments:
                entry["function"]["arguments"] += tc.function.arguments


class CompletionAdapter(StreamAdapter):
    """
    Completions API 适配器
    
    使用 litellm.completion() 进行流式调用。
    """

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None
    ) -> Generator[StreamDelta | LLMResponse, None, None]:
        dump_llm_input(messages, tools, self.model)

        response = litellm.completion(
            model=self.model,
            messages=messages,
            tools=tools if tools else None,
            api_key=self.api_key,
            api_base=self.api_base,
            stream=True,
            stream_options={"include_usage": True},
        )

        accumulator = ToolCallAccumulator()
        content_buffer: list[str] = []
        usage = None

        try:
            for chunk in response:
                if self.is_cancelled:
                    break

                if hasattr(chunk, "usage") and chunk.usage:
                    usage = chunk.usage
                if not hasattr(chunk, "choices") or not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    tc_deltas = [accumulator.feed(tc) for tc in delta.tool_calls]
                    yield StreamDelta(tool_calls=tc_deltas)

                if hasattr(delta, "content") and delta.content:
                    content_buffer.append(delta.content)
                    yield StreamDelta(content=delta.content)

            if accumulator.has_data:
                yield LLMResponse(
                    content="".join(content_buffer) if content_buffer else None,
                    tool_calls=accumulator.build(),
                    usage=usage,
                )
            else:
                if usage:
                    yield StreamDelta(usage=usage)
        finally:
            pass
