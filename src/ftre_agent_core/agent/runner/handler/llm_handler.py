"""
LLMHandler - LLM 调用封装

职责：封装 LiteLLM 的流式调用，统一输出格式。
对上层屏蔽流式 chunk 拼接、tool_calls 累积等细节。
"""
import json
import logging
import litellm
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Generator, Any
from uuid import uuid4

logger = logging.getLogger(__name__)

_DEBUG_LLM_INPUT_DIR = Path("data/logs/llm_input")

def _dump_llm_input(messages: list[dict], tools: list[dict] | None, model: str) -> None:
    try:
        now = datetime.now()
        day_dir = _DEBUG_LLM_INPUT_DIR / now.strftime("%Y%m%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        file_path = day_dir / f"{now.strftime('%H%M%S_%f')}_{uuid4().hex[:8]}.json"
        payload = {
            "timestamp": now.isoformat(),
            "model": model,
            "messages": messages,
            "tools": tools,
        }
        file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass

@dataclass
class LLMError:
    """LLM 调用错误"""
    message: str
    code: str

    @staticmethod
    def classify(e: Exception) -> "LLMError":
        """根据异常类型分类错误（适配 LiteLLM）"""
        if isinstance(e, litellm.RateLimitError):
            return LLMError(message=f"请求频率超限: {e}", code="rate_limit")
        if isinstance(e, litellm.Timeout):
            return LLMError(message=f"请求超时: {e}", code="timeout")
        if isinstance(e, litellm.APIConnectionError):
            return LLMError(message=f"网络连接失败: {e}", code="network")
        if isinstance(e, litellm.ContentPolicyViolationError):
            return LLMError(message=f"内容审核未通过: {e}", code="content_filter")
        if isinstance(e, litellm.APIError):
            return LLMError(message=f"API 错误: {e}", code="api_error")
        return LLMError(message=f"未知错误: {e}", code="unknown")

@dataclass
class LLMResponse:
    """LLM 完整响应（用于 tool_calls 场景）"""
    content: str | None = None
    tool_calls: list[Any] = field(default_factory=list)
    usage: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

@dataclass
class ToolCallDeltaChunk:
    """单个 tool_call 的增量信息"""
    index: int
    id: str | None = None
    name: str | None = None
    arguments_delta: str | None = None

@dataclass
class StreamDelta:
    """流式输出的 delta 片段"""
    content: str | None = None
    tool_calls: list[ToolCallDeltaChunk] | None = None
    usage: Any = None


# =========================================================================
#  ToolCallAccumulator — 流式 tool_call 拼接器
# =========================================================================

class _SplitReason(Enum):
    """index 复用拆分原因"""
    ID_CHANGED = auto()       # 同 index 收到不同 id
    ARGS_COMPLETE = auto()    # 已有 arguments 是完整 JSON，新 delta 不应追加
    NAME_CHANGED = auto()     # 同 index 收到不同 tool name


class ToolCallAccumulator:
    """
    流式 tool_call delta 累积器。

    职责：
    - 按 index 累积 id / name / arguments
    - 检测 provider 复用 index 的异常情况，自动拆分到新的虚拟 slot
    - 最终输出去重后的完整 tool_call 列表

    用法：
        acc = ToolCallAccumulator()
        for tc_delta in chunk.tool_calls:
            delta_chunk = acc.feed(tc_delta)
        tool_calls = acc.build()
    """

    def __init__(self):
        self._buffer: dict[int, dict] = {}       # idx -> {"id", "type", "function": {"name", "arguments"}}
        self._index_remap: dict[int, int] = {}    # raw_idx -> virtual_idx（仅在检测到复用时添加）

    # -----------------------------------------------------------------
    #  公开接口
    # -----------------------------------------------------------------

    def feed(self, tc) -> ToolCallDeltaChunk:
        """
        喂入一个 tool_call delta，返回前端需要的 ToolCallDeltaChunk。
        tc 是 LiteLLM 流式 chunk 中的 tool_call 对象。
        """
        raw_idx = tc.index
        idx = self._index_remap.get(raw_idx, raw_idx)
        existing = self._buffer.get(idx)

        # 检测 index 复用
        split_reason = self._detect_split(existing, tc)
        if split_reason is not None:
            idx = self._allocate_new_slot(raw_idx, tc.id, split_reason)
            existing = None

        # 初始化或更新 buffer
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

    def build(self) -> list["_ToolCallWrapper"]:
        """累积结束，构建最终的 tool_call 对象列表（按 index 排序）。"""
        return [
            _ToolCallWrapper(tc_data)
            for _, tc_data in sorted(self._buffer.items())
        ]

    # -----------------------------------------------------------------
    #  内部方法
    # -----------------------------------------------------------------

    def _detect_split(self, existing: dict | None, tc) -> _SplitReason | None:
        """
        检测当前 delta 是否应该拆分为新的 tool_call。
        返回拆分原因，None 表示不需要拆分。
        """
        if not existing:
            return None

        # Case 1: 同 index 出现了不同的 id → 明确是新的 tool_call
        if tc.id and existing["id"] and tc.id != existing["id"]:
            return _SplitReason.ID_CHANGED

        # Case 2: 已有 arguments 已经是完整 JSON，新 delta 还在追加
        #         说明 provider 把多个 tool_call 塞进了同一个 index
        if tc.function and tc.function.arguments:
            cur_args = existing["function"]["arguments"]
            if cur_args:
                try:
                    json.loads(cur_args)
                    return _SplitReason.ARGS_COMPLETE
                except json.JSONDecodeError:
                    pass

        # Case 3: 同 index 出现了不同的 tool name
        if tc.function and tc.function.name:
            cur_name = existing["function"]["name"]
            if cur_name and cur_name != tc.function.name:
                return _SplitReason.NAME_CHANGED

        return None

    def _allocate_new_slot(self, raw_idx: int, tc_id: str | None, reason: _SplitReason) -> int:
        """分配虚拟 index 并更新 remap 表。"""
        new_idx = max(self._buffer.keys()) + 1 if self._buffer else 0
        self._index_remap[raw_idx] = new_idx
        logger.warning(
            "tool_call index 复用，拆分到新 slot: raw_idx=%d -> idx=%d, id=%s, reason=%s",
            raw_idx, new_idx, tc_id, reason.name,
        )
        return new_idx

    def _merge_delta(self, idx: int, tc) -> None:
        """将 delta 数据合并到 buffer[idx]。"""
        entry = self._buffer[idx]
        if tc.id:
            entry["id"] = tc.id
        if tc.function:
            if tc.function.name:
                entry["function"]["name"] = tc.function.name
            if tc.function.arguments:
                entry["function"]["arguments"] += tc.function.arguments


# =========================================================================
#  LLMHandler
# =========================================================================

class LLMHandler:
    """
    LLM 调用封装

    唯一的调用方式是 stream()，返回 Generator：
    - 检测到 tool_calls → 累积完毕后 yield LLMResponse
    - 只有 content → 边读边 yield StreamDelta

    取消机制：
    - cancel() 设置标志位（软取消）
    - stream() 生成器在每次 yield 前检查标志位
    """

    def __init__(self, model: str, api_key: str, api_base: str | None = None):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self._cancelled = False

    def cancel(self) -> None:
        """设置取消标志位（软取消）。线程安全。"""
        self._cancelled = True

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None
    ) -> Generator[StreamDelta | LLMResponse, None, None]:
        """流式调用 LLM（使用 LiteLLM）"""
        self._cancelled = False
        _dump_llm_input(messages, tools, self.model)

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
                if self._cancelled:
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


class _ToolCallWrapper:
    """模拟 tool_call 对象"""
    def __init__(self, data: dict):
        self.id = data["id"]
        self.type = data["type"]
        self.function = _FunctionWrapper(data["function"])

class _FunctionWrapper:
    """模拟 function 对象（tc.function.name / tc.function.arguments）"""
    def __init__(self, data: dict):
        self.name = data["name"]
        self.arguments = data["arguments"]
