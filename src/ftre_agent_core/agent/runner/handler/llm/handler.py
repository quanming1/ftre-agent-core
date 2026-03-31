"""
LLMHandler - LLM 调用封装

职责：封装 LiteLLM 的流式调用，统一输出格式。
根据 api_type 选择对应的适配器。
"""
from typing import Generator

from .base import StreamAdapter
from .completion import CompletionAdapter
from .responses import ResponsesAdapter
from .types import StreamDelta, LLMResponse


class LLMHandler:
    """
    LLM 调用封装

    唯一的调用方式是 stream()，返回 Generator：
    - 检测到 tool_calls → 累积完毕后 yield LLMResponse
    - 只有 content → 边读边 yield StreamDelta

    取消机制：
    - cancel() 设置标志位（软取消）
    - 适配器在每次 yield 前检查标志位

    协议支持：
    - api_type="completions" → CompletionAdapter
    - api_type="responses"   → ResponsesAdapter
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str | None = None,
        api_type: str = "completions"
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.api_type = api_type
        self._cancelled = False
        self._adapter = self._create_adapter()

    def _create_adapter(self) -> StreamAdapter:
        """根据 api_type 创建对应的适配器"""
        if self.api_type == "responses":
            return ResponsesAdapter(
                model=self.model,
                api_key=self.api_key,
                api_base=self.api_base,
                cancelled_check=lambda: self._cancelled,
            )
        return CompletionAdapter(
            model=self.model,
            api_key=self.api_key,
            api_base=self.api_base,
            cancelled_check=lambda: self._cancelled,
        )

    def cancel(self) -> None:
        """设置取消标志位（软取消）。线程安全。"""
        self._cancelled = True

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None
    ) -> Generator[StreamDelta | LLMResponse, None, None]:
        """流式调用 LLM，委托给适配器"""
        self._cancelled = False
        yield from self._adapter.stream(messages, tools)
