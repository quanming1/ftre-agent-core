"""
流式适配器基类
"""
from abc import ABC, abstractmethod
from typing import Generator, Callable
from .types import StreamDelta, LLMResponse


class StreamAdapter(ABC):
    """
    流式调用适配器基类
    
    职责：封装特定协议的 LLM 流式调用，输出统一的 StreamDelta / LLMResponse。
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str | None = None,
        cancelled_check: Callable[[], bool] | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self._cancelled_check = cancelled_check or (lambda: False)

    @property
    def is_cancelled(self) -> bool:
        """检查是否已取消"""
        return self._cancelled_check()

    @abstractmethod
    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None
    ) -> Generator[StreamDelta | LLMResponse, None, None]:
        """
        流式调用 LLM
        
        Args:
            messages: 消息列表
            tools: 工具定义列表
            
        Yields:
            StreamDelta: 流式增量（content / tool_calls）
            LLMResponse: 完整响应（当有 tool_calls 时）
        """
        pass
