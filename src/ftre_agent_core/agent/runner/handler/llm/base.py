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
        self._active_response = None  # 当前活跃的流式 response，供硬取消用

    @property
    def is_cancelled(self) -> bool:
        """检查是否已取消"""
        return self._cancelled_check()

    def close_stream(self) -> None:
        """
        硬关活跃的流式连接，立即中断底层 HTTP。

        LiteLLM 的 CustomStreamWrapper 持有 completion_stream（底层迭代器），
        关闭它可以让 __next__ 立即抛异常，从而跳出 for chunk in response 循环。
        """
        resp = self._active_response
        if resp is None:
            return
        try:
            # LiteLLM CustomStreamWrapper 的底层 stream
            inner = getattr(resp, "completion_stream", None)
            if inner and hasattr(inner, "close"):
                inner.close()
            # httpx response 层
            http_resp = getattr(inner, "response", None)
            if http_resp and hasattr(http_resp, "close"):
                http_resp.close()
        except Exception:
            pass

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
