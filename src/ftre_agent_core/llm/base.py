"""
协议适配器基类

子类实现 6 个翻译方法，基类负责：
- 暴露 client.chat.completions.create() 鸭子接口
- HTTP 调用
- 流式 SSE 解析骨架
"""
import json
import httpx
from abc import ABC, abstractmethod

from .types import FakeUsage, FakeChunk, FakeResponse

class BaseProtocolAdapter(ABC):

    def __init__(self, api_key: str, base_url: str, timeout: float = 120.0):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
        self._last_usage: FakeUsage | None = None
        self._active_stream_response: httpx.Response | None = None  # 取消用
        self.chat = _ChatNamespace(self)

    def cancel_stream(self) -> None:
        """强关活跃的 httpx 流式响应，线程安全（关的是 socket fd）。"""
        resp = self._active_stream_response
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass

    # ---------- 子类必须实现 ----------

    @abstractmethod
    def _get_endpoint(self) -> str:
        """目标 API 路径，如 '/responses'"""
        ...

    @abstractmethod
    def _convert_messages(self, messages: list[dict]) -> any:
        """Chat Completions messages → 目标协议输入"""
        ...

    @abstractmethod
    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Chat Completions tools → 目标协议工具格式"""
        ...

    @abstractmethod
    def _build_request_body(self, model: str, converted_input: any,
                            converted_tools: list[dict] | None, **kwargs) -> dict:
        """组装完整请求体"""
        ...

    @abstractmethod
    def _convert_response(self, data: dict) -> FakeResponse:
        """目标协议响应 → FakeResponse"""
        ...

    @abstractmethod
    def _convert_stream_event(self, event: dict) -> FakeChunk | None:
        """目标协议 SSE 事件 → FakeChunk"""
        ...

    # ---------- 公共逻辑 ----------

    def _call(self, model: str, messages: list[dict],
              tools: list[dict] | None = None, stream: bool = False, **kwargs):
        converted_input = self._convert_messages(messages)
        converted_tools = self._convert_tools(tools) if tools else None
        body = self._build_request_body(model, converted_input, converted_tools, **kwargs)

        if stream:
            body["stream"] = True
            return self._stream(body)

        resp = self._http.post(self._get_endpoint(), json=body)
        resp.raise_for_status()
        return self._convert_response(resp.json())

    def _stream(self, body: dict):
        """通用 SSE 流式解析"""
        self._last_usage = None
        with self._http.stream("POST", self._get_endpoint(), json=body) as resp:
            self._active_stream_response = resp
            try:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    chunk = self._convert_stream_event(event)
                    if chunk is not None:
                        yield chunk
            finally:
                self._active_stream_response = None

        yield FakeChunk(choices=[], usage=self._last_usage or FakeUsage())

class _ChatNamespace:
    def __init__(self, adapter: BaseProtocolAdapter):
        self.completions = _CompletionsNamespace(adapter)

class _CompletionsNamespace:
    def __init__(self, adapter: BaseProtocolAdapter):
        self._adapter = adapter

    def create(self, model, messages, tools=None, stream=False, **kwargs):
        return self._adapter._call(
            model=model, messages=messages, tools=tools, stream=stream, **kwargs
        )
