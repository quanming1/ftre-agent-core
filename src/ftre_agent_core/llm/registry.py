"""协议注册表 + 工厂（PRD-B2 FR2）。

协议是数据不是代码：api_type 字符串 → 适配器类的映射表。
新协议接入 = adapters/ 加一个文件 + 这里加一行（消费方零改动）。
"""

from __future__ import annotations

from .adapters.openai_completions import OpenAICompletionsAdapter
from .adapters.openai_responses import OpenAIResponsesAdapter
from .base import LLMAdapter
from .errors import LLMError

# 协议注册表：api_type 字符串 → 适配器类。
PROTOCOLS: dict[str, type[LLMAdapter]] = {
    "completions": OpenAICompletionsAdapter,
    "responses": OpenAIResponsesAdapter,
}


def supported_protocols() -> list[str]:
    """当前构建支持的协议标识（注册表顺序，稳定）。"""
    return list(PROTOCOLS)


def create_llm_handler(api_type: str = "completions", **kwargs) -> LLMAdapter:
    """工厂：按 api_type 构造适配器。

    kwargs 与原 LLMHandler.__init__ 参数一致：
    model / api_key / api_base / timeout / max_retries / max_tokens /
    temperature / reasoning_effort（api_type 由第一参数承担）。
    """
    adapter_cls = PROTOCOLS.get(api_type)
    if adapter_cls is None:
        raise LLMError(
            f"unknown api_type {api_type!r}; supported: {supported_protocols()}",
            "INVALID_API_TYPE",
        )
    return adapter_cls(**kwargs)
