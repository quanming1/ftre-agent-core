"""
LLM 客户端层

协议适配的统一入口。上层只需调 create_client() 拿到兼容 OpenAI SDK 的 client。

用法：
    from packages.core.llm import create_client

    client = create_client(api_key, base_url, api_type="completions")
    # client.chat.completions.create(...) — 所有协议统一接口
"""
from .registry import create_client, register_adapter, ADAPTER_REGISTRY

# 导入适配器包，触发所有 @register_adapter 注册
from . import adapters  # noqa: F401

__all__ = ["create_client", "register_adapter", "ADAPTER_REGISTRY"]
