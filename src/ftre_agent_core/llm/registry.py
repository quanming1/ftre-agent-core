"""
适配器注册表 + 统一 client 工厂

唯一的对外入口：create_client(api_key, base_url, api_type)
"""
from typing import Callable
from openai import OpenAI

# api_type → factory(api_key, base_url) -> client
ADAPTER_REGISTRY: dict[str, Callable[[str, str], object]] = {}


def register_adapter(api_type: str):
    """装饰器：注册 api_type → client 工厂"""
    def decorator(factory_fn):
        ADAPTER_REGISTRY[api_type] = factory_fn
        return factory_fn
    return decorator


def create_client(api_key: str, base_url: str, api_type: str = "completions"):
    """
    统一 client 工厂。

    completions 走原生 OpenAI SDK，其他走注册的适配器。
    未注册的 api_type fallback 到 OpenAI SDK。
    """
    if api_type == "completions" or api_type not in ADAPTER_REGISTRY:
        return OpenAI(api_key=api_key, base_url=base_url)
    return ADAPTER_REGISTRY[api_type](api_key, base_url)
