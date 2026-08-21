"""ftre_agent_core.llm —— LLM 适配层公共入口（PRD-B2）。

StreamChunk 协议（DSH 形态）+ BlockAssembler + LLMAdapter 契约 +
协议注册表工厂。新协议接入 SOP 见 docs/prd/PRD-B2-llm-adapter.md 3.4 节。
"""

from .adapters.openai_completions import OpenAICompletionsAdapter
from .adapters.openai_responses import OpenAIResponsesAdapter
from .base import LLMAdapter, OpenAIAdapterBase
from .block_assembler import BlockAssembler
from .errors import LLMError
from .events import (
    BlockEnd,
    BlockStart,
    FinishChunk,
    FinishReason,
    LlmFailure,
    ReasoningDeltaChunk,
    StreamChunk,
    TextDeltaChunk,
    ToolCall,
    ToolCallDeltaChunk,
    UsageChunk,
)
from .registry import PROTOCOLS, create_llm_handler, supported_protocols

__all__ = [
    "PROTOCOLS",
    # 组装器与错误
    "BlockAssembler",
    "BlockEnd",
    "BlockStart",
    "FinishChunk",
    "FinishReason",
    # 契约与工厂
    "LLMAdapter",
    "LLMError",
    "LlmFailure",
    "OpenAIAdapterBase",
    "OpenAICompletionsAdapter",
    "OpenAIResponsesAdapter",
    "ReasoningDeltaChunk",
    # StreamChunk 家族
    "StreamChunk",
    "TextDeltaChunk",
    "ToolCall",
    "ToolCallDeltaChunk",
    "UsageChunk",
    "create_llm_handler",
    "supported_protocols",
]
