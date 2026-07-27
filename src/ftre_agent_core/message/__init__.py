"""类型化消息内容块 —— AgentScope 消息协议的 ftre 适配版（层次 A）。

层次 A 只提供 Block 类型 + 状态机 + OpenAI dict 转换器，
不引入 Msg / append_event（层次 B）。
"""
from ._block import (
    # 状态机
    ToolCallState,
    ToolResultState,
    # 数据源
    Base64Source,
    URLSource,
    # 6 种内容块
    TextBlock,
    ThinkingBlock,
    DataBlock,
    HintBlock,
    ToolCallBlock,
    ToolResultBlock,
    # 类型别名
    ContentBlock,
    ContentBlockTypes,
)
from ._convert import (
    from_openai_part,
    to_openai_part,
    from_openai_message,
    to_openai_message,
)

__all__ = [
    "ToolCallState",
    "ToolResultState",
    "Base64Source",
    "URLSource",
    "TextBlock",
    "ThinkingBlock",
    "DataBlock",
    "HintBlock",
    "ToolCallBlock",
    "ToolResultBlock",
    "ContentBlock",
    "ContentBlockTypes",
    "from_openai_part",
    "to_openai_part",
    "from_openai_message",
    "to_openai_message",
]
