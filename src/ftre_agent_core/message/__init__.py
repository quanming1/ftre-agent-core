"""类型化消息内容块 + Msg 实体 —— AgentScope 消息协议的 ftre 适配版。

层次 A: ContentBlock + 状态机 + OpenAI dict 转换器
层次 B: Msg 实体 + append_event 重建引擎 + 工厂函数
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
from ._msg import (
    Usage,
    Msg,
    UserMsg,
    AssistantMsg,
    SystemMsg,
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
    # 层次 B
    "Usage",
    "Msg",
    "UserMsg",
    "AssistantMsg",
    "SystemMsg",
]
