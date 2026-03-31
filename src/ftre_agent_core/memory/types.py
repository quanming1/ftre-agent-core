"""
Memory 类型定义
"""
from typing import TypedDict


class MemoryOptions(TypedDict, total=False):
    """MemoryManager 配置"""
    max_messages: int      # 最大消息数量
    system_prompt: str     # 系统提示词
