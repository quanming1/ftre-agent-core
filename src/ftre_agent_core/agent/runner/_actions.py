"""ReAct 动作模型与执行器间数据载体。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ...llm import ToolCall, LLMError
from ...types import ReplyFinishedReason


class Reasoning(BaseModel):
    """下一步：调用大模型进行推理。"""
    hint: str | None = None
    tool_choice: str | None = None
    force_no_tools: bool = False


class Acting(BaseModel):
    """下一步：执行模型产生的工具调用。"""
    tool_calls: list[ToolCall]


class Exit(BaseModel):
    """下一步：结束（或暂停）当前回复。"""
    finished_reason: ReplyFinishedReason
    exit_msg: Any | None = None
    error: str | None = None
    error_code: str | None = None


@dataclass
class TurnResult:
    """一轮 LLM 推理的结构化产物，供 _decide() 消费。"""
    text: str
    reasoning: str
    tool_calls: list[ToolCall]
    finish_reason: str
    usage: dict | None = None
    error: LLMError | None = None


@dataclass
class ExitOutcome:
    """Exit 执行结果，可能让主循环继续而非退出。"""
    should_continue: bool = False
    continue_hint: str | None = None
