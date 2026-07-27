"""Msg 实体 + append_event 重建引擎（层次 B）。

对齐 AgentScope ``message/_base.py`` 的 Msg + append_event，适配 ftre:
  - pydantic v2 BaseModel（统一技术栈）
  - ToolCallBlock.arguments 是 dict（非 AgentScope 的 str），delta 用内部缓冲
  - error 用 dict 占位（AgentScope ErrorInfo 未引入）
  - 暂不实现人工介入 4 个 case（无权限系统）

append_event 照搬 AgentScope 映射规则，用字符串 match event.type 避免循环 import。
"""
from __future__ import annotations

import base64
import json
import logging
import uuid
from datetime import datetime
from typing import Annotated, Any, Literal, Sequence, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from ._block import (
    Base64Source,
    ContentBlock,
    DataBlock,
    HintBlock,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolCallState,
    ToolResultBlock,
    ToolResultState,
)
from ..types import ReplyFinishedReason

# AgentStreamEvent 仅类型注解用（TYPE_CHECKING），运行时注解字符串化不求值，
# 避免顶层 import event 导致循环（event 依赖 message 的 Block）
if TYPE_CHECKING:
    from ..event import AgentStreamEvent

logger = logging.getLogger(__name__)


def _gen_id() -> str:
    return uuid.uuid4().hex[:16]


def _now_iso() -> str:
    return datetime.now().isoformat()


def _to_blocks(content: str | list) -> list:
    """字符串 content 自动包装为单元素 TextBlock 列表；空字符串返回空列表。"""
    if isinstance(content, str):
        return [TextBlock(text=content)] if content else []
    return list(content)


# ══════════════════════════════════════════════════════════════════
# Usage
# ══════════════════════════════════════════════════════════════════

class Usage(BaseModel):
    """token 用量统计。"""
    input_tokens: int
    output_tokens: int


# ══════════════════════════════════════════════════════════════════
# Msg
# ══════════════════════════════════════════════════════════════════

class Msg(BaseModel):
    """消息实体 —— 事件流的重建目标，AgentScope 协议的"快照视图"。

    一次 reply_stream 产出的所有事件（共享 reply_id）经 append_event 增量重建为本实例。
    """
    model_config = ConfigDict(use_enum_values=True)

    # ── 进 context 的字段 ──
    name: str
    content: list[Annotated[ContentBlock, Field(discriminator="type")]] = Field(default_factory=list)
    role: Literal["user", "assistant", "system"]
    id: str = Field(default_factory=_gen_id)

    # ── 元数据 ──
    metadata: dict = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now_iso)
    usage: Usage | None = Field(default=None)

    # ── 工作流控制 ──
    finished_at: str | None = Field(default=None)
    finished_reason: ReplyFinishedReason | None = Field(default=None)
    structured_output: dict | None = Field(default=None)
    error: dict[str, Any] | None = Field(default=None)

    # ── 内部缓冲：ToolCall delta 累积（str → END 时 parse 为 dict）──
    _tool_call_input_buf: dict[str, str] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _validate_role_content(self) -> "Msg":
        """角色约束（对齐 AgentScope）。"""
        for block in self.content:
            if self.role == "user" and block.type not in ("text", "data"):
                raise ValueError("User message can only contain text/data blocks.")
            if self.role == "system" and block.type != "text":
                raise ValueError("System message can only contain text blocks.")
        return self

    # ── 内容访问辅助 ──

    def _find_block(self, block_type: str, block_id: str) -> ContentBlock | None:
        """按 type + id 查找块（对齐 AgentScope）。"""
        for block in self.content:
            if block.type == block_type and block.id == block_id:
                return block
        return None

    def has_content_blocks(self, block_type: str | list[str] | None = None) -> bool:
        if block_type is None:
            return len(self.content) > 0
        typs = [block_type] if isinstance(block_type, str) else block_type
        return any(b.type in typs for b in self.content)

    def get_text_content(self, separator: str = "\n") -> str | None:
        gathered = [b.text for b in self.content if b.type == "text"]
        return separator.join(gathered) if gathered else None

    def get_content_blocks(
        self, block_type: str | list[str] | None = None
    ) -> Sequence[ContentBlock]:
        blocks = self.content or []
        if isinstance(block_type, str):
            return [b for b in blocks if b.type == block_type]
        if isinstance(block_type, list):
            return [b for b in blocks if b.type in block_type]
        return blocks

    # ── 核心：append_event 重建引擎 ──

    def append_event(self, event: AgentStreamEvent) -> "Msg":
        """把一个流式事件增量应用到 Msg（对齐 AgentScope append_event）。

        映射规则见 Obsidian「Msg与append_event设计.md」。
        暂不处理人工介入事件（REQUIRE_USER_CONFIRM 等），收到时静默跳过。
        """
        # reply_id 校验
        reply_id = getattr(event, "reply_id", None)
        if reply_id is not None and reply_id != self.id:
            logger.warning(
                "Event %s reply_id %r != msg id %r, skipping.",
                event.__class__.__name__, reply_id, self.id,
            )
            return self

        et = event.type

        # ── 生命周期 ──
        if et == "REPLY_END":
            self.finished_at = event.created_at
            self.finished_reason = ReplyFinishedReason(event.finished_reason)
            self.error = event.error

        elif et == "MODEL_CALL_END":
            if self.usage is None:
                self.usage = Usage(
                    input_tokens=event.input_tokens,
                    output_tokens=event.output_tokens,
                )
            else:
                self.usage.input_tokens += event.input_tokens
                self.usage.output_tokens += event.output_tokens

        # ── 文本块三段式 ──
        elif et == "TEXT_BLOCK_START":
            self.content.append(TextBlock(id=event.block_id, text=""))

        elif et == "TEXT_BLOCK_DELTA":
            block = self._find_block("text", event.block_id)
            if block is not None:
                block.text += event.delta
            else:
                logger.warning("TextBlock %r not found, skipping.", event.block_id)

        elif et == "TEXT_BLOCK_END":
            block = self._find_block("text", event.block_id)
            if block is not None:
                block.finished_at = event.created_at

        # ── 思考块三段式 ──
        elif et == "THINKING_BLOCK_START":
            self.content.append(ThinkingBlock(id=event.block_id, thinking=""))

        elif et == "THINKING_BLOCK_DELTA":
            block = self._find_block("thinking", event.block_id)
            if block is not None:
                block.thinking += event.delta
            else:
                logger.warning("ThinkingBlock %r not found, skipping.", event.block_id)

        elif et == "THINKING_BLOCK_END":
            block = self._find_block("thinking", event.block_id)
            if block is not None:
                block.finished_at = event.created_at

        # ── 数据块三段式 ──
        elif et == "DATA_BLOCK_START":
            self.content.append(
                DataBlock(
                    id=event.block_id,
                    source=Base64Source(data="", media_type=event.media_type),
                )
            )

        elif et == "DATA_BLOCK_DELTA":
            block = self._find_block("data", event.block_id)
            if block is None:
                logger.warning("DataBlock %r not found, skipping.", event.block_id)
            elif event.data:
                # 每个 delta 是独立 base64 chunk（自带 padding），必须
                # decode→拼接 bytes→encode，否则字节流损坏
                existing = (
                    base64.b64decode(block.source.data)
                    if block.source.data else b""
                )
                incoming = base64.b64decode(event.data)
                block.source.data = base64.b64encode(
                    existing + incoming
                ).decode("ascii")

        elif et == "DATA_BLOCK_END":
            block = self._find_block("data", event.block_id)
            if block is not None:
                block.finished_at = event.created_at

        # ── 提示块（一次性）──
        elif et == "HINT_BLOCK":
            hint_block = HintBlock(
                id=event.block_id,
                source=event.source,
                hint=event.hint,
            )
            hint_block.finished_at = hint_block.created_at
            self.content.append(hint_block)

        # ── 工具调用三段式（ftre 适配：delta 缓冲 → END parse）──
        elif et == "TOOL_CALL_START":
            self.content.append(
                ToolCallBlock(
                    id=event.tool_call_id,
                    name=event.tool_call_name,
                    arguments={},
                )
            )
            self._tool_call_input_buf[event.tool_call_id] = ""

        elif et == "TOOL_CALL_DELTA":
            # 累积 delta 字符串到缓冲（ftre arguments 是 dict，不能直接 +=）
            buf = self._tool_call_input_buf.get(event.tool_call_id, "")
            self._tool_call_input_buf[event.tool_call_id] = buf + event.delta

        elif et == "TOOL_CALL_END":
            block = self._find_block("tool_call", event.tool_call_id)
            if block is not None:
                raw = self._tool_call_input_buf.pop(event.tool_call_id, "")
                try:
                    block.arguments = json.loads(raw) if raw else {}
                except (json.JSONDecodeError, TypeError):
                    logger.warning("ToolCall %r args parse failed: %r",
                                   event.tool_call_id, raw)
                    block.arguments = {}
                block.finished_at = event.created_at

        # ── 工具结果三段式 ──
        elif et == "TOOL_RESULT_START":
            self.content.append(
                ToolResultBlock(
                    id=event.tool_call_id,
                    name=event.tool_call_name,
                    output=[],
                    state=ToolResultState.RUNNING,
                )
            )

        elif et == "TOOL_RESULT_TEXT_DELTA":
            block = self._find_block("tool_result", event.tool_call_id)
            if block is None:
                logger.warning("ToolResultBlock %r not found, skipping.",
                               event.tool_call_id)
            else:
                # output 可能是 str（初始）或 list，统一为 list
                if isinstance(block.output, str):
                    block.output = [TextBlock(text=block.output)]
                # 连续 TextDelta 合并到同一个 TextBlock（避免每 delta 一块）
                if not block.output or block.output[-1].type != "text":
                    block.output.append(TextBlock(text=event.delta))
                else:
                    block.output[-1].text += event.delta

        elif et == "TOOL_RESULT_DATA_DELTA":
            block = self._find_block("tool_result", event.tool_call_id)
            if block is None:
                logger.warning("ToolResultBlock %r not found, skipping.",
                               event.tool_call_id)
            else:
                if isinstance(block.output, str):
                    block.output = [TextBlock(text=block.output)]
                src = (
                    Base64Source(data=event.data, media_type=event.media_type)
                    if event.data is not None
                    else __import__("ftre_agent_core.message._block",
                                    fromlist=["URLSource"]).URLSource(
                        url=str(event.url), media_type=event.media_type)
                )
                block.output.append(DataBlock(id=event.block_id, source=src))

        elif et == "TOOL_RESULT_END":
            block = self._find_block("tool_result", event.tool_call_id)
            if block is not None:
                block.state = event.state
                block.metadata = event.metadata
                block.finished_at = event.created_at
                # 配对 ToolCall 置 FINISHED（保证 SSE 重建与 agent 内部一致）
                call_block = self._find_block("tool_call", event.tool_call_id)
                if call_block is not None:
                    call_block.state = ToolCallState.FINISHED

        # ── 人工介入事件暂不处理（ftre 无权限系统），静默跳过 ──
        # REQUIRE_USER_CONFIRM / USER_CONFIRM_RESULT
        # REQUIRE_EXTERNAL_EXECUTION / EXTERNAL_EXECUTION_RESULT

        return self


# ══════════════════════════════════════════════════════════════════
# 工厂函数（对齐 AgentScope UserMsg/AssistantMsg/SystemMsg）
# ══════════════════════════════════════════════════════════════════

def UserMsg(name: str, content: str | list, **kwargs) -> Msg:
    """创建 user 消息（content str 自动包 TextBlock）。"""
    return Msg(name=name, content=_to_blocks(content), role="user", **kwargs)


def AssistantMsg(name: str, content: str | list = "", **kwargs) -> Msg:
    """创建 assistant 消息（content str 自动包 TextBlock，默认空）。"""
    return Msg(name=name, content=_to_blocks(content), role="assistant", **kwargs)


def SystemMsg(name: str, content: str | list, **kwargs) -> Msg:
    """创建 system 消息（content str 自动包 TextBlock）。"""
    return Msg(name=name, content=_to_blocks(content), role="system", **kwargs)
