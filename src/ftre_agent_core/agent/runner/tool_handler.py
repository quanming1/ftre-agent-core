"""
工具执行器和 assistant 消息构造器。

本模块职责：
  - run_one(): 执行单个工具调用，返回 ToolResult。
    react_runner._stream_turn() 会用 asyncio.create_task() 调它，
    因此不会阻塞 LLM 流的消费。
  - build_assistant_message_from_tool_calls(): 根据 LLMHandler 组装好的
    ToolCall 对象，构造要写入 memory 的 assistant 原始消息。
  - 执行工具中间件 before / after 链。

本模块不负责：
  - 决定何时向调用方 yield 事件。
  - 写入 memory。
  - 协调多个工具的并发、取消和批量结果补齐。
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ftre_agent_core.llm import ToolCall
from ftre_agent_core.tool import ToolRegistry
from ftre_agent_core.tool.registry import ToolContext

if TYPE_CHECKING:
    from .react_runner import RunState

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """单个工具调用的执行结果。"""
    call_id: str
    name: str
    result: str
    error: str | None = None
    status: str = "completed"   # completed / failed / cancelled
    metadata: dict = field(default_factory=dict)

    @property
    def cancelled(self) -> bool:
        return self.status == "cancelled"


class ToolHandler:

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    # 执行单个工具调用。
    async def run_one(
        self,
        call_id: str,
        name: str,
        arguments: dict,
        state: "RunState",
        parse_failed: bool = False,
    ) -> ToolResult:
        """执行一个工具调用并返回 ToolResult。

        这个方法预期由 asyncio.create_task() 调用，这样多个工具可以并发执行。
        parse_failed=True 表示模型输出的工具参数 JSON 格式错误，直接返回失败结果。
        """
        if parse_failed:
            return ToolResult(
                call_id=call_id,
                name=name,
                result="[PARSE_ERROR] Tool call arguments were malformed JSON.",
                error="malformed JSON arguments",
                status="failed",
            )

        ctx = ToolContext(call_id=call_id, name=name, arguments=arguments)
        ctx.cancel_token = state.cancel_token
        ctx.metadata["runtime_context"] = state.runtime_context

        # 执行 before 中间件链。
        ctx = self._run_before(ctx)

        if ctx.skipped:
            result = ToolResult(call_id=call_id, name=name, result=ctx.skip_result)
            return self._run_after(ctx, result)

        try:
            tool = self.registry.get(name)
            if tool is not None and tool.is_async():
                # 异步工具必须直接 await 底层协程函数，不能走 Tool.execute()。
                # Tool.execute() 面向同步调用方，会使用 asyncio.run()。
                raw = await tool._get_callable()(**ctx.arguments)
            else:
                raw = await asyncio.to_thread(
                    self.registry.execute,
                    name,
                    runtime_context=ctx.metadata.get("runtime_context"),
                    **ctx.arguments,
                )
            result = ToolResult(call_id=call_id, name=name, result=str(raw))
        except asyncio.CancelledError:
            result = ToolResult(
                call_id=call_id, name=name,
                result="[CANCELLED] Tool execution was cancelled.",
                status="cancelled",
            )
        except Exception as exc:
            logger.warning("[tool] %s failed: %s", name, exc)
            result = ToolResult(
                call_id=call_id, name=name,
                result=str(exc), error=str(exc), status="failed",
            )

        return self._run_after(ctx, result)

    # 构造带 tool_calls 的 assistant 消息。
    @staticmethod
    def build_assistant_message_from_tool_calls(
        tool_calls: list[ToolCall],
        content: str | None = None,
        reasoning: str | None = None,
    ) -> dict:
        """构造要写入 memory.add_raw() 的 assistant 原始消息。

        assistant 消息只包含 tool_calls；对应的 role="tool" 结果消息由
        react_runner 在所有工具完成后统一写入。
        """
        msg: dict = {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.input, ensure_ascii=False)
                        if tc.input is not None else "{}",
                    },
                }
                for tc in tool_calls
            ],
        }
        if reasoning:
            msg["reasoning_content"] = reasoning
        return msg

    # 工具中间件
    def _run_before(self, ctx: ToolContext) -> ToolContext:
        for mw in self.registry.middlewares:
            ctx = mw.before(ctx)
            if ctx.skipped:
                break
        return ctx

    def _run_after(self, ctx: ToolContext, result: ToolResult) -> ToolResult:
        for mw in reversed(self.registry.middlewares):
            result = mw.after(ctx, result)
        return result
