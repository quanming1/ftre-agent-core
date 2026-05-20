"""
ReActRunner - ReAct 执行引擎

职责：
- run()    → 启动 ReAct 循环
- cancel() → 用户取消
- _loop()  → 主循环（CancelledError 唯一捕获点）
- _step()  → 单次迭代（LLM 调用 → 处理响应）
"""
import logging
from typing import Generator, TYPE_CHECKING
from ftre_agent_core.tool.builtins import BUILTIN_TOOL_FACTORIES
from .state import RunState, CancelledError
from .handler import LLMHandler, LLMResponse, LLMError, StreamDelta, ToolHandler
from ..event import (
    DoneReason,
    AgentEvent,
    message_event,
    reasoning_event,
    message_complete_event,
    done_event,
    usage_update_event,
    error_event,
    tool_call_streaming_event,
    tool_result_event,
)

if TYPE_CHECKING:
    from ..react import ReActAgent

logger = logging.getLogger(__name__)


class ReActRunner:
    """
    ReAct 执行引擎

    编排 LLMHandler / ToolHandler 完成 ReAct 循环。
    """

    def __init__(self, agent: "ReActAgent"):
        self.agent = agent
        self.state = RunState()
        self.llm = LLMHandler(agent.model, agent.api_key, agent.api_base, agent.api_type)
        self.tool_handler = ToolHandler(agent.tools)

        # 注入内置工具
        for factory in BUILTIN_TOOL_FACTORIES:
            self.agent.tools.register(factory())

    # ============================================================
    # 对外 API
    # ============================================================

    def run(self, message) -> Generator[AgentEvent, None, None]:
        """启动 ReAct 循环。message: str 或 list[dict]"""
        self.state.start()

        if isinstance(message, str):
            self.agent.memory.add_user(message)
        else:
            for msg in message:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    continue
                self.agent.memory.add_raw(msg)

        yield from self._loop()

    def cancel(self, timeout: float | None = None) -> bool:
        """用户取消，阻塞等待善后完成。"""
        if not self.state.is_running:
            return True
        self.state.cancel()
        self.llm.cancel()
        return self.state.wait_done(timeout)

    # ============================================================
    # 主循环
    # ============================================================

    def _loop(self) -> Generator[AgentEvent, None, None]:
        """ReAct 主循环。CancelledError 在此统一捕获。"""
        try:
            for _ in range(self.agent.max_iterations):
                self.state.check_cancel()
                self.state.next_iteration()

                yield from self._step()

                if self.state.is_done:
                    return

            # 达到最大迭代次数
            yield done_event(success=False, reason=DoneReason.MAX_ITERATIONS)
            self.state.complete()

        except CancelledError:
            yield done_event(success=False, reason=DoneReason.CANCELLED)
            self.state.mark_done()

    # ============================================================
    # 单次迭代
    # ============================================================

    def _step(self) -> Generator[AgentEvent, None, None]:
        """单次迭代：调 LLM → 处理响应。"""
        messages = self.agent.memory.get_messages()
        tools = self.agent.tools.to_openai_tools() or None
        full_content = ""

        try:
            for item in self.llm.stream(messages, tools):
                self.state.check_cancel()

                if isinstance(item, LLMResponse):
                    if item.usage:
                        yield usage_update_event(item.usage)
                    if full_content:
                        yield message_complete_event(content=full_content)
                    yield from self._handle_tool_calls(item)
                    return

                elif isinstance(item, StreamDelta):
                    if item.content:
                        full_content += item.content
                        yield message_event(content=item.content)
                    if item.reasoning:
                        yield reasoning_event(content=item.reasoning)
                    if item.tool_calls:
                        yield tool_call_streaming_event(item.tool_calls)
                    if item.usage:
                        yield usage_update_event(item.usage)

            # 取消导致的 break
            if self.state.is_cancelled:
                raise CancelledError()

            # 纯文本回复 → 完成
            if full_content:
                self.agent.memory.add_assistant(full_content)
                yield message_complete_event(content=full_content)
            yield done_event(success=True, reason=DoneReason.COMPLETED)
            self.state.complete()

        except CancelledError:
            if full_content:
                self.agent.memory.add_assistant(full_content)
                yield message_complete_event(content=full_content)
            raise

        except Exception as e:
            if self.state.is_cancelled:
                if full_content:
                    self.agent.memory.add_assistant(full_content)
                    yield message_complete_event(content=full_content)
                raise CancelledError() from e

            err = LLMError.classify(e)
            logger.warning(f"LLM 调用失败: [{err.code}] {err.message}")
            yield error_event(message=err.message, code=err.code)
            yield done_event(success=False, reason=DoneReason.ERROR)
            self.state.fail(f"[{err.code}] {err.message}")

    # ============================================================
    # 工具调用处理
    # ============================================================

    def _handle_tool_calls(self, response: LLMResponse) -> Generator[AgentEvent, None, None]:
        """处理 LLM 返回的工具调用。"""
        tool_calls = response.tool_calls

        # 解析
        parsed: list[tuple[str, str, dict | None]] = [
            self.tool_handler.parse_tool_call(tc) for tc in tool_calls
        ]

        # 写入 assistant message
        self.agent.memory.add_raw(
            self.tool_handler.build_assistant_message(response),
            usage=response.usage,
        )

        # 处理解析失败的
        failed = [(cid, name) for cid, name, args in parsed if args is None]
        for call_id, name in failed:
            error_msg = "[PARSE_ERROR] Tool call JSON truncated or malformed. Please retry."
            yield tool_result_event(id=call_id, name=name, result=error_msg, error=error_msg, status="failed")
            self.agent.memory.add_tool_result(call_id, error_msg)

        # 过滤出可执行的
        valid = [(cid, name, args) for cid, name, args in parsed if args is not None]
        if not valid:
            return

        # 执行
        for event in self.tool_handler.execute(valid, self.state):
            yield event
            if event["type"].value == "tool_result":
                self.agent.memory.add_tool_result(event["data"]["id"], event["data"]["result"])
                if event["data"].get("status") == "cancelled":
                    raise CancelledError()
