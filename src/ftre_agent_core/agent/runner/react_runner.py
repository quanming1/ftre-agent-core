"""
ReActRunner - ReAct 执行引擎
"""
import logging
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Generator, TYPE_CHECKING

from ftre_agent_core.tool import CancellationToken, ToolCancelledError
from ftre_agent_core.llm import LLMHandler, LLMResponse, LLMError, StreamDelta
from .tool_handler import ToolHandler
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


# ============================================================
# 状态
# ============================================================

class RunStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class CancelledError(Exception):
    pass


@dataclass
class RunState:
    status: RunStatus = RunStatus.IDLE
    iteration: int = 0
    error: str | None = None
    cancel_token: CancellationToken = field(default_factory=CancellationToken)
    _done_event: threading.Event = field(default_factory=threading.Event)

    @property
    def is_running(self) -> bool:
        return self.status == RunStatus.RUNNING

    @property
    def is_cancelled(self) -> bool:
        return self.status == RunStatus.CANCELLED

    @property
    def is_done(self) -> bool:
        return self.status in (RunStatus.COMPLETED, RunStatus.ERROR, RunStatus.CANCELLED)

    def start(self) -> None:
        self.status = RunStatus.RUNNING
        self.iteration = 0
        self.error = None
        self.cancel_token = CancellationToken()
        self._done_event.clear()

    def next_iteration(self) -> None:
        self.iteration += 1

    def complete(self) -> None:
        self.status = RunStatus.COMPLETED
        self._done_event.set()

    def fail(self, error: str) -> None:
        self.status = RunStatus.ERROR
        self.error = error
        self._done_event.set()

    def cancel(self) -> None:
        if not self.is_running:
            self._done_event.set()
            return
        self.status = RunStatus.CANCELLED
        self.cancel_token.cancel("user_cancelled")

    def mark_done(self) -> None:
        self._done_event.set()

    def wait_done(self, timeout: float | None = None) -> bool:
        return self._done_event.wait(timeout)

    def wait_or_cancelled(self, timeout: float) -> bool:
        return self.cancel_token.wait(timeout)

    def check_cancel(self) -> None:
        try:
            self.cancel_token.raise_if_cancelled()
        except ToolCancelledError as exc:
            raise CancelledError(str(exc)) from exc


# ============================================================
# Runner
# ============================================================

class ReActRunner:
    """ReAct 执行引擎"""

    def __init__(self, agent: "ReActAgent"):
        self.agent = agent
        self.state = RunState()
        self.llm = LLMHandler(agent.model, agent.api_key, agent.api_base, agent.api_type)
        self.tool_handler = ToolHandler(agent.tools)

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
        try:
            max_iter = self.agent.max_iterations
            iteration = 0
            while max_iter is None or iteration < max_iter:
                self.state.check_cancel()
                self.state.next_iteration()
                iteration += 1
                yield from self._step()
                if self.state.is_done:
                    return

            yield done_event(success=False, reason=DoneReason.MAX_ITERATIONS)
            self.state.complete()

        except CancelledError:
            yield done_event(success=False, reason=DoneReason.CANCELLED)
            self.state.mark_done()

    # ============================================================
    # 单次迭代
    # ============================================================

    def _step(self) -> Generator[AgentEvent, None, None]:
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

            if self.state.is_cancelled:
                raise CancelledError()

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
    # 工具调用
    # ============================================================

    def _handle_tool_calls(self, response: LLMResponse) -> Generator[AgentEvent, None, None]:
        parsed: list[tuple[str, str, dict | None]] = [
            self.tool_handler.parse_tool_call(tc) for tc in response.tool_calls
        ]

        self.agent.memory.add_raw(
            self.tool_handler.build_assistant_message(response),
            usage=response.usage,
        )

        # 解析失败的
        for call_id, name, args in parsed:
            if args is None:
                error_msg = "[PARSE_ERROR] Tool call JSON truncated or malformed. Please retry."
                yield tool_result_event(id=call_id, name=name, result=error_msg, error=error_msg, status="failed")
                self.agent.memory.add_tool_result(call_id, error_msg)

        # 可执行的
        valid = [(cid, name, args) for cid, name, args in parsed if args is not None]
        if not valid:
            return

        # 执行
        cancelled = False
        for event in self.tool_handler.execute(valid, self.state):
            yield event
            if event["type"].value == "tool_result":
                self.agent.memory.add_tool_result(event["data"]["id"], event["data"]["result"])
                if event["data"].get("status") == "cancelled":
                    cancelled = True

        if cancelled:
            raise CancelledError()
