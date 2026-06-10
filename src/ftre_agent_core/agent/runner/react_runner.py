"""
ReActRunner - 异步 ReAct 执行引擎
"""
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator, TYPE_CHECKING

from ftre_agent_core.tool import CancellationToken, ToolCancelledError
from ftre_agent_core.llm import LLMHandler, LLMError, LLMResponse, StreamDelta
from .tool_handler import ToolHandler
from ..event import (
    DoneReason,
    AgentEvent,
    message_event,
    reasoning_event,
    reasoning_complete_event,
    message_complete_event,
    done_event,
    usage_update_event,
    error_event,
    retry_event,
    tool_call_streaming_event,
    tool_result_event,
)

if TYPE_CHECKING:
    from ..react import ReActAgent

logger = logging.getLogger(__name__)


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
    runtime_context: dict = field(default_factory=dict)

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

    def next_iteration(self) -> None:
        self.iteration += 1

    def complete(self) -> None:
        self.status = RunStatus.COMPLETED

    def fail(self, error: str) -> None:
        self.status = RunStatus.ERROR
        self.error = error

    def cancel(self) -> None:
        if not self.is_running:
            return
        self.status = RunStatus.CANCELLED
        self.cancel_token.cancel("user_cancelled")

    def check_cancel(self) -> None:
        try:
            self.cancel_token.raise_if_cancelled()
        except ToolCancelledError as exc:
            raise CancelledError(str(exc)) from exc


class ReActRunner:

    def __init__(self, agent: "ReActAgent"):
        self.agent = agent
        self.state = RunState()
        self.llm = LLMHandler(agent.model, agent.api_key, agent.api_base, agent.api_type)
        self.tool_handler = ToolHandler(agent.tools)

    async def run(
        self, message, runtime_context: dict | None = None
    ) -> AsyncGenerator[AgentEvent, None]:
        self.state.start()
        self.state.runtime_context = runtime_context or {}

        if isinstance(message, str):
            self.agent.memory.add_user(message)
        else:
            for msg in message:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    continue
                self.agent.memory.add_raw(msg)

        async for event in self._loop():
            yield event

    def cancel(self) -> None:
        self.state.cancel()
        self.llm.cancel()

    async def _loop(self) -> AsyncGenerator[AgentEvent, None]:
        try:
            max_iter = self.agent.max_iterations
            iteration = 0
            while max_iter is None or iteration < max_iter:
                self.state.check_cancel()
                self.state.next_iteration()
                iteration += 1
                async for event in self._step():
                    yield event
                if self.state.is_done:
                    return

            yield done_event(success=False, reason=DoneReason.MAX_ITERATIONS)
            self.state.complete()

        except CancelledError:
            yield done_event(success=False, reason=DoneReason.CANCELLED)

    UNRETRYABLE_ERROR_CODES = {"auth_error", "bad_request", "content_filter"}

    def _is_retryable(self, err: LLMError) -> bool:
        return err.code not in self.UNRETRYABLE_ERROR_CODES

    async def _step(self) -> AsyncGenerator[AgentEvent, None]:
        messages = self.agent.memory.get_messages()
        tools = self.agent.tools.to_openai_tools() or None

        max_attempts = 1 + self.agent.max_retries
        last_err: LLMError | None = None

        for attempt in range(max_attempts):
            full_content = ""
            full_reasoning = ""

            try:
                async for item in self.llm.stream(messages, tools):
                    self.state.check_cancel()

                    if isinstance(item, LLMResponse):
                        if item.usage:
                            yield usage_update_event(item.usage)
                        if full_reasoning:
                            yield reasoning_complete_event(content=full_reasoning)
                        if full_content:
                            yield message_complete_event(content=full_content)
                        async for event in self._handle_tool_calls(item):
                            yield event
                        return

                    elif isinstance(item, StreamDelta):
                        if item.content:
                            full_content += item.content
                            yield message_event(content=item.content)
                        if item.reasoning:
                            full_reasoning += item.reasoning
                            yield reasoning_event(content=item.reasoning)
                        if item.tool_calls:
                            yield tool_call_streaming_event(item.tool_calls)
                        if item.usage:
                            yield usage_update_event(item.usage)

                if self.state.is_cancelled:
                    raise CancelledError()

                if full_content:
                    self.agent.memory.add_assistant(full_content, reasoning=full_reasoning or None)
                    if full_reasoning:
                        yield reasoning_complete_event(content=full_reasoning)
                    yield message_complete_event(content=full_content)
                yield done_event(success=True, reason=DoneReason.COMPLETED)
                self.state.complete()
                return

            except CancelledError:
                if full_content:
                    self.agent.memory.add_assistant(full_content, reasoning=full_reasoning or None)
                    if full_reasoning:
                        yield reasoning_complete_event(content=full_reasoning)
                    yield message_complete_event(content=full_content)
                raise

            except Exception as e:
                if self.state.is_cancelled:
                    if full_content:
                        self.agent.memory.add_assistant(full_content, reasoning=full_reasoning or None)
                        if full_reasoning:
                            yield reasoning_complete_event(content=full_reasoning)
                        yield message_complete_event(content=full_content)
                    raise CancelledError() from e

                err = LLMError.classify(e)
                last_err = err
                is_last_attempt = (attempt >= max_attempts - 1)

                if not self._is_retryable(err) or is_last_attempt:
                    logger.warning(f"LLM 调用失败: [{err.code}] {err.message}")
                    yield error_event(message=err.message, code=err.code)
                    yield done_event(success=False, reason=DoneReason.ERROR)
                    self.state.fail(f"[{err.code}] {err.message}")
                    return

                logger.info(f"LLM 调用可重试错误 [{err.code}]，第 {attempt + 1}/{max_attempts - 1} 次重试")
                yield retry_event(
                    code=err.code,
                    message=err.message,
                    attempt=attempt + 1,
                    max_attempts=max_attempts - 1,
                )

                await asyncio.sleep(self.agent.retry_delay)
                if self.state.cancel_token.is_cancelled():
                    raise CancelledError()

    async def _handle_tool_calls(self, response: LLMResponse) -> AsyncGenerator[AgentEvent, None]:
        parsed: list[tuple[str, str, dict | None]] = [
            self.tool_handler.parse_tool_call(tc) for tc in response.tool_calls
        ]

        self.agent.memory.add_raw(
            self.tool_handler.build_assistant_message(response),
            usage=response.usage,
        )

        for call_id, name, args in parsed:
            if args is None:
                error_msg = "[PARSE_ERROR] Tool call JSON truncated or malformed. Please retry."
                yield tool_result_event(id=call_id, name=name, result=error_msg, error=error_msg, status="failed")
                self.agent.memory.add_tool_result(call_id, error_msg)

        valid = [(cid, name, args) for cid, name, args in parsed if args is not None]
        if not valid:
            return

        cancelled = False
        async for event in self.tool_handler.execute(valid, self.state):
            yield event
            if event["type"].value == "tool_result":
                self.agent.memory.add_tool_result(event["data"]["id"], event["data"]["result"])
                if event["data"].get("status") == "cancelled":
                    cancelled = True

        if cancelled:
            raise CancelledError()
