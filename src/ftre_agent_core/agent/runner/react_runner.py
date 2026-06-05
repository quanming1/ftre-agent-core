"""
ReActRunner - ReAct 执行引擎

这个模块负责驱动整个 ReAct Agent 的执行流程，核心职责包括：
1. 维护一次运行的状态（运行中、完成、报错、取消等）
2. 驱动 LLM 的流式输出
3. 将流式增量转换为统一的 AgentEvent 事件
4. 处理模型返回的工具调用，并将工具结果写回 memory
5. 协调取消信号，让 LLM 调用和工具执行都能及时停止

从项目结构上看，ReActRunner 位于 ReActAgent 的执行层：
- Agent 负责组装配置、memory、tools 等能力
- Runner 负责真正"跑起来"
- LLMHandler 负责和模型通信
- ToolHandler 负责解析与执行工具调用
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


# ============================================================
# 状态
# ============================================================

class RunStatus(str, Enum):
    """一次 Runner 运行的生命周期状态。"""

    # 初始态，尚未开始执行
    IDLE = "idle"
    # 正在执行 ReAct 循环
    RUNNING = "running"
    # 正常完成
    COMPLETED = "completed"
    # 发生错误后结束
    ERROR = "error"
    # 被用户或外部信号取消
    CANCELLED = "cancelled"


class CancelledError(Exception):
    """内部使用的取消异常。

    底层取消信号来自 CancellationToken / ToolCancelledError，
    这里统一转成 Runner 自己的取消异常，方便主循环统一处理。
    """

    pass


@dataclass
class RunState:
    """保存一次运行过程中的可变状态。

    这个对象会在 Runner、ToolHandler 等组件之间共享，
    既保存运行状态，也保存取消令牌和运行时上下文。
    """

    # 当前运行状态
    status: RunStatus = RunStatus.IDLE
    # 当前已经执行到第几轮 ReAct 迭代
    iteration: int = 0
    # 错误信息，仅在失败时有值
    error: str | None = None
    # 取消令牌：主循环和工具执行都通过它感知取消
    cancel_token: CancellationToken = field(default_factory=CancellationToken)
    # 运行期上下文，供工具执行等环节读取
    runtime_context: dict = field(default_factory=dict)
    # 用于 cancel() 阻塞等待"完全结束"的同步事件
    _done_event: threading.Event = field(default_factory=threading.Event)

    @property
    def is_running(self) -> bool:
        """当前是否处于运行中。"""
        return self.status == RunStatus.RUNNING

    @property
    def is_cancelled(self) -> bool:
        """当前是否已标记为取消。"""
        return self.status == RunStatus.CANCELLED

    @property
    def is_done(self) -> bool:
        """当前是否已经结束。

        结束包括三种：成功、失败、取消。
        """
        return self.status in (RunStatus.COMPLETED, RunStatus.ERROR, RunStatus.CANCELLED)

    def start(self) -> None:
        """开始一次全新的运行。

        这里会重置迭代计数、错误信息、取消令牌和 done 事件。
        """
        self.status = RunStatus.RUNNING
        self.iteration = 0
        self.error = None
        self.cancel_token = CancellationToken()
        self._done_event.clear()

    def next_iteration(self) -> None:
        """进入下一轮 ReAct 迭代。"""
        self.iteration += 1

    def complete(self) -> None:
        """标记为正常完成，并唤醒等待方。"""
        self.status = RunStatus.COMPLETED
        self._done_event.set()

    def fail(self, error: str) -> None:
        """标记为失败，并记录错误信息。"""
        self.status = RunStatus.ERROR
        self.error = error
        self._done_event.set()

    def cancel(self) -> None:
        """请求取消当前运行。

        注意：这里只是发出取消信号，不代表已经完全停止。
        真正停止要等待主循环 / 工具执行自己感知到取消后收尾。
        """
        if not self.is_running:
            self._done_event.set()
            return
        self.status = RunStatus.CANCELLED
        self.cancel_token.cancel("user_cancelled")

    def mark_done(self) -> None:
        """仅标记 done 事件完成。

        主要用于取消路径：状态可能已经是 CANCELLED，
        此时不再改状态，只通知等待方"善后已结束"。
        """
        self._done_event.set()

    def wait_done(self, timeout: float | None = None) -> bool:
        """阻塞等待运行彻底结束。"""
        return self._done_event.wait(timeout)

    def wait_or_cancelled(self, timeout: float) -> bool:
        """等待一段时间，或在取消时提前返回。"""
        return self.cancel_token.wait(timeout)

    def check_cancel(self) -> None:
        """检查是否已取消，若已取消则抛出统一的取消异常。"""
        try:
            self.cancel_token.raise_if_cancelled()
        except ToolCancelledError as exc:
            raise CancelledError(str(exc)) from exc


# ============================================================
# Runner
# ============================================================

class ReActRunner:
    """ReAct 执行引擎。

    一个 ReActAgent 通常持有一个 Runner，Runner 负责：
    - 从 memory 读取消息历史
    - 调用 LLM 进行流式推理
    - 输出统一的事件流
    - 处理工具调用
    - 根据是否还有工具调用决定继续下一轮还是结束
    """

    def __init__(self, agent: "ReActAgent"):
        # 绑定所属 Agent，Runner 通过它访问 memory、tools、配置等
        self.agent = agent
        # 本次/最近一次运行状态
        self.state = RunState()
        # LLM 调用适配层
        self.llm = LLMHandler(agent.model, agent.api_key, agent.api_base, agent.api_type)
        # 工具调用处理器：负责解析和执行模型发出的 tool_calls
        self.tool_handler = ToolHandler(agent.tools)

    def run(self, message, runtime_context: dict | None = None) -> Generator[AgentEvent, None, None]:
        """启动 ReAct 循环。

        参数：
        - message: str 或 list[dict]
          - str：表示一条新的用户消息
          - list[dict]：表示一批已经结构化好的消息
        - runtime_context:
          提供给运行时使用的上下文信息，通常会被工具执行阶段消费

        返回：
        - 一个 Generator，持续产出 AgentEvent 事件
        """
        self.state.start()
        self.state.runtime_context = runtime_context or {}

        # 将本次输入写入 memory。
        # 如果是字符串，按 user message 处理；
        # 如果是消息列表，则逐条写入，但跳过 system 消息，
        # 因为 system prompt 一般由 Agent 自己统一管理。
        if isinstance(message, str):
            self.agent.memory.add_user(message)
        else:
            for msg in message:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    continue
                self.agent.memory.add_raw(msg)

        yield from self._loop()

    def cancel(self, timeout: float | None = None) -> bool:
        """用户取消，阻塞等待善后完成。

        取消分两步：
        1. 先给 RunState 发取消信号
        2. 再调用 LLMHandler.cancel()，尽可能中断流式请求

        返回值表示是否在 timeout 内等到了彻底结束。
        """
        if not self.state.is_running:
            return True
        self.state.cancel()
        self.llm.cancel()
        return self.state.wait_done(timeout)

    # ============================================================
    # 主循环
    # ============================================================

    def _loop(self) -> Generator[AgentEvent, None, None]:
        """ReAct 主循环。

        一轮 _step() 的语义是：
        - 让模型基于当前 memory 思考一次
        - 如果模型给出工具调用，则执行工具并把结果写回 memory
        - 然后主循环继续下一轮，让模型基于新的工具结果再次思考
        - 如果模型直接给出最终回答且没有工具调用，则结束
        """
        try:
            max_iter = self.agent.max_iterations
            iteration = 0
            while max_iter is None or iteration < max_iter:
                # 每轮开始前先检查取消，避免已取消后还继续往下跑
                self.state.check_cancel()
                self.state.next_iteration()
                iteration += 1
                yield from self._step()
                if self.state.is_done:
                    return

            # 超出最大迭代次数，防止无限循环
            yield done_event(success=False, reason=DoneReason.MAX_ITERATIONS)
            self.state.complete()

        except CancelledError:
            # 取消不是"错误"，这里发出取消结束事件即可
            yield done_event(success=False, reason=DoneReason.CANCELLED)
            self.state.mark_done()

    # ============================================================
    # 重试辅助
    # ============================================================

    # 不可重试的错误码（认证、请求无效、内容审核等，重试也没用）
    UNRETRYABLE_ERROR_CODES = {"auth_error", "bad_request", "content_filter"}

    def _is_retryable(self, err: LLMError) -> bool:
        """判断该错误是否可重试。"""
        return err.code not in self.UNRETRYABLE_ERROR_CODES

    # ============================================================
    # 单次迭代（含重试）
    # ============================================================

    def _step(self) -> Generator[AgentEvent, None, None]:
        """执行一轮模型推理（带自动重试）。

        这一轮会：
        1. 从 memory 取出完整消息历史
        2. 把可用工具 schema 一起传给模型
        3. 消费 LLM 的流式输出，实时发出 message/reasoning/tool_call 等事件
        4. 收到最终 LLMResponse 后，根据是否包含工具调用决定下一步

        如果 LLM 调用抛出可重试错误（网络 / 超时 / 频率限制等），
        会在退避后自动重试，直到成功或耗尽重试次数。
        """
        messages = self.agent.memory.get_messages()
        tools = self.agent.tools.to_openai_tools() or None

        max_attempts = 1 + self.agent.max_retries  # 首次 + 重试次数
        last_err: LLMError | None = None

        for attempt in range(max_attempts):
            # 累积最终输出内容，每次重试从头开始拼接
            full_content = ""
            full_reasoning = ""

            try:
                for item in self.llm.stream(messages, tools):
                    self.state.check_cancel()

                    # LLMResponse 表示一次流式响应已经收束，拿到了最终结构化结果
                    if isinstance(item, LLMResponse):
                        if item.usage:
                            yield usage_update_event(item.usage)
                        if full_reasoning:
                            yield reasoning_complete_event(content=full_reasoning)
                        if full_content:
                            yield message_complete_event(content=full_content)
                        yield from self._handle_tool_calls(item)
                        return

                    # StreamDelta 表示流式增量
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

                # stream 正常结束，检查取消
                if self.state.is_cancelled:
                    raise CancelledError()

                # 没有 LLMResponse 分支 = 模型直接输出了最终文本（无工具调用）
                if full_content:
                    self.agent.memory.add_assistant(full_content, reasoning=full_reasoning or None)
                    if full_reasoning:
                        yield reasoning_complete_event(content=full_reasoning)
                    yield message_complete_event(content=full_content)
                yield done_event(success=True, reason=DoneReason.COMPLETED)
                self.state.complete()
                return

            except CancelledError:
                # 取消不重试，保存已有内容后向上传播
                if full_content:
                    self.agent.memory.add_assistant(full_content, reasoning=full_reasoning or None)
                    if full_reasoning:
                        yield reasoning_complete_event(content=full_reasoning)
                    yield message_complete_event(content=full_content)
                raise

            except Exception as e:
                # 如果底层抛错时已经是取消态，统一转取消
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

                # 不可重试 或 已用完重试次数 → 直接失败
                if not self._is_retryable(err) or is_last_attempt:
                    logger.warning(f"LLM 调用失败: [{err.code}] {err.message}")
                    yield error_event(message=err.message, code=err.code)
                    yield done_event(success=False, reason=DoneReason.ERROR)
                    self.state.fail(f"[{err.code}] {err.message}")
                    return

                # 可重试：发出 retry 事件，固定等待后重试
                logger.info(
                    f"LLM 调用可重试错误 [{err.code}]，第 {attempt + 1}/{max_attempts - 1} 次重试"
                )
                yield retry_event(
                    code=err.code,
                    message=err.message,
                    attempt=attempt + 1,
                    max_attempts=max_attempts - 1,
                )

                # 固定等待（cancel-aware：取消信号到达则立即醒来）
                was_cancelled = self.state.cancel_token.wait(self.agent.retry_delay)
                if was_cancelled:
                    raise CancelledError()

    # ============================================================
    # 工具调用
    # ============================================================

    def _handle_tool_calls(self, response: LLMResponse) -> Generator[AgentEvent, None, None]:
        """处理模型返回的工具调用。

        处理顺序：
        1. 先解析 response.tool_calls
        2. 将 assistant 这条"发起工具调用"的消息写入 memory
        3. 对解析失败的调用，直接写入错误结果
        4. 对解析成功的调用，交给 ToolHandler.execute 执行
        5. 将工具结果写回 memory，供下一轮模型继续使用
        """
        parsed: list[tuple[str, str, dict | None]] = [
            self.tool_handler.parse_tool_call(tc) for tc in response.tool_calls
        ]

        # 先把 assistant 原始消息（含 tool_calls）记进 memory。
        # 这一步很关键，否则后续工具结果会缺少对应的调用上下文。
        self.agent.memory.add_raw(
            self.tool_handler.build_assistant_message(response),
            usage=response.usage,
        )

        # 处理解析失败的工具调用。
        # 典型场景是模型输出了不完整或非法 JSON 参数。
        for call_id, name, args in parsed:
            if args is None:
                error_msg = "[PARSE_ERROR] Tool call JSON truncated or malformed. Please retry."
                yield tool_result_event(id=call_id, name=name, result=error_msg, error=error_msg, status="failed")
                self.agent.memory.add_tool_result(call_id, error_msg)

        # 可执行的：仅保留解析成功的工具调用
        valid = [(cid, name, args) for cid, name, args in parsed if args is not None]
        if not valid:
            return

        # 顺序执行工具。
        # ToolHandler.execute 会持续产出工具相关事件（例如 tool_result）。
        cancelled = False
        for event in self.tool_handler.execute(valid, self.state):
            yield event
            if event["type"].value == "tool_result":
                # 工具结果写回 memory，供下一轮模型读取
                self.agent.memory.add_tool_result(event["data"]["id"], event["data"]["result"])
                # 某些工具可能在执行中感知到取消，这里做额外标记
                if event["data"].get("status") == "cancelled":
                    cancelled = True

        # 如果工具执行阶段被取消，则抛出取消异常交给上层统一收尾
        if cancelled:
            raise CancelledError()
