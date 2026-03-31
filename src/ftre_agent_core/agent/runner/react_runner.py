"""
ReActRunner - ReAct 执行引擎

只负责 ReAct 循环编排：
- run()    → 启动循环
- resume() → 从中断处恢复
- cancel() → 用户主动取消
- _loop()  → 主循环
- _step()  → 单次迭代（调 LLM → 处理响应）

具体的 LLM 调用、工具执行、中断逻辑分别委托给 handler/ 下的处理器。

取消策略：
- cancel() 设标志位 + threading.Event.set()（线程安全，立即唤醒等待者）
- 工具在子线程执行，主线程每 100ms 轮询取消信号
- 任何检查点检测到取消 → 抛出 CancelledError → _loop() 统一捕获善后
- 善后：已有内容写入 Memory，未执行的 tool_calls 补上结果
"""
import logging
from typing import Generator, TYPE_CHECKING
from ftre_agent_core.tool.builtins import BUILTIN_TOOL_FACTORIES
from .state import RunState, CancelledError
from .handler import LLMHandler, LLMResponse, LLMError, StreamDelta, ToolHandler, InterruptHandler
from ..event import (
    DoneReason,
    AgentEvent,
    message_event,
    message_complete_event,
    max_iterations_event,
    done_event,
    usage_update_event,
    error_event,
    tool_call_streaming_event,
)

if TYPE_CHECKING:
    from ..react import ReActAgent

logger = logging.getLogger(__name__)


class ReActRunner:
    """
    ReAct 执行引擎

    编排 LLMHandler / ToolHandler / InterruptHandler 完成 ReAct 循环。
    自身不包含工具执行或中断判断的具体逻辑。

    取消机制：
    - 外部调 cancel() → state 设标志位 + Event.set()
    - 各检查点调 state.check_cancel() → 抛 CancelledError
    - _loop() 统一 catch CancelledError，执行善后逻辑
    """

    # 连续空响应最大重试次数，超过则视为异常终止
    MAX_EMPTY_RETRIES = 3

    def __init__(self, agent: "ReActAgent"):
        self.agent = agent
        self.state = RunState()
        self._consecutive_empty = 0

        self.llm = LLMHandler(agent.model, agent.api_key, agent.api_base)
        self.tool_handler = ToolHandler(agent.tools)
        self.interrupt_handler = InterruptHandler(
            state=self.state,
            memory=agent.memory,
            tool_handler=self.tool_handler,
            interrupt_before=getattr(agent, "interrupt_before", None),
            interrupt_all=getattr(agent, "interrupt_all", False),
        )

        # 注入内置工具（think）
        for factory in BUILTIN_TOOL_FACTORIES:
            self.agent.tools.register(factory())

    def cancel(self, timeout: float | None = None) -> bool:
        """
        用户主动取消，阻塞等待善后完成。

        Returns:
            True = 善后已完成，False = 超时
        """
        if not self.state.is_running:
            return True
        self.state.cancel()
        self.llm.cancel()  # 强关 HTTP 连接，避免线程卡在等待 chunk 上
        return self.state.wait_done(timeout)

    def run(self, message) -> Generator[AgentEvent, None, None]:
        """
        启动 ReAct 循环。

        Args:
            message: str 或 list
                - str: 单条用户消息，自动 add_user 到 memory
                - list: 完整消息列表（含历史 + 当前用户消息），
                  直接设置到 memory（不触发持久化，因为这些消息已在 MongoDB 中）
        """
        self.state.start()

        if isinstance(message, str):
            self.agent.memory.add_user(message)
            yield from self._loop(message)
        else:
            self._restore_messages(message)
            # 取最后一条 user 消息作为 label
            label = ""
            for m in reversed(message):
                content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
                role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
                if role == "user" and content:
                    label = content
                    break
            yield from self._loop(label)

    def _restore_messages(self, messages: list) -> None:
        """
        将已有消息列表恢复到 memory。

        所有消息都是 openai 格式 dict，直接用 add_raw 逐条加入。
        这些是纯内存操作（默认 MemoryManager），不涉及持久化。

        每轮都是全新的 agent（engine 不复用），memory 一定是空的，
        直接 append 即可。
        """
        memory = self.agent.memory
        for msg in messages:
            if isinstance(msg, dict):
                if msg.get("role") == "system":
                    continue
                memory.add_raw(msg)
            else:
                if getattr(msg, "role", "") == "system":
                    continue
                memory.add_raw(msg)

    def resume(self, approved: bool = True) -> Generator[AgentEvent, None, None]:
        """从中断处恢复执行"""
        if not self.state.is_interrupted:
            return

        self.state.resume()

        try:
            yield from self.interrupt_handler.resume_tool_calls(approved)
        except CancelledError:
            yield from self._on_cancelled()
            return

        if not self.state.is_done and not self.state.is_interrupted:
            yield from self._loop()

    # ============================================================
    # 主循环（CancelledError 的唯一捕获点）
    # ============================================================

    def _token_usage(self) -> dict:
        """获取当前累计 token 用量（用于 done_event）"""
        return self.agent.memory.token.to_dict()

    def _loop(self, label: str = "") -> Generator[AgentEvent, None, None]:
        """
        ReAct 主循环

        所有 CancelledError 在此统一捕获，执行善后后 yield DONE。
        内部的 _step / _handle_tool_calls 只管抛，不管接。
        """
        try:
            remaining = self.agent.max_iterations - self.state.iteration

            for _ in range(remaining):
                self.state.check_cancel()
                self.state.next_iteration()

                # Compaction: 每次迭代开始前检查上下文是否溢出
                if hasattr(self.agent.memory, "compact"):
                    self.agent.memory.compact()

                yield from self._step()

                # 每次迭代结束后实时更新 usage（未结束时推送中间状态）
                if not self.state.is_done:
                    yield usage_update_event(self._token_usage())

                if self.state.is_interrupted:
                    return

                if self.state.is_done:
                    self.agent.memory.after_loop()
                    self.agent.memory.save_checkpoint(
                        label=label[:50] if label else ""
                    )
                    return

            yield max_iterations_event(iterations=self.agent.max_iterations)
            yield done_event(success=False, reason=DoneReason.MAX_ITERATIONS, usage=self._token_usage())
            self.state.complete()
            self.agent.memory.after_loop()
            self.agent.memory.save_checkpoint(
                label=f"[max_iterations] {label[:30]}" if label else "[max_iterations]"
            )

        except CancelledError:
            yield from self._on_cancelled()

    # ============================================================
    # 单次迭代
    # ============================================================

    def _step(self) -> Generator[AgentEvent, None, None]:
        """
        单次迭代：调用 LLM，处理响应。

        取消时抛 CancelledError，由 _loop() 捕获。
        """
        messages = self.agent.memory.get_messages()
        tools = self.agent.tools.to_openai_tools()
        full_content = ""
        last_usage = None

        try:
            for item in self.llm.stream(messages, tools if tools else None):
                self.state.check_cancel()

                if isinstance(item, LLMResponse):
                    self._consecutive_empty = 0
                    last_usage = item.usage
                    if last_usage:
                        self.agent.memory.token.add(last_usage)
                    # 如果 LLM 在调用工具的同时输出了文本，记录完整内容
                    if full_content:
                        yield message_complete_event(content=full_content)
                    yield from self._handle_tool_calls(item)
                    return

                elif isinstance(item, StreamDelta):
                    if item.content:
                        full_content += item.content
                        yield message_event(content=item.content)
                    if item.tool_calls:
                        yield tool_call_streaming_event(item.tool_calls)
                    if item.usage:
                        last_usage = item.usage
                        self.agent.memory.token.add(item.usage)

            # LLM 返回纯文本、无 tool_calls → 任务自然结束
            if full_content:
                self._consecutive_empty = 0
                self.agent.memory.add_assistant(full_content, usage=last_usage)
                yield message_complete_event(content=full_content)
                yield done_event(success=True, reason=DoneReason.COMPLETED, usage=self._token_usage())
                self.state.complete()
            else:
                # LLM 返回了空内容（无文本、无工具调用）
                self._consecutive_empty += 1
                logger.warning(
                    f"LLM 返回空响应（无文本、无工具调用）"
                    f"[连续第 {self._consecutive_empty} 次]"
                )
                if self._consecutive_empty >= self.MAX_EMPTY_RETRIES:
                    logger.error(
                        f"连续 {self.MAX_EMPTY_RETRIES} 次空响应，终止循环"
                    )
                    self.agent.memory.add_assistant(
                        "[系统] LLM 连续返回空响应，无法继续"
                    )
                    yield error_event(
                        message=f"LLM 连续 {self.MAX_EMPTY_RETRIES} 次返回空响应",
                        code="empty_response",
                    )
                    yield done_event(success=False, reason=DoneReason.ERROR, usage=self._token_usage())
                    self.state.fail("连续空响应")
                # 未达上限：不设 complete，_loop 会继续下一次迭代

        except CancelledError:
            # 善后：把已有的部分内容写入 Memory，再重新抛出
            if full_content:
                self.agent.memory.add_assistant(full_content)
                yield message_complete_event(content=full_content)
            else:
                self.agent.memory.add_assistant("[用户取消]")
            raise

        except Exception as e:
            # cancel() 强关 HTTP 连接会导致网络异常（非 CancelledError），
            # 通过 state.is_cancelled 识别：转为 CancelledError 走正常善后路径
            if self.state.is_cancelled:
                logger.info(f"[_step] LLM 流被取消关闭: {e}")
                if full_content:
                    self.agent.memory.add_assistant(full_content)
                    yield message_complete_event(content=full_content)
                else:
                    self.agent.memory.add_assistant("[用户取消]")
                raise CancelledError() from e

            err = LLMError.classify(e)
            err_str = f"LLM 调用失败: [{err.code}] {err.message}"
            logger.warning(err_str)
            self.agent.memory.add_assistant(f"[系统错误] {err_str}")
            yield error_event(message=err.message, code=err.code)
            yield done_event(success=False, reason=DoneReason.ERROR, usage=self._token_usage())
            self.state.fail(err_str)

    def _handle_tool_calls(self, response: LLMResponse) -> Generator[AgentEvent, None, None]:
        """
        处理工具调用（自动选择并行或串行）。

        策略：
        - 单个 tool_call → 串行（无并行开销）
        - 多个 tool_calls 且无需 interrupt → 并行执行
        - 多个 tool_calls 但有需要 interrupt 的 → 串行（中断语义要求逐个处理）

        CancelledError 可能从 check_cancel() 或工具执行中抛出，
        在此捕获后补齐未执行的 tool results，再重新抛出给 _loop()。
        """
        self.state.pending_tool_calls = response.tool_calls

        tool_calls = response.tool_calls

        # 预解析所有 tool_calls（必须在写入 memory 之前，
        # 以免 parse 失败时留下不完整的 assistant message 污染上下文）
        parsed: list[tuple[str, str, dict]] = []
        for tc in tool_calls:
            parsed.append(self.tool_handler.parse_tool_call(tc))

        assistant_msg = self.tool_handler.build_assistant_message(response)
        self.agent.memory.add_raw(assistant_msg, usage=response.usage)

        # 判断是否可以并行：多个 tool_call 且全部不需要 interrupt
        can_parallel = (
            len(parsed) > 1
            and not any(
                self.interrupt_handler.should_interrupt(name)
                for _, name, _ in parsed
            )
        )

        if can_parallel:
            yield from self._handle_tool_calls_parallel(parsed)
        else:
            yield from self._handle_tool_calls_sequential(parsed)

    def _handle_tool_calls_parallel(
        self, parsed: list[tuple[str, str, dict]]
    ) -> Generator[AgentEvent, None, None]:
        """
        并行执行所有工具调用。

        前提：调用方已确认所有工具都不需要 interrupt。
        """
        try:
            self.state.check_cancel()

            cancelled = False
            for event in self.tool_handler.execute_parallel(
                parsed, self.state
            ):
                yield event
                # 收到 tool_result 时写入 Memory
                if event["type"].value == "tool_result":
                    self.agent.memory.add_tool_result(
                        event["data"]["id"], event["data"]["result"]
                    )
                    if event["data"].get("status") == "cancelled":
                        cancelled = True

            if cancelled:
                raise CancelledError("user_cancelled")

        except CancelledError:
            # 并行取消时，已执行的工具已经通过事件流 yield 了 tool_result，
            # 只需为尚未执行的补齐
            for call_id, name, _ in parsed:
                self.agent.memory.add_tool_result(call_id, "[用户取消，未执行]")
            raise

        except Exception as e:
            logger.warning(f"[并行工具执行异常] {e}")
            yield error_event(message=str(e), code="tool_error")

    def _handle_tool_calls_sequential(
        self,
        parsed: list[tuple[str, str, dict]],
    ) -> Generator[AgentEvent, None, None]:
        """
        串行执行工具调用（原始逻辑，支持 interrupt）。
        """
        for i, (call_id, name, arguments) in enumerate(parsed):
            try:
                self.state.check_cancel()

                self.state.tool_call_index = i

                # 检查中断
                if self.interrupt_handler.should_interrupt(name):
                    yield from self.interrupt_handler.do_interrupt(call_id, name, arguments)
                    return

                # 执行工具，事件流保证完整（tool_call → tool_result）
                result = None
                for event in self.tool_handler.execute_and_emit(
                    call_id, name, arguments, self.state
                ):
                    yield event
                    if event["type"].value == "tool_result":
                        self.agent.memory.add_tool_result(
                            event["data"]["id"], event["data"]["result"]
                        )
                        result = event

                # 工具被取消：补齐剩余 tool results，然后中断循环
                if result and result["data"].get("status") == "cancelled":
                    self._fill_remaining(parsed, from_index=i + 1)
                    raise CancelledError("user_cancelled")

            except CancelledError:
                self._fill_remaining(parsed, from_index=i)
                raise

            except Exception as e:
                err_msg = f"[工具执行异常] {name}: {e}"
                logger.warning(err_msg)
                self.agent.memory.add_tool_result(call_id, err_msg)
                yield error_event(message=str(e), code="tool_error")

    # ============================================================
    # Cancel 善后
    # ============================================================

    def _on_cancelled(self) -> Generator[AgentEvent, None, None]:
        """CancelledError 的统一善后：yield DONE 事件，然后通知等待者"""
        try:
            yield done_event(success=False, reason=DoneReason.CANCELLED, usage=self._token_usage())
        finally:
            self.state.mark_done()

    def _fill_remaining(self, parsed: list[tuple[str, str, dict]], from_index: int) -> None:
        """为未执行的 tool_calls 补上取消结果（OpenAI 要求每个 tool_call 必须有 result）"""
        for i in range(from_index, len(parsed)):
            call_id, name, _ = parsed[i]
            self.agent.memory.add_tool_result(call_id, "[用户取消，未执行]")
