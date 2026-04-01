"""
InterruptHandler - 中断/恢复处理器

职责：
- 判断某个工具是否需要中断确认
- 执行中断：保存 checkpoint，更新 state，yield INTERRUPT 事件
- 执行恢复：从断点处继续执行剩余 tool_calls

不直接执行工具，通过 ToolHandler 代理执行。
取消时 CancelledError 自然冒泡到 ReActRunner._loop() 统一处理。
"""
from typing import Generator, TYPE_CHECKING
from ftre_agent_core.checkpoint import CheckpointType
from ftre_agent_core.agent.runner.state import CancelledError
from ftre_agent_core.agent.event import (
    AgentEvent,
    interrupt_event,
    tool_result_event,
)

if TYPE_CHECKING:
    from ..state import RunState
    from .tool_handler import ToolHandler
    from ftre_agent_core.memory import MemoryManager


class InterruptHandler:
    """中断/恢复处理器"""

    def __init__(
        self,
        state: "RunState",
        memory: "MemoryManager",
        tool_handler: "ToolHandler",
        interrupt_before: list[str] | None = None,
        interrupt_all: bool = False,
    ):
        self.state = state
        self.memory = memory
        self.tool_handler = tool_handler
        self.interrupt_before = interrupt_before or []
        self.interrupt_all = interrupt_all

    def should_interrupt(self, tool_name: str) -> bool:
        """判断某个工具是否需要中断确认"""
        if self.interrupt_all:
            return True
        if self.interrupt_before and tool_name in self.interrupt_before:
            return True
        return False

    def do_interrupt(
        self, call_id: str, name: str, arguments: dict
    ) -> Generator[AgentEvent, None, None]:
        """
        执行中断：保存 checkpoint → 更新 state → yield INTERRUPT 事件

        调用后 runner 应立即 return，等待 resume。
        """
        cp = self.memory.save_checkpoint(
            label=f"[interrupt] {name}",
            type=CheckpointType.INTERRUPT,
            metadata={"tool_name": name, "tool_call_id": call_id},
        )
        self.state.interrupt(checkpoint_id=cp.id)

        yield interrupt_event(
            tool_call_id=call_id,
            tool_name=name,
            arguments=arguments,
            checkpoint_id=cp.id,
        )

    def resume_tool_calls(self, approved: bool) -> Generator[AgentEvent, None, None]:
        """
        从中断处继续执行 pending_tool_calls。

        策略：
        - 先处理被中断的那个工具（approve/reject）
        - 剩余工具：如果全部不需要 interrupt → 并行执行
        - 否则 → 串行执行（可能再次中断）

        CancelledError 时补齐未执行的 tool results 后 re-raise，
        由 ReActRunner.resume() 或 _loop() 统一处理。
        """
        tool_calls = self.state.pending_tool_calls
        start_index = self.state.tool_call_index

        if start_index >= len(tool_calls):
            return

        remaining_parsed: list[tuple[str, str, dict]] = []

        try:
            # 处理被中断的那个工具
            current_tc = tool_calls[start_index]
            call_id, name, arguments = self.tool_handler.parse_tool_call(current_tc)

            # 处理解析失败的情况
            if arguments is None:
                error_msg = (
                    f"[PARSE_ERROR] Tool call JSON appears truncated or malformed. "
                    f"Please retry this tool call with complete arguments."
                )
                yield tool_result_event(id=call_id, name=name, result=error_msg, error=error_msg, status="failed")
                self.memory.add_tool_result(call_id, error_msg)
            elif approved:
                yield from self._execute_and_record(call_id, name, arguments)
            else:
                reject_result = f"[用户拒绝执行此工具: {name}]"
                yield tool_result_event(id=call_id, name=name, result=reject_result)
                self.memory.add_tool_result(call_id, reject_result)

            # 收集剩余的工具
            remaining_start = start_index + 1
            if remaining_start >= len(tool_calls):
                return

            for i in range(remaining_start, len(tool_calls)):
                tc = tool_calls[i]
                remaining_parsed.append(self.tool_handler.parse_tool_call(tc))

            # 处理解析失败的 tool_calls
            parse_failed = [(cid, n) for cid, n, args in remaining_parsed if args is None]
            for cid, n in parse_failed:
                error_msg = (
                    f"[PARSE_ERROR] Tool call JSON appears truncated or malformed. "
                    f"Please retry this tool call with complete arguments."
                )
                yield tool_result_event(id=cid, name=n, result=error_msg, error=error_msg, status="failed")
                self.memory.add_tool_result(cid, error_msg)

            # 过滤掉解析失败的
            remaining_parsed = [(cid, n, args) for cid, n, args in remaining_parsed if args is not None]
            if not remaining_parsed:
                return

            # 判断剩余工具是否可以并行
            has_interrupt = any(
                self.should_interrupt(n) for _, n, _ in remaining_parsed
            )

            if not has_interrupt and len(remaining_parsed) > 1:
                # 并行执行剩余工具
                self.state.check_cancel()
                for event in self.tool_handler.execute_parallel(
                    remaining_parsed, self.state
                ):
                    yield event
                    if event["type"].value == "tool_result":
                        self.memory.add_tool_result(
                            event["data"]["id"], event["data"]["result"]
                        )
            else:
                # 串行执行剩余工具（可能再次中断）
                for i, (r_call_id, r_name, r_arguments) in enumerate(remaining_parsed):
                    actual_index = remaining_start + i
                    self.state.tool_call_index = actual_index

                    self.state.check_cancel()

                    if self.should_interrupt(r_name):
                        yield from self.do_interrupt(r_call_id, r_name, r_arguments)
                        return

                    yield from self._execute_and_record(r_call_id, r_name, r_arguments)

        except CancelledError:
            remaining_from = self.state.tool_call_index - remaining_start
            self._fill_remaining(remaining_parsed, from_index=max(0, remaining_from))
            raise


    def _execute_and_record(
        self, call_id: str, name: str, arguments: dict
    ) -> Generator[AgentEvent, None, None]:
        """
        执行工具并记录结果到 Memory。

        内部调用 execute_and_emit，事件流保证完整。
        取消时通过检查 tool_result status 判断，抛 CancelledError 冒泡。
        """
        for event in self.tool_handler.execute_and_emit(
            call_id, name, arguments, self.state
        ):
            yield event
            if event["type"].value == "tool_result":
                self.memory.add_tool_result(
                    event["data"]["id"], event["data"]["result"]
                )
                if event["data"].get("status") == "cancelled":
                    raise CancelledError("user_cancelled")

    def _fill_remaining(self, parsed: list[tuple[str, str, dict]], from_index: int) -> None:
        """为未执行的 tool_calls 补上取消结果"""
        for i in range(from_index, len(parsed)):
            call_id, _, _ = parsed[i]
            self.memory.add_tool_result(call_id, "[用户取消，未执行]")
