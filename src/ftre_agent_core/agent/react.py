"""ReAct Agent - 异步推理与行动循环。"""
import logging
from collections.abc import AsyncGenerator

from ftre_agent_core.message_context import MessageContext
from ftre_agent_core.permission import PermissionEngine
from ftre_agent_core.state import AgentState
from ftre_agent_core.tool import ToolRegistry
from ftre_agent_core.tracing import Tracer

from ..event import AgentStreamEvent
from ..hooks import HookDispatcher
from .runner import ReActRunner

logger = logging.getLogger(__name__)


class ReActAgent:
    """一个可复用的 Agent 外壳，持有持久状态并委托一次性运行给 Runner。

    这里故意不保存 Channel、Session 或 Plugin 注册表：宿主每次调用 ``run``
    时通过 ``runtime_context`` 提供本轮坐标，Hook Dispatcher 则作为纯协议
    注入。这样同一个 Core 包既能被 ftre 使用，也能在没有 Gateway 的测试中运行。
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str | None = None,
        api_type: str = "completions",
        system_prompt: str = "",
        tool_registry: ToolRegistry | None = None,
        max_iterations: int | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str = "",
        state: AgentState | None = None,
        max_retries: int = 5,
        retry_delay: float = 3.0,
        tracer: Tracer | None = None,
        hooks: HookDispatcher | None = None,
        hook_context: object | None = None,
    ):
        # 下面这些字段是“静态 Agent 配置”；本次 run 的 iteration、reply_id、
        # cancellation 等短生命周期状态统一放在 ReActRunner.state，避免多次
        # 调用之间串用临时数据。
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.api_type = api_type
        self._system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.tracer = tracer or Tracer()
        self.hooks = hooks
        self.hook_context = hook_context
        self._state = state if state is not None else AgentState()
        # 权限引擎始终由 Agent 内部创建：规则的唯一事实源是
        # AgentState.permission_context，调用方无需也不应注入引擎实例。
        self._permission_engine = PermissionEngine()
        self._registry = tool_registry if tool_registry is not None else ToolRegistry()
        self._runner = ReActRunner(self)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        self._system_prompt = value

    @property
    def state(self) -> AgentState:
        """Persistent, injectable agent state."""
        return self._state

    @property
    def permission_engine(self) -> PermissionEngine:
        """权限决策引擎（Agent 内部创建，始终可用）。"""
        return self._permission_engine

    @property
    def run_state(self):
        """Temporary state for the current or most recent run."""
        return self._runner.state

    @property
    def messages(self) -> list[dict]:
        """Provider-compatible view of the persistent message context."""
        return MessageContext.messages(self._state.context)

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._registry

    @property
    def runner(self) -> ReActRunner:
        return self._runner

    async def run(
        self, message, runtime_context: dict | None = None
    ) -> AsyncGenerator[AgentStreamEvent, None]:
        """启动一次异步 ReAct 流；真正执行发生在调用方开始迭代之后。

        ``message`` 可以是新用户输入，也可以是权限确认事件。Core 只负责把
        两者交给 Runner 的相应状态机路径；session_id、turn_id 等宿主坐标不
        作为 Core 全局状态保存，而是随 ``runtime_context`` 进入本次 RunState。
        """
        async for event in self._runner.run(message, runtime_context=runtime_context):
            yield event

    def cancel_nowait(self) -> None:
        """请求取消当前 Task；取消结果通过正常的 ReplyEnd 流通知宿主。"""
        self._runner.cancel_nowait()
