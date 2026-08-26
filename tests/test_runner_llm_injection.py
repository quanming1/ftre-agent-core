"""验证宿主可以在 Runner 构造阶段注入统一 LLM Service seam。"""

from __future__ import annotations

import pytest

from ftre_agent_core.agent import ReActAgent


class InjectedAdapter:
    """只实现 Core Runner 需要的最小 LLM 适配器形状。"""

    model = "injected-model"
    provider = "injected-provider"
    api_type = "completions"

    async def stream(self, messages, tools=None):
        del tools
        # 保留 async iterator 形状；Core 实际调用总会传入消息列表。
        if messages is None:
            yield None

    def cancel(self) -> None:
        return None


class RunningTask:
    def done(self) -> bool:
        return False


def test_agent_accepts_injected_llm_without_constructing_default_client() -> None:
    adapter = InjectedAdapter()

    # 空 api_key 以前会在 ReActRunner 初始化 OpenAI 客户端时直接失败；
    # 注入 seam 后，Core 不需要知道宿主凭据或 Provider 实现。
    agent = ReActAgent(model="unused", api_key="", llm=adapter)

    assert agent.runner.llm is adapter


def test_runner_rejects_llm_replacement_while_running() -> None:
    agent = ReActAgent(model="unused", api_key="fake", llm=InjectedAdapter())
    agent.runner._run_task = RunningTask()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="while agent is running"):
        agent.runner.set_llm(InjectedAdapter())
