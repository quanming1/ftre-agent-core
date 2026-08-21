"""演示：工具请求确认后，Core 如何恢复执行。"""
import asyncio
import json
from pathlib import Path

from ftre_agent_core.agent.react import ReActAgent
from ftre_agent_core.event import UserConfirmResultEvent
from ftre_agent_core.permission import (
    PermissionBehavior,
    PermissionEngine,
    PermissionRule,
)
from ftre_agent_core.state import AgentState
from ftre_agent_core.tool import ToolRegistry, tool

CONFIG_PATH = Path(r"C:\Users\蒋全明\.ftre\config.json")


@tool(description="回显文本")
def echo(text: str) -> str:
    return f"123: {text}"


async def main() -> None:
    # 固定读取本机 DeepSeek 官方配置，密钥不写入代码。
    provider = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["providers"]["DeepSeek 官方"]
    registry = ToolRegistry()
    registry.register(echo)
    state = AgentState(permission_context={"permission_rules": [
        PermissionRule(id="ask-echo", tool_name="echo", behavior=PermissionBehavior.ASK).model_dump()
    ]})
    agent = ReActAgent(
        model="deepseek-v4-flash", api_key=provider["api_key"], api_base=provider["api_base"],
        tool_registry=registry, state=state,
        permission_engine=PermissionEngine(), max_retries=0,
    )

    pending = [e async for e in agent.run("试试你的echo tool？看看输出是什么")]
    confirm = next(e for e in pending if e.type == "REQUIRE_USER_CONFIRM")
    print("待确认：", confirm.model_dump())

    # 确认事件是输入；Core 从 AgentState 的 ASKING 状态恢复。
    answer = input("输入“我确定执行”以执行 echo：").strip()
    confirm_input = UserConfirmResultEvent(
        reply_id=confirm.reply_id,
        tool_call_id=confirm.tool_call_id,
        approved=answer == "1",
    )
    print("终端输入：", repr(answer))
    print("确认输入：", json.dumps(confirm_input.model_dump(), ensure_ascii=False))
    resumed = [e async for e in agent.run(confirm_input)]
    print("恢复事件：", [e.type for e in resumed])
    print("上下文：", agent.messages[-1]["content"])


if __name__ == "__main__":
    asyncio.run(main())
