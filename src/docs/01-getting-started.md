# 快速开始

## 安装

```bash
pip install -e ".[dev]"
```

## 运行 Agent

`ReActAgent.run()` 是异步生成器，产出扁平的 `AgentStreamEvent`：

```python
import asyncio
import os

from ftre_agent_core.agent import ReActAgent


async def main():
    agent = ReActAgent(
        model="openai/gpt-4.1-mini",
        api_key=os.environ["OPENAI_API_KEY"],
        system_prompt="You are a helpful assistant.",
    )

    async for event in agent.run("你好"):
        if event.type == "TEXT_BLOCK_DELTA":
            print(event.delta, end="", flush=True)
        elif event.type == "REPLY_END":
            print(f"\nfinished: {event.finished_reason}")


asyncio.run(main())
```

事件通过 `model_dump(mode="json")` 扁平序列化。聚合消息使用
`Msg.append_event()` 在内存中构建；流式 Event 不应当被当作消息快照持久化。

完整事件列表见 [08-events.md](08-events.md)。
