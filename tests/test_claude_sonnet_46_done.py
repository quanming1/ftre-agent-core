"""
诊断脚本：用明略网关的 claude-sonnet-4-6 跑一次简单对话，
打印事件流，确认 done 事件是否产出。

运行:
    python tests/test_claude_sonnet_46_done.py
"""
import json
import sys
from pathlib import Path

# 让脚本能直接 import ftre_agent_core
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ftre_agent_core.agent import ReActAgent, EventType  # noqa: E402


API_KEY = "sk-HIYFHsm6Oyx1MotZXpxtXOMfDGj6azzPKw3GPQX4RxASrAZH"
API_BASE = "https://llm-gateway.mlamp.cn/v1"
MODEL = "openai/claude-sonnet-4-6"


def run_case(prompt: str) -> dict:
    """跑一次对话，返回 {events, types, has_done, full_text}"""
    agent = ReActAgent(
        model=MODEL,
        api_key=API_KEY,
        api_base=API_BASE,
        system_prompt="你是一个简洁的助手，用一两句话回答。",
        tools=[],
    )

    events = []
    for ev in agent.run(prompt):
        events.append(ev)

    types = [e["type"].value if hasattr(e["type"], "value") else str(e["type"]) for e in events]
    full_text_parts = []
    for e in events:
        t = e["type"]
        if t == EventType.MESSAGE:
            full_text_parts.append(e["data"].get("content", ""))
        elif t == EventType.MESSAGE_COMPLETE and not full_text_parts:
            full_text_parts.append(e["data"].get("content", ""))
    full_text = "".join(full_text_parts)

    has_done = any(e["type"] == EventType.DONE for e in events)
    has_complete = any(e["type"] == EventType.MESSAGE_COMPLETE for e in events)
    return {
        "events": events,
        "types": types,
        "has_done": has_done,
        "has_complete": has_complete,
        "full_text": full_text,
    }


def print_event_summary(prompt: str, result: dict) -> None:
    print("=" * 70)
    print(f"PROMPT: {prompt}")
    print("-" * 70)
    print("EVENT TIMELINE:")
    for i, ev in enumerate(result["events"]):
        t = ev["type"].value if hasattr(ev["type"], "value") else str(ev["type"])
        data = ev.get("data", {})
        # message 类型只显示长度，避免刷屏
        if t in ("message", "reasoning"):
            content = data.get("content", "")
            print(f"  [{i:03d}] {t:<22s} content_len={len(content)}")
        else:
            try:
                preview = json.dumps(data, ensure_ascii=False, default=str)[:200]
            except Exception as exc:
                preview = f"<unserializable: {exc}>"
            print(f"  [{i:03d}] {t:<22s} {preview}")
    print("-" * 70)
    print(f"TYPES UNIQUE   : {sorted(set(result['types']))}")
    print(f"HAS_DONE       : {result['has_done']}")
    print(f"HAS_COMPLETE   : {result['has_complete']}")
    print(f"FULL_TEXT_LEN  : {len(result['full_text'])}")
    print(f"FULL_TEXT      : {result['full_text']!r}")
    print()


def main() -> int:
    prompts = [
        "你好",
        "1+1 等于几？只回答一个数字。",
        "用一句话介绍自己",
    ]
    failures = 0
    for p in prompts:
        try:
            r = run_case(p)
        except Exception as e:
            print(f"[FAIL] prompt={p!r} 抛出异常: {e}")
            failures += 1
            continue
        print_event_summary(p, r)
        if not r["has_done"]:
            print(f"[FAIL] prompt={p!r} 缺少 done 事件")
            failures += 1
        if not r["full_text"]:
            print(f"[WARN] prompt={p!r} 无文本输出")

    print("=" * 70)
    if failures:
        print(f"总结: {failures} 个用例失败")
        return 1
    print("总结: 所有用例都产出了 done 事件 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
