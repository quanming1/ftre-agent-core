"""把 state.json 中的一条 Msg 转成 Core 实际发送给 OpenAI-compatible API 的格式。"""

import json
import sys
from pathlib import Path

from ftre_agent_core.message import Msg
from ftre_agent_core.message_context import MessageContext

DEFAULT_INPUT = Path(__file__).resolve().parent / "Njiknfgjinr.json"


def main() -> None:
    # Event 会先经 Msg.append_event() 聚合；附件已经是聚合完成的 Msg。
    path = Path(next((arg for arg in sys.argv[1:] if not arg.startswith("--")), DEFAULT_INPUT))
    msg = Msg.model_validate_json(path.read_text(encoding="utf-8"))
    messages = MessageContext.messages([msg])

    # 在本测试脚本同级目录（ftre-agent-core 根目录）保存结果。
    output_path = Path(__file__).resolve().parent / f"{path.stem}.openai.json"
    output_path.write_text(
        json.dumps(messages, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已生成: {output_path}")

    if "--summary" not in sys.argv:
        print(json.dumps(messages, ensure_ascii=False, indent=2))
        return

    print(f"原始 Msg 内容块: {len(msg.content)}，转换后 OpenAI 消息: {len(messages)}")
    for index, item in enumerate(messages):
        calls = [call["id"] for call in item.get("tool_calls", [])]
        print(index, item["role"], f"tool_call_id={item.get('tool_call_id')}", f"tool_calls={calls}",
              f"content_chars={len(str(item.get('content', '')))}",
              f"reasoning_chars={len(item.get('reasoning_content', ''))}")


if __name__ == "__main__":
    main()
