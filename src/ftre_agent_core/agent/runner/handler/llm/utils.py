"""
LLM 模块工具函数
"""
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

_DEBUG_LLM_INPUT_DIR = Path("data/logs/llm_input")


def dump_llm_input(messages: list[dict], tools: list[dict] | None, model: str) -> None:
    """记录 LLM 调用输入（用于调试）"""
    try:
        now = datetime.now()
        day_dir = _DEBUG_LLM_INPUT_DIR / now.strftime("%Y%m%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        file_path = day_dir / f"{now.strftime('%H%M%S_%f')}_{uuid4().hex[:8]}.json"
        payload = {
            "timestamp": now.isoformat(),
            "model": model,
            "messages": messages,
            "tools": tools,
        }
        file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass
