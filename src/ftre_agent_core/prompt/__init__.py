from .manager import PromptManager, PromptNotFoundError
from pathlib import Path

__all__ = ["PromptManager", "PromptNotFoundError", "prompts"]

_TEMPLATES_DIR = Path(__file__).parent / "templates"

prompts = PromptManager()
if _TEMPLATES_DIR.is_dir():
    prompts.load_dir(_TEMPLATES_DIR)
