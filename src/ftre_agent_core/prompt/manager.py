"""
PromptManager - 提示词管理器

支持：
- hardcode 字符串提示词注册
- 从 .md / .txt 文件加载提示词
- 提示词变量渲染（{{ variable }} 语法）
- 按 key 获取、列出所有已注册提示词

用法：
    pm = PromptManager()

    # 注册 hardcode 提示词
    pm.register("greeting", "你好，{{ name }}！")

    # 从文件加载
    pm.load_file("react_system", "core/prompt/templates/react_system.md")

    # 渲染变量
    pm.render("greeting", name="世界")  # → "你好，世界！"

    # 直接获取原始文本
    pm.get("react_system")
"""
import re
from pathlib import Path


class PromptNotFoundError(KeyError):
    pass


class PromptManager:
    """
    提示词管理器

    统一管理项目中所有提示词，支持变量渲染。
    变量语法：{{ variable_name }}
    """

    def __init__(self):
        self._prompts: dict[str, str] = {}

    # ============================================================
    # 注册 / 加载
    # ============================================================

    def register(self, key: str, text: str) -> None:
        """注册一条 hardcode 提示词"""
        self._prompts[key] = text

    def load_file(self, key: str, path: str | Path) -> None:
        """
        从文件加载提示词并注册。

        支持 .md / .txt 等任意文本文件。

        Args:
            key:  提示词标识
            path: 文件路径（相对或绝对）
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Prompt file not found: {p}")
        self._prompts[key] = p.read_text(encoding="utf-8")

    def load_dir(self, directory: str | Path, ext: tuple[str, ...] = (".md", ".txt")) -> None:
        """
        批量加载目录下所有指定扩展名的文件。

        key 为文件名（不含扩展名），例如 react_system.md → key="react_system"

        Args:
            directory: 目录路径
            ext:       要加载的文件扩展名
        """
        d = Path(directory)
        if not d.is_dir():
            raise NotADirectoryError(f"Not a directory: {d}")
        for f in d.iterdir():
            if f.suffix in ext:
                self._prompts[f.stem] = f.read_text(encoding="utf-8")

    # ============================================================
    # 获取 / 渲染
    # ============================================================

    def get(self, key: str) -> str:
        """获取原始提示词文本（不渲染变量）"""
        if key not in self._prompts:
            raise PromptNotFoundError(f"Prompt '{key}' not found")
        return self._prompts[key]

    def render(self, key: str, **variables) -> str:
        """
        获取提示词并渲染变量。

        变量语法：{{ variable_name }}
        未提供的变量保持原样不替换。

        Args:
            key:       提示词标识
            **variables: 变量键值对

        Returns:
            渲染后的提示词字符串
        """
        text = self.get(key)
        return self._render(text, variables)

    def render_text(self, text: str, **variables) -> str:
        """直接渲染任意文本中的变量，不需要先注册"""
        return self._render(text, variables)

    # ============================================================
    # 管理
    # ============================================================

    def has(self, key: str) -> bool:
        return key in self._prompts

    def unregister(self, key: str) -> None:
        self._prompts.pop(key, None)

    @property
    def keys(self) -> list[str]:
        return list(self._prompts.keys())

    def __len__(self) -> int:
        return len(self._prompts)

    def __repr__(self) -> str:
        return f"PromptManager({self.keys})"

    # ============================================================
    # 内部
    # ============================================================

    @staticmethod
    def _render(text: str, variables: dict) -> str:
        """将 {{ key }} 替换为对应变量值，未提供的变量保持原样"""
        def replacer(match: re.Match) -> str:
            var = match.group(1).strip()
            return str(variables[var]) if var in variables else match.group(0)

        return re.sub(r"\{\{\s*(\w+)\s*\}\}", replacer, text)