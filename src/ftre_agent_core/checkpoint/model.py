"""
Checkpoint - 对话状态快照

设计原则：
- Checkpoint 本身只是一个"状态容器"，不关心具体存了什么业务数据
- 业务状态统一放在 state 字典中，由各模块自行定义 key
- 这样新增状态字段时不需要修改 Checkpoint 类本身

state 的约定 key（按模块划分）：
    - "messages":     list[dict]  — 对话消息列表（不含 system prompt）
    - "token_usage":  dict        — token 统计快照
    - "runner":       dict        — runner 执行状态（iteration, pending_tool_calls 等）
    - 未来可扩展更多 key...

典型用法：
    cp = Checkpoint.create(
        turn=3,
        state={"messages": [...], "token_usage": {...}},
        label="查询天气",
    )
    cp.state["messages"]  # 获取消息快照
"""
import copy
import time
import uuid
from enum import Enum
from dataclasses import dataclass, field
from typing import Any


class CheckpointType(str, Enum):
    """
    快照类型，区分快照的产生方式

    AUTO     — 每轮对话结束后自动保存
    MANUAL   — 用户主动保存（"存档"）
    INTERRUPT — 中断时保存（Interrupt/Resume 场景）
    """
    AUTO = "auto"
    MANUAL = "manual"
    INTERRUPT = "interrupt"


@dataclass
class Checkpoint:
    """
    一个对话状态快照

    核心字段：
        id        — 唯一标识
        turn      — 对话轮次（第几轮用户交互后产生）
        state     — 状态字典，存放所有业务数据（messages, token_usage, runner 等）
        type      — 快照类型（auto/manual/interrupt）
        parent_id — 父快照 ID（用于 Branching，构建快照树）

    辅助字段：
        timestamp — 创建时间戳
        label     — 可选标签（方便用户识别）
        metadata  — 可扩展的元数据字典（存放非核心信息）
    """
    # 唯一标识
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # 创建时间
    timestamp: float = field(default_factory=time.time)

    # 对话轮次
    turn: int = 0

    # === 核心：可扩展的状态字典 ===
    state: dict[str, Any] = field(default_factory=dict)

    # 快照类型
    type: CheckpointType = CheckpointType.AUTO

    # 父快照 ID（Branching 用，None 表示根节点或线性链）
    parent_id: str | None = None

    # 可选标签
    label: str = ""

    # 可扩展元数据（非核心信息，如触发原因、备注等）
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        turn: int,
        state: dict[str, Any],
        type: CheckpointType = CheckpointType.AUTO,
        parent_id: str | None = None,
        label: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "Checkpoint":
        """
        创建快照（自动深拷贝 state）

        Args:
            turn:      对话轮次
            state:     状态字典，会被深拷贝
            type:      快照类型
            parent_id: 父快照 ID
            label:     可选标签
            metadata:  可选元数据
        """
        return cls(
            turn=turn,
            state=copy.deepcopy(state),
            type=type,
            parent_id=parent_id,
            label=label,
            metadata=metadata or {},
        )

    # ============================================================
    # 状态访问便捷方法
    # ============================================================

    def get_state(self, key: str, default: Any = None) -> Any:
        """获取 state 中的某个 key"""
        return self.state.get(key, default)

    def has_state(self, key: str) -> bool:
        """检查 state 中是否有某个 key"""
        return key in self.state

    # ============================================================
    # 序列化
    # ============================================================

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典（用于持久化存储）"""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "turn": self.turn,
            "state": self.state,
            "type": self.type.value,
            "parent_id": self.parent_id,
            "label": self.label,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        """从字典反序列化"""
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            turn=data.get("turn", 0),
            state=data.get("state", {}),
            type=CheckpointType(data.get("type", "auto")),
            parent_id=data.get("parent_id"),
            label=data.get("label", ""),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        state_keys = list(self.state.keys())
        return (
            f"Checkpoint(id={self.id!r}, turn={self.turn}, "
            f"type={self.type.value}, state_keys={state_keys})"
        )