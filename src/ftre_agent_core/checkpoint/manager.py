"""
CheckpointManager - 快照管理器

管理快照的生命周期：创建、恢复、查询、删除。
支持线性链和未来的分支树结构。

设计说明：
- CheckpointManager 只负责"存储和检索"快照，不关心快照里存了什么
- "状态的收集和恢复"由调用方（如 MemoryManager）负责
- restore 时自动截断后续快照（时间线分叉），为 Branching 做准备
"""
import copy
from .model import Checkpoint, CheckpointType


class CheckpointManager:
    """
    快照管理器

    职责：
    - save:    创建并保存快照
    - restore: 恢复到指定快照（返回深拷贝，截断后续快照）
    - list:    列出快照（支持按类型过滤）
    - get:     获取单个快照
    - delete:  删除指定快照
    - latest:  获取最新快照
    """

    def __init__(self, max_checkpoints: int = 50):
        self._checkpoints: list[Checkpoint] = []
        self._max_checkpoints = max_checkpoints

    def save(
        self,
        turn: int,
        state: dict,
        type: CheckpointType = CheckpointType.AUTO,
        label: str = "",
        metadata: dict | None = None,
    ) -> Checkpoint:
        """
        创建并保存一个快照

        Args:
            turn:     对话轮次
            state:    状态字典（会被深拷贝）
            type:     快照类型
            label:    可选标签
            metadata: 可选元数据

        Returns:
            新创建的 Checkpoint
        """
        # 自动设置 parent_id 为当前最新快照
        parent_id = self._checkpoints[-1].id if self._checkpoints else None

        cp = Checkpoint.create(
            turn=turn,
            state=state,
            type=type,
            parent_id=parent_id,
            label=label,
            metadata=metadata,
        )
        self._checkpoints.append(cp)

        # 超限清理最早的
        while len(self._checkpoints) > self._max_checkpoints:
            self._checkpoints.pop(0)

        return cp

    def restore(self, checkpoint_id: str) -> Checkpoint:
        """
        恢复到指定快照

        返回该快照 state 的深拷贝（避免后续操作污染原始快照），
        同时截断该快照之后的所有快照（时间线分叉）。

        Args:
            checkpoint_id: 快照 ID

        Returns:
            快照的深拷贝

        Raises:
            ValueError: 快照不存在
        """
        index = self._find_index(checkpoint_id)
        if index is None:
            raise ValueError(f"Checkpoint '{checkpoint_id}' not found")

        target = self._checkpoints[index]

        # 截断：保留该快照及之前的
        self._checkpoints = self._checkpoints[:index + 1]

        # 返回深拷贝，防止调用方修改影响存档
        return Checkpoint(
            id=target.id,
            timestamp=target.timestamp,
            turn=target.turn,
            state=copy.deepcopy(target.state),
            type=target.type,
            parent_id=target.parent_id,
            label=target.label,
            metadata=copy.deepcopy(target.metadata),
        )

    def get(self, checkpoint_id: str) -> Checkpoint | None:
        """获取指定快照（不修改任何状态）"""
        for cp in self._checkpoints:
            if cp.id == checkpoint_id:
                return cp
        return None

    def list(self, type: CheckpointType | None = None) -> list[Checkpoint]:
        """
        列出快照

        Args:
            type: 可选，按类型过滤（None 表示全部）
        """
        if type is None:
            return list(self._checkpoints)
        return [cp for cp in self._checkpoints if cp.type == type]

    def latest(self) -> Checkpoint | None:
        """获取最新的快照"""
        return self._checkpoints[-1] if self._checkpoints else None

    def delete(self, checkpoint_id: str) -> bool:
        """删除指定快照"""
        index = self._find_index(checkpoint_id)
        if index is not None:
            self._checkpoints.pop(index)
            return True
        return False

    def clear(self) -> None:
        """清空所有快照"""
        self._checkpoints.clear()

    def _find_index(self, checkpoint_id: str) -> int | None:
        """查找快照索引"""
        for i, cp in enumerate(self._checkpoints):
            if cp.id == checkpoint_id:
                return i
        return None

    @property
    def count(self) -> int:
        return len(self._checkpoints)

    def __len__(self) -> int:
        return len(self._checkpoints)

    def __repr__(self) -> str:
        return f"CheckpointManager(count={self.count})"