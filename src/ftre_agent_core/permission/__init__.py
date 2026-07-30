# -*- coding: utf-8 -*-
"""权限决策模型与纯 PermissionEngine。"""
from ._types import (
    PermissionBehavior,
    PermissionRequest,
    PermissionRule,
    PermissionDecision,
)
from ._engine import PermissionEngine

__all__ = [
    "PermissionBehavior",
    "PermissionRequest",
    "PermissionRule",
    "PermissionDecision",
    "PermissionEngine",
]
