"""
核心模块

包含多模型协作的核心组件
"""

from .collaborator import MultiModelCollaborator
from .adapters import TaskAwareAdapter, HiddenStateAdapter
from .projector import SemanticProjector
from .processor import CentralProcessingLayer

__all__ = [
    "MultiModelCollaborator",
    "TaskAwareAdapter", 
    "HiddenStateAdapter",
    "SemanticProjector",
    "CentralProcessingLayer"
]
