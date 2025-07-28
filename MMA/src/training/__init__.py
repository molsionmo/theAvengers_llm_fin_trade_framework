"""
训练模块

包含对齐训练器和任务感知训练器
"""

from .alignment_trainer import AlignmentTrainer
from .task_aware_trainer import TaskAwareTrainer

__all__ = ["AlignmentTrainer", "TaskAwareTrainer"]
