"""
任务感知多模型协作框架

这是一个支持任务感知的多模型协作系统，能够根据不同的任务类型
动态调整模型间的协作策略和Hidden State适配方式。

主要功能:
- 多模型Hidden State协作
- 任务类型自动检测
- 任务感知适配器
- 对齐训练和评估
"""

from .core.collaborator import MultiModelCollaborator
from .core.adapters import TaskAwareAdapter, HiddenStateAdapter
from .core.projector import SemanticProjector
from .core.processor import CentralProcessingLayer
from .tasks.detector import TaskDetector, TaskType
from .training.alignment_trainer import AlignmentTrainer
from .training.task_aware_trainer import TaskAwareTrainer
from .utils.evaluator import AlignmentEvaluator
from .utils.tokenizer import UnifiedTokenizer

__version__ = "1.0.0"
__author__ = "Task-Aware Collaboration Team"

__all__ = [
    "MultiModelCollaborator",
    "TaskAwareAdapter", 
    "HiddenStateAdapter",
    "SemanticProjector",
    "CentralProcessingLayer",
    "TaskDetector",
    "TaskType",
    "AlignmentTrainer",
    "TaskAwareTrainer", 
    "AlignmentEvaluator",
    "UnifiedTokenizer"
]
