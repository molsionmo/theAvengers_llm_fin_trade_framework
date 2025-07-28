"""
中心处理器模块

任务感知的中心处理层，统一处理和分发不同模型的Hidden State
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from .projector import SemanticProjector
from .adapters import HiddenStateAdapter, TaskAwareAdapter
from ..tasks.detector import TaskDetector, TaskType


class CentralProcessingLayer(nn.Module):
    """任务感知的中心处理层，统一处理和分发不同模型的Hidden State"""
    def __init__(self, model_dims, shared_dim, task_types=None):
        super().__init__()
        
        # 确保shared_dim是有效值
        if shared_dim is None or shared_dim <= 0:
            shared_dim = max(model_dims) if model_dims else 768
        
        self.token_mapper = None  # 可选的Token映射器
        self.semantic_projector = SemanticProjector(model_dims, shared_dim)
        self.adapters = nn.ModuleDict()  # 动态生成的适配器
        self.shared_dim = shared_dim  # 保存shared_dim以便在适配器中使用
        
        # 任务检测器
        self.task_detector = TaskDetector()
        
        # 支持的任务类型
        if task_types is None:
            self.task_types = [TaskType.QUESTION_ANSWERING, TaskType.TEXT_CLASSIFICATION, 
                              TaskType.SENTIMENT_ANALYSIS, TaskType.TEXT_GENERATION, TaskType.GENERAL]
        else:
            self.task_types = task_types
    
    def register_token_mapper(self, token_mapper):
        """注册Token映射器"""
        self.token_mapper = token_mapper
    
    def get_adapter(self, source_model_idx, target_model_idx, task_type=None):
        """获取或创建从source到target的任务感知适配器"""
        if task_type:
            key = f"{source_model_idx}_{target_model_idx}_{task_type.value}"
        else:
            key = f"{source_model_idx}_{target_model_idx}"
            
        if key not in self.adapters:
            if task_type:
                # 创建任务感知适配器
                self.adapters[key] = TaskAwareAdapter(self.shared_dim, self.shared_dim, self.task_types)
            else:
                # 创建通用适配器
                self.adapters[key] = HiddenStateAdapter(self.shared_dim, self.shared_dim)
        return self.adapters[key]
    
    def process(self, hidden_states, model_indices=None, text=None, task_type=None):
        """处理并分发Hidden State，支持任务感知"""
        # 投影到共享空间
        projected_states = self.semantic_projector(hidden_states)
        
        # 如果提供了文本但没有任务类型，自动检测任务类型
        if text and task_type is None:
            task_type = self.task_detector.detect_task(text)
        
        # 如果需要特定模型的适配版本
        if model_indices is not None:
            source_idx, target_idx = model_indices
            adapter = self.get_adapter(source_idx, target_idx, task_type)
            
            # 如果是任务感知适配器，传递任务类型
            if isinstance(adapter, TaskAwareAdapter):
                return adapter(projected_states[source_idx], task_type)
            else:
                return adapter(projected_states[source_idx])
        
        return projected_states
    
    def get_task_detection_confidence(self, text: str):
        """获取任务检测的置信度分数"""
        return self.task_detector.get_task_confidence(text)
    
    def add_task_type(self, task_type: TaskType):
        """添加新的任务类型支持"""
        if task_type not in self.task_types:
            self.task_types.append(task_type)
            
            # 为现有的任务感知适配器添加新任务支持
            for key, adapter in self.adapters.items():
                if isinstance(adapter, TaskAwareAdapter):
                    adapter.add_task_adapter(task_type)
    
    def get_adapter_keys(self):
        """获取所有适配器的键"""
        return list(self.adapters.keys())
    
    def get_supported_tasks(self):
        """获取支持的任务类型"""
        return self.task_types.copy()
    
    def set_task_detection_patterns(self, task_type: TaskType, patterns: List[str]):
        """设置特定任务类型的检测模式"""
        # 清除现有模式
        self.task_detector.patterns[task_type] = patterns
    
    def add_task_detection_pattern(self, task_type: TaskType, pattern: str):
        """添加任务检测模式"""
        self.task_detector.add_pattern(task_type, pattern)
