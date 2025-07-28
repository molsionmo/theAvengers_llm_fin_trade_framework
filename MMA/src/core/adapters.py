"""
适配器模块

包含任务感知适配器和通用Hidden State适配器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from ..tasks.detector import TaskType


class HiddenStateAdapter(nn.Module):
    """适配器层，使模型能理解其他模型的Hidden State"""
    def __init__(self, source_dim, target_dim):
        super().__init__()
        
        # 确保维度参数有效
        if source_dim is None or source_dim <= 0:
            source_dim = 768
        if target_dim is None or target_dim <= 0:
            target_dim = 768
            
        self.adapter = nn.Sequential(
            nn.Linear(source_dim, target_dim),
            nn.GELU(),
            nn.Linear(target_dim, target_dim),
            nn.LayerNorm(target_dim)
        )
    
    def forward(self, hidden_state):
        return self.adapter(hidden_state)


class TaskAwareAdapter(nn.Module):
    """任务感知适配器，根据任务类型调整Hidden State转换"""
    
    def __init__(self, source_dim: int, target_dim: int, task_types: List[TaskType]):
        super().__init__()
        
        # 确保维度参数有效
        if source_dim is None or source_dim <= 0:
            source_dim = 768
        if target_dim is None or target_dim <= 0:
            target_dim = 768
            
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.task_types = task_types
        
        # 为每种任务类型创建专门的适配器
        self.task_adapters = nn.ModuleDict({
            task_type.value: nn.Sequential(
                nn.Linear(source_dim, target_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(target_dim, target_dim),
                nn.LayerNorm(target_dim)
            )
            for task_type in task_types
        })
        
        # 通用适配器作为后备
        self.general_adapter = nn.Sequential(
            nn.Linear(source_dim, target_dim),
            nn.GELU(),
            nn.Linear(target_dim, target_dim),
            nn.LayerNorm(target_dim)
        )
        
        # 任务权重网络，用于混合不同任务的适配结果
        self.task_weight_network = nn.Sequential(
            nn.Linear(source_dim, len(task_types)),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, hidden_state: torch.Tensor, task_type: Optional[TaskType] = None) -> torch.Tensor:
        """前向传播，根据任务类型进行适配"""
        if task_type and task_type.value in self.task_adapters:
            # 使用特定任务的适配器
            return self.task_adapters[task_type.value](hidden_state)
        elif task_type is None:
            # 自动混合多个任务适配器
            task_weights = self.task_weight_network(hidden_state.mean(dim=1))  # [batch_size, num_tasks]
            
            adapted_outputs = []
            for i, task_type in enumerate(self.task_types):
                if task_type.value in self.task_adapters:
                    adapted = self.task_adapters[task_type.value](hidden_state)
                    adapted_outputs.append(adapted)
            
            if adapted_outputs:
                # 加权组合不同任务的输出
                stacked_outputs = torch.stack(adapted_outputs, dim=-1)  # [batch, seq, dim, num_tasks]
                weighted_output = torch.sum(stacked_outputs * task_weights.unsqueeze(1).unsqueeze(2), dim=-1)
                return weighted_output
            else:
                return self.general_adapter(hidden_state)
        else:
            # 使用通用适配器
            return self.general_adapter(hidden_state)
    
    def get_task_specific_adapter(self, task_type: TaskType):
        """获取特定任务类型的适配器"""
        if task_type.value in self.task_adapters:
            return self.task_adapters[task_type.value]
        return self.general_adapter
    
    def add_task_adapter(self, task_type: TaskType):
        """为新任务类型添加适配器"""
        if task_type.value not in self.task_adapters:
            self.task_adapters[task_type.value] = nn.Sequential(
                nn.Linear(self.source_dim, self.target_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.target_dim, self.target_dim),
                nn.LayerNorm(self.target_dim)
            )
            self.task_types.append(task_type)
            
            # 更新任务权重网络
            old_weight_network = self.task_weight_network[0]
            new_weight_network = nn.Linear(self.source_dim, len(self.task_types))
            
            # 复制旧权重（如果维度兼容）
            with torch.no_grad():
                if old_weight_network.weight.size(0) < new_weight_network.weight.size(0):
                    new_weight_network.weight[:old_weight_network.weight.size(0)] = old_weight_network.weight
                    new_weight_network.bias[:old_weight_network.bias.size(0)] = old_weight_network.bias
            
            self.task_weight_network = nn.Sequential(
                new_weight_network,
                nn.Softmax(dim=-1)
            )
