"""
语义投影器模块

将不同模型的Hidden State投影到共享语义空间
"""

import torch
import torch.nn as nn


class SemanticProjector(nn.Module):
    """将不同模型的Hidden State投影到共享语义空间"""
    def __init__(self, model_dims, shared_dim):
        super().__init__()
        
        # 确保参数有效
        if shared_dim is None or shared_dim <= 0:
            shared_dim = max(model_dims) if model_dims else 768
        
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, shared_dim),
                nn.LayerNorm(shared_dim),
                nn.GELU(),
                nn.Linear(shared_dim, shared_dim),
                nn.LayerNorm(shared_dim)
            )
            for dim in model_dims
        ])
        
        self.shared_dim = shared_dim
        self.model_dims = model_dims
    
    def forward(self, hidden_states):
        """投影多个模型的hidden states到共享空间"""
        return [proj(hs) for proj, hs in zip(self.projections, hidden_states)]
    
    def project_single(self, hidden_state, model_idx):
        """投影单个模型的hidden state"""
        if model_idx >= len(self.projections):
            raise ValueError(f"Model index {model_idx} out of range. Available models: {len(self.projections)}")
        return self.projections[model_idx](hidden_state)
    
    def get_projection_for_model(self, model_idx):
        """获取特定模型的投影层"""
        if model_idx >= len(self.projections):
            raise ValueError(f"Model index {model_idx} out of range. Available models: {len(self.projections)}")
        return self.projections[model_idx]
    
    def add_model_projection(self, model_dim):
        """为新模型添加投影层"""
        new_projection = nn.Sequential(
            nn.Linear(model_dim, self.shared_dim),
            nn.LayerNorm(self.shared_dim),
            nn.GELU(),
            nn.Linear(self.shared_dim, self.shared_dim),
            nn.LayerNorm(self.shared_dim)
        )
        self.projections.append(new_projection)
        self.model_dims.append(model_dim)
        
        return len(self.projections) - 1  # 返回新模型的索引
