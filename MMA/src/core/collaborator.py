"""
多模型协作器模块

核心的多模型协作系统
"""

import torch
from transformers import AutoTokenizer
from typing import List, Optional

from .processor import CentralProcessingLayer
from ..utils.tokenizer import UnifiedTokenizer


class MultiModelCollaborator:
    """多模型协作系统"""
    def __init__(self, models, tokenizers=None, shared_dim=None):
        self.models = models
        self.model_dims = [model.config.hidden_size for model in models]
        
        # 如果没有指定shared_dim，使用最大的模型维度
        if shared_dim is None:
            self.shared_dim = max(self.model_dims)
        else:
            self.shared_dim = shared_dim
        
        # 初始化中心处理层
        self.central_processor = CentralProcessingLayer(self.model_dims, self.shared_dim)
        
        # 初始化统一Tokenizer（如果未提供）
        if tokenizers is None:
            self.tokenizers = []
            for model in models:
                tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
                # 为没有padding token的tokenizer设置padding token
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                self.tokenizers.append(tokenizer)
            self.unified_tokenizer = UnifiedTokenizer(self.tokenizers[0])
        else:
            self.tokenizers = tokenizers
            self.unified_tokenizer = UnifiedTokenizer(tokenizers[0])
    
    def get_hidden_states(self, text, model_idx):
        """获取特定模型的Hidden State"""
        if model_idx >= len(self.models):
            raise ValueError(f"Model index {model_idx} out of range. Available models: {len(self.models)}")
            
        inputs = self.tokenizers[model_idx].encode_plus(
            text, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = self.models[model_idx](**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1]  # 最后一层的Hidden State
    
    def collaborate(self, text, source_model_idx, target_model_idx, task_type=None):
        """任务感知的模型间协作：从source_model获取信息，传递给target_model"""
        if source_model_idx >= len(self.models) or target_model_idx >= len(self.models):
            raise ValueError("Model indices out of range")
            
        # 获取源模型的Hidden State
        source_hidden = self.get_hidden_states(text, source_model_idx)
        
        # 通过任务感知的中心处理层进行转换
        adapted_hidden = self.central_processor.process(
            [source_hidden], 
            model_indices=(source_model_idx, target_model_idx),
            text=text,
            task_type=task_type
        )
        
        # 获取目标模型的正常输出进行比较
        target_inputs = self.tokenizers[target_model_idx].encode_plus(
            text, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            # 正常的目标模型输出
            normal_outputs = self.models[target_model_idx](**target_inputs, output_hidden_states=True)
            
        return {
            'adapted_hidden': adapted_hidden,
            'normal_outputs': normal_outputs,
            'normal_hidden': normal_outputs.hidden_states[-1],
            'source_hidden': source_hidden,
            'task_type': task_type
        }
    
    def multi_model_collaborate(self, text, task_type=None):
        """多模型协作：获取所有模型的Hidden State并进行协作"""
        all_hidden_states = []
        for i, model in enumerate(self.models):
            hidden_state = self.get_hidden_states(text, i)
            all_hidden_states.append(hidden_state)
        
        # 通过中心处理层投影到共享空间
        projected_states = self.central_processor.process(
            all_hidden_states, 
            text=text, 
            task_type=task_type
        )
        
        return {
            'original_hidden_states': all_hidden_states,
            'projected_states': projected_states,
            'task_type': task_type,
            'shared_dim': self.shared_dim
        }
    
    def get_model_count(self):
        """获取模型数量"""
        return len(self.models)
    
    def get_model_dims(self):
        """获取所有模型的hidden dimensions"""
        return self.model_dims.copy()
    
    def get_shared_dim(self):
        """获取共享维度"""
        return self.shared_dim
    
    def add_model(self, model, tokenizer=None):
        """添加新模型到协作系统"""
        self.models.append(model)
        model_dim = model.config.hidden_size
        self.model_dims.append(model_dim)
        
        # 添加tokenizer
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        self.tokenizers.append(tokenizer)
        
        # 为新模型添加投影层
        model_idx = self.central_processor.semantic_projector.add_model_projection(model_dim)
        
        return model_idx
    
    def detect_task_for_text(self, text):
        """为文本检测任务类型"""
        return self.central_processor.task_detector.detect_task(text)
    
    def get_task_confidence(self, text):
        """获取任务检测置信度"""
        return self.central_processor.get_task_detection_confidence(text)
    
    def get_supported_tasks(self):
        """获取支持的任务类型"""
        return self.central_processor.get_supported_tasks()
    
    def set_device(self, device):
        """设置设备（CPU/GPU）"""
        for model in self.models:
            model.to(device)
        self.central_processor.to(device)
        
    def train_mode(self):
        """设置为训练模式"""
        self.central_processor.train()
        
    def eval_mode(self):
        """设置为评估模式"""
        self.central_processor.eval()
