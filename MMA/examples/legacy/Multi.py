import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import re


class TaskType(Enum):
    """任务类型枚举"""
    QUESTION_ANSWERING = "qa"
    TEXT_CLASSIFICATION = "classification"
    SENTIMENT_ANALYSIS = "sentiment"
    NAMED_ENTITY_RECOGNITION = "ner"
    TEXT_GENERATION = "generation"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CONVERSATION = "conversation"
    GENERAL = "general"


class TaskDetector:
    """任务类型检测器，基于文本特征自动检测任务类型"""
    
    def __init__(self):
        self.patterns = {
            TaskType.QUESTION_ANSWERING: [
                r"\bwhat\b", r"\bhow\b", r"\bwhy\b", r"\bwhen\b", r"\bwhere\b", r"\bwho\b",
                r"\?\s*$", r"\bexplain\b", r"\bdescribe\b", r"\bdefine\b", r"\btell me\b"
            ],
            TaskType.SENTIMENT_ANALYSIS: [
                r"\blove\b", r"\bhate\b", r"\blike\b", r"\bdislike\b", r"\bfeel\b", 
                r"\bopinion\b", r"\bthink\b", r"\bemotion\b", r"\bmood\b",
                r"\bpositive\b", r"\bnegative\b", r"\bhappy\b", r"\bsad\b",
                r"\bawesome\b", r"\bterrible\b", r"\bgreat\b", r"\bbad\b",
                r"\bwonderful\b", r"\bawful\b", r"\bamazing\b", r"\bhorrible\b"
            ],
            TaskType.TEXT_GENERATION: [
                r"\bgenerate\b", r"\bcreate\b", r"\bwrite\b", r"\bcontinue\b", r"\bcomplete\b",
                r"\bstory\b", r"\bpoem\b", r"\barticle\b", r"\bcompose\b"
            ],
            TaskType.SUMMARIZATION: [
                r"\bsummarize\b", r"\bsummary\b", r"\bmain points\b", r"\bkey points\b"
            ],
            TaskType.TRANSLATION: [
                r"\btranslate\b", r"\btranslation\b", r"\bin [a-z]+ language\b"
            ],
            TaskType.CONVERSATION: [
                r"\bhello\b", r"\bhi\b", r"\bhow are you\b", r"\bgood morning\b", r"\bgood evening\b",
                r"\bnice to meet\b", r"\bbye\b", r"\bgoodbye\b", r"\bsee you\b"
            ]
        }
    
    def detect_task(self, text: str) -> TaskType:
        """检测文本对应的任务类型"""
        text_lower = text.lower()
        
        task_scores = {}
        for task_type, patterns in self.patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, text_lower))
            if score > 0:
                task_scores[task_type] = score
        
        if task_scores:
            return max(task_scores.items(), key=lambda x: x[1])[0]
        return TaskType.GENERAL


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


class UnifiedTokenizer:
    """统一Tokenizer，为所有模型提供一致的编码接口"""
    def __init__(self, base_tokenizer):
        self.tokenizer = base_tokenizer
    
    def encode(self, text, model_name=None):
        return self.tokenizer.encode(text)
    
    def decode(self, tokens, model_name=None):
        return self.tokenizer.decode(tokens)


class TokenMapper(nn.Module):
    """学习将不同模型的Token映射到共享空间"""
    def __init__(self, tokenizers, shared_vocab_size):
        super().__init__()
        self.mappings = nn.ModuleList([
            nn.Embedding(tok.vocab_size, shared_vocab_size) 
            for tok in tokenizers
        ])
    
    def forward(self, token_ids, model_idx):
        return self.mappings[model_idx](token_ids)


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
    
    def forward(self, hidden_states):
        return [proj(hs) for proj, hs in zip(self.projections, hidden_states)]


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
        inputs = self.tokenizers[model_idx].encode_plus(
            text, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = self.models[model_idx](**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1]  # 最后一层的Hidden State
    
    def collaborate(self, text, source_model_idx, target_model_idx, task_type=None):
        """任务感知的模型间协作：从source_model获取信息，传递给target_model"""
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
            
            # 这里我们返回适配后的hidden states和正常的模型输出进行比较
            # 在实际应用中，您可能想要将adapted_hidden进一步处理或用于特定任务
            
        return {
            'adapted_hidden': adapted_hidden,
            'normal_outputs': normal_outputs,
            'normal_hidden': normal_outputs.hidden_states[-1]
        }


class AlignmentEvaluator:
    """Hidden State对齐评估器"""
    def __init__(self, metrics=['cosine', 'mmd', 'tsne']):
        self.metrics = metrics
    
    def cosine_similarity(self, hidden_states_1, hidden_states_2):
        """计算余弦相似度"""
        # 确保两个hidden states有相同的序列长度
        min_len = min(hidden_states_1.size(1), hidden_states_2.size(1))
        hidden_states_1 = hidden_states_1[:, :min_len, :]
        hidden_states_2 = hidden_states_2[:, :min_len, :]
        
        # 如果hidden states的特征维度不同，使用平均池化
        if hidden_states_1.size(-1) != hidden_states_2.size(-1):
            # 使用平均池化来标准化特征维度
            hidden_states_1 = hidden_states_1.mean(dim=-1, keepdim=True)
            hidden_states_2 = hidden_states_2.mean(dim=-1, keepdim=True)
        
        sim_matrix = F.cosine_similarity(hidden_states_1, hidden_states_2, dim=-1)
        return sim_matrix.mean().item()
    
    def mmd_loss(self, x, y, kernel='rbf'):
        """计算MMD损失评估分布匹配程度"""
        # 确保两个张量有相同的特征维度
        if x.size(-1) != y.size(-1):
            # 如果维度不同，将它们投影到较小的维度
            min_dim = min(x.size(-1), y.size(-1))
            x = x[:, :min_dim]
            y = y[:, :min_dim]
        
        # 确保有足够的样本
        min_samples = min(x.size(0), y.size(0))
        if min_samples < 2:
            return torch.tensor(0.0)
        
        x = x[:min_samples]
        y = y[:min_samples]
        
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
        
        dxx = rx.t() + rx - 2. * xx
        dyy = ry.t() + ry - 2. * yy
        dxy = rx.t() + ry - 2. * zz
        
        gamma = 1.0 / x.size(1) if x.size(1) > 0 else 1.0
        XX = torch.exp(-gamma * dxx)
        YY = torch.exp(-gamma * dyy)
        XY = torch.exp(-gamma * dxy)
        
        return torch.mean(XX + YY - 2. * XY)
    
    def visualize_tsne(self, hidden_states_1, hidden_states_2, labels=None, save_path=None):
        """使用t-SNE可视化对齐效果"""
        combined = torch.cat([hidden_states_1, hidden_states_2], dim=0).cpu().numpy()
        
        tsne = TSNE(n_components=2, random_state=42)
        reduced = tsne.fit_transform(combined)
        
        n1 = hidden_states_1.shape[0]
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced[:n1, 0], reduced[:n1, 1], c='blue', label='Model 1')
        plt.scatter(reduced[n1:, 0], reduced[n1:, 1], c='red', label='Model 2')
        
        if labels is not None:
            pass  # 可根据标签进一步着色
        
        plt.legend()
        plt.title('Hidden State Alignment Visualization')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def evaluate(self, model_A, model_B, dataset, central_processor=None):
        """评估两个模型之间的Hidden State对齐效果"""
        results = {}
        
        for text, labels in dataset:
            # 为两个模型准备输入
            tokenizer_A = AutoTokenizer.from_pretrained(model_A.config._name_or_path)
            tokenizer_B = AutoTokenizer.from_pretrained(model_B.config._name_or_path)
            
            # 设置padding token
            if tokenizer_A.pad_token is None:
                tokenizer_A.pad_token = tokenizer_A.eos_token
            if tokenizer_B.pad_token is None:
                tokenizer_B.pad_token = tokenizer_B.eos_token
            
            inputs_A = tokenizer_A.encode_plus(text, return_tensors="pt", padding=True, truncation=True)
            inputs_B = tokenizer_B.encode_plus(text, return_tensors="pt", padding=True, truncation=True)
            
            # 获取原始Hidden State
            with torch.no_grad():
                hidden_A = model_A(**inputs_A, output_hidden_states=True).hidden_states[-1]
                hidden_B = model_B(**inputs_B, output_hidden_states=True).hidden_states[-1]
            
            # 如果提供了中心处理层，使用对齐后的Hidden State
            if central_processor:
                projected_states = central_processor.process([hidden_A, hidden_B])
                hidden_A, hidden_B = projected_states
            
            # 计算各项指标
            if 'cosine' in self.metrics:
                results['cosine'] = self.cosine_similarity(hidden_A, hidden_B)
            
            if 'mmd' in self.metrics:
                results['mmd'] = self.mmd_loss(hidden_A.view(-1, hidden_A.size(-1)), 
                                              hidden_B.view(-1, hidden_B.size(-1)))
            
            if 'tsne' in self.metrics and len(results) < 100:  # 限制可视化样本量
                self.visualize_tsne(hidden_A, hidden_B)
            
            # 只处理第一个样本，避免重复计算
            break
        
        return results


class AlignmentTrainer:
    """对齐训练器，用于训练中心处理层"""
    def __init__(self, collaborator, learning_rate=1e-4):
        self.collaborator = collaborator
        self.optimizer = torch.optim.Adam(collaborator.central_processor.parameters(), lr=learning_rate)
        
    def contrastive_loss(self, proj1, proj2, temperature=0.1):
        """对比学习损失函数"""
        # 确保两个投影有相同的序列长度
        min_len = min(proj1.size(1), proj2.size(1))
        proj1 = proj1[:, :min_len, :]
        proj2 = proj2[:, :min_len, :]
        
        # 计算余弦相似度
        cosine_sim = F.cosine_similarity(proj1, proj2, dim=-1)
        
        # 对比损失：希望相同文本的投影尽可能相似
        loss = 1 - cosine_sim.mean()
        
        return loss
    
    def train_epoch(self, train_dataset):
        """训练一个epoch"""
        self.collaborator.central_processor.train()
        total_loss = 0
        num_batches = 0
        
        for text in train_dataset:
            try:
                # 获取两个模型的Hidden State
                hidden1 = self.collaborator.get_hidden_states(text, 0)
                hidden2 = self.collaborator.get_hidden_states(text, 1)
                
                # 投影到共享空间
                projected_states = self.collaborator.central_processor.process([hidden1, hidden2])
                proj1, proj2 = projected_states
                
                # 计算对比损失
                loss = self.contrastive_loss(proj1, proj2)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Error processing text '{text}': {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def train(self, train_dataset, epochs=10, validation_dataset=None):
        """训练对齐器"""
        print(f"开始训练对齐器，共{epochs}个epoch...")
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_dataset)
            
            # 验证
            val_results = None
            if validation_dataset is not None:
                val_results = self.evaluate_alignment(validation_dataset)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  训练损失: {train_loss:.4f}")
            if val_results:
                print(f"  验证余弦相似度: {val_results['cosine']:.4f}")
                print(f"  验证MMD损失: {val_results['mmd']:.4f}")
            print("-" * 50)
    
    def evaluate_alignment(self, dataset):
        """评估对齐效果"""
        self.collaborator.central_processor.eval()
        evaluator = AlignmentEvaluator(metrics=['cosine', 'mmd'])
        
        # 准备评估数据集格式
        eval_dataset = [(text, torch.tensor([0])) for text in dataset[:5]]  # 只用前5个样本评估
        
        with torch.no_grad():
            results = evaluator.evaluate(
                self.collaborator.models[0], 
                self.collaborator.models[1], 
                eval_dataset, 
                self.collaborator.central_processor
            )
        
        return results


class TaskAwareTrainer(AlignmentTrainer):
    """任务感知的对齐训练器，专门针对不同任务类型优化"""
    
    def __init__(self, collaborator, learning_rate=1e-4, task_weight_decay=0.01):
        super().__init__(collaborator, learning_rate)
        self.task_weight_decay = task_weight_decay
        self.task_detector = TaskDetector()
        
        # 为不同任务类型维护独立的损失统计
        self.task_losses = {task_type: [] for task_type in TaskType}
    
    def task_specific_loss(self, proj1, proj2, task_type, temperature=0.1):
        """针对特定任务的损失函数"""
        # 基础对比损失
        base_loss = self.contrastive_loss(proj1, proj2, temperature)
        
        # 根据任务类型调整损失权重
        task_weights = {
            TaskType.QUESTION_ANSWERING: 1.2,  # 问答任务需要更强的语义对齐
            TaskType.SENTIMENT_ANALYSIS: 1.1,  # 情感分析需要情感信息保留
            TaskType.TEXT_GENERATION: 0.9,     # 生成任务允许更多创造性
            TaskType.CONVERSATION: 1.0,        # 对话任务标准权重
            TaskType.GENERAL: 1.0              # 通用任务标准权重
        }
        
        weight = task_weights.get(task_type, 1.0)
        
        # 添加任务特定的正则化
        if task_type == TaskType.QUESTION_ANSWERING:
            # 问答任务：增加信息保持损失
            info_preserve_loss = torch.mean(torch.abs(proj1.norm(dim=-1) - proj2.norm(dim=-1)))
            base_loss += 0.1 * info_preserve_loss
        elif task_type == TaskType.TEXT_GENERATION:
            # 生成任务：增加多样性奖励
            diversity_reward = -0.05 * torch.mean(torch.var(proj2, dim=1))
            base_loss += diversity_reward
        
        return weight * base_loss
    
    def train_epoch_with_tasks(self, train_dataset):
        """基于任务类型的训练epoch"""
        self.collaborator.central_processor.train()
        task_losses = {task_type: 0.0 for task_type in TaskType}
        task_counts = {task_type: 0 for task_type in TaskType}
        
        for text in train_dataset:
            try:
                # 检测任务类型
                detected_task = self.task_detector.detect_task(text)
                
                # 获取两个模型的Hidden State
                hidden1 = self.collaborator.get_hidden_states(text, 0)
                hidden2 = self.collaborator.get_hidden_states(text, 1)
                
                # 使用任务感知投影
                projected_states = self.collaborator.central_processor.process(
                    [hidden1, hidden2], 
                    text=text, 
                    task_type=detected_task
                )
                proj1, proj2 = projected_states
                
                # 计算任务特定损失
                loss = self.task_specific_loss(proj1, proj2, detected_task)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 记录损失
                task_losses[detected_task] += loss.item()
                task_counts[detected_task] += 1
                
            except Exception as e:
                print(f"Error processing text '{text}': {e}")
                continue
        
        # 计算平均损失
        avg_task_losses = {}
        for task_type in TaskType:
            if task_counts[task_type] > 0:
                avg_task_losses[task_type] = task_losses[task_type] / task_counts[task_type]
                self.task_losses[task_type].append(avg_task_losses[task_type])
        
        return avg_task_losses
    
    def train_with_task_awareness(self, train_dataset, epochs=10, validation_dataset=None):
        """任务感知训练"""
        print(f"开始任务感知训练，共{epochs}个epoch...")
        
        for epoch in range(epochs):
            # 训练
            task_losses = self.train_epoch_with_tasks(train_dataset)
            
            # 验证
            val_results = None
            if validation_dataset is not None:
                val_results = self.evaluate_alignment(validation_dataset)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print("  任务特定训练损失:")
            for task_type, loss in task_losses.items():
                print(f"    {task_type.value}: {loss:.4f}")
            
            if val_results:
                print(f"  验证余弦相似度: {val_results['cosine']:.4f}")
                print(f"  验证MMD损失: {val_results['mmd']:.4f}")
            print("-" * 50)
    
    def get_task_performance_summary(self):
        """获取各任务的性能摘要"""
        summary = {}
        for task_type, losses in self.task_losses.items():
            if losses:
                summary[task_type.value] = {
                    'average_loss': np.mean(losses),
                    'final_loss': losses[-1],
                    'improvement': losses[0] - losses[-1] if len(losses) > 1 else 0,
                    'stability': np.std(losses[-5:]) if len(losses) >= 5 else np.std(losses)
                }
        return summary


# 新增任务感知演示函数
def task_aware_demo():
    print("正在进行任务感知多模型协作演示...")
    
    # 加载两个不同的模型
    model1 = AutoModel.from_pretrained("bert-base-uncased")
    model2 = AutoModel.from_pretrained("gpt2")
    
    print("创建任务感知多模型协作系统...")
    collaborator = MultiModelCollaborator([model1, model2])
    
    # 测试不同任务类型的文本
    test_cases = [
        ("What is the capital of France?", TaskType.QUESTION_ANSWERING),
        ("I love this movie, it's amazing!", TaskType.SENTIMENT_ANALYSIS),
        ("Generate a story about a brave knight.", TaskType.TEXT_GENERATION),
        ("Hello, how are you today?", TaskType.CONVERSATION),
        ("The weather is nice today.", TaskType.GENERAL)
    ]
    
    print("\n测试任务感知协作:")
    for text, expected_task in test_cases:
        print(f"\n文本: '{text}'")
        print(f"预期任务类型: {expected_task.value}")
        
        # 自动检测任务类型
        detected_task = collaborator.central_processor.task_detector.detect_task(text)
        print(f"检测到的任务类型: {detected_task.value}")
        
        # 进行任务感知协作
        outputs = collaborator.collaborate(text, 0, 1, task_type=detected_task)
        print(f"适配后的hidden state shape: {outputs['adapted_hidden'].shape}")
    
    # 任务感知训练演示
    print("\n开始任务感知训练...")
    trainer = TaskAwareTrainer(collaborator, learning_rate=1e-4)
    
    # 准备多样化的训练数据集
    train_texts = [
        "What is machine learning?",  # QA
        "I hate rainy days.",         # Sentiment
        "Write a poem about spring.", # Generation
        "Good morning!",              # Conversation
        "The book is on the table.",  # General
        "How does this work?",        # QA
        "This is fantastic!",         # Sentiment
        "Continue this story...",     # Generation
        "Nice to meet you.",          # Conversation
        "Data science is important."  # General
    ]
    
    # 训练3个epoch进行演示
    trainer.train_with_task_awareness(train_texts, epochs=3)
    
    # 显示任务性能摘要
    print("\n任务性能摘要:")
    summary = trainer.get_task_performance_summary()
    for task, metrics in summary.items():
        print(f"{task}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\n任务感知演示完成！")


# 新增训练演示函数
def train_demo():
    print("正在加载模型进行训练演示...")
    # 加载两个不同的模型
    model1 = AutoModel.from_pretrained("bert-base-uncased")
    model2 = AutoModel.from_pretrained("gpt2")
    
    print("创建多模型协作系统...")
    collaborator = MultiModelCollaborator([model1, model2])
    
    print("创建训练器...")
    trainer = AlignmentTrainer(collaborator, learning_rate=1e-4)
    
    # 准备训练数据集
    train_texts = [
        "What is the capital of France?",
        "How are you today?",
        "The weather is nice.",
        "I love machine learning.",
        "Python is a great programming language.",
        "The sun is shining brightly.",
        "Coffee tastes delicious.",
        "Books are full of knowledge."
    ]
    
    # 准备验证数据集
    val_texts = [
        "What is your name?",
        "Good morning everyone.",
        "Technology is advancing rapidly."
    ]
    
    print("训练前的对齐效果:")
    pre_train_results = trainer.evaluate_alignment(val_texts)
    print(f"  余弦相似度: {pre_train_results['cosine']:.4f}")
    print(f"  MMD损失: {pre_train_results['mmd']:.4f}")
    print()
    
    # 训练对齐器
    trainer.train(train_texts, epochs=5, validation_dataset=val_texts)
    
    print("\n训练完成！")
    print("训练后的对齐效果:")
    post_train_results = trainer.evaluate_alignment(val_texts)
    print(f"  余弦相似度: {post_train_results['cosine']:.4f}")
    print(f"  MMD损失: {post_train_results['mmd']:.4f}")
    
    # 测试协作效果
    test_text = "What is the capital of France?"
    outputs = collaborator.collaborate(test_text, 0, 1)
    print(f"\n协作测试结果:")
    print(f"  适配后的hidden state shape: {outputs['adapted_hidden'].shape}")
    print(f"  正常hidden state shape: {outputs['normal_hidden'].shape}")


# 示例使用
def demo():
    print("正在加载模型...")
    # 加载两个不同的模型
    model1 = AutoModel.from_pretrained("bert-base-uncased")
    model2 = AutoModel.from_pretrained("gpt2")
    
    print("创建多模型协作系统...")
    # 创建多模型协作系统
    collaborator = MultiModelCollaborator([model1, model2])
    
    # 示例文本
    text = "What is the capital of France?"
    
    print("模型间协作测试...")
    # 模型间协作：从BERT获取信息，传递给GPT2
    outputs = collaborator.collaborate(text, 0, 1)
    
    print(f"适配后的hidden state shape: {outputs['adapted_hidden'].shape}")
    print(f"正常hidden state shape: {outputs['normal_hidden'].shape}")
    
    # 评估对齐效果（不使用t-SNE可视化）
    print("评估对齐效果...")
    evaluator = AlignmentEvaluator(metrics=['cosine', 'mmd'])
    # 假设有一个数据集
    dataset = [("What is your name?", torch.tensor([0]))] * 3
    alignment_results = evaluator.evaluate(model1, model2, dataset, collaborator.central_processor)
    
    print("Alignment Results:", alignment_results)
    print("演示完成！")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            train_demo()
        elif sys.argv[1] == "task_aware":
            task_aware_demo()
        elif sys.argv[1] == "help":
            print("使用方法:")
            print("  python Multi.py          - 运行基础演示")
            print("  python Multi.py train    - 运行训练演示")
            print("  python Multi.py task_aware - 运行任务感知演示")
            print("  python Multi.py help     - 显示帮助信息")
        else:
            print("未知参数，使用 'python Multi.py help' 查看使用方法")
    else:
        demo()