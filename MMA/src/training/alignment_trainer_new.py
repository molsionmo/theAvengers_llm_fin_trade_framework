"""
对齐训练器模块

用于训练中心处理层的对齐，支持数据集选择参数
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
from torch.utils.data import DataLoader
import logging

from ..utils.evaluator import AlignmentEvaluator
from ..data.dataset_loader import DatasetLoader
from ..data.task_dataset import TaskAwareDataset

logger = logging.getLogger(__name__)


class AlignmentTrainer:
    """对齐训练器，用于训练中心处理层"""
    
    def __init__(self, collaborator, learning_rate=1e-4, 
                 dataset_config: Optional[Dict[str, Any]] = None):
        """
        初始化对齐训练器
        
        Args:
            collaborator: 多模型协作器
            learning_rate: 学习率
            dataset_config: 数据集配置，包含数据集选择参数
        """
        self.collaborator = collaborator
        self.optimizer = torch.optim.Adam(collaborator.central_processor.parameters(), lr=learning_rate)
        self.dataset_config = dataset_config or {}
        
        # 数据集加载器
        self.dataset_loader = None
        if 'data_dir' in self.dataset_config:
            self.dataset_loader = DatasetLoader(
                self.dataset_config['data_dir'],
                tokenizer_name=self.dataset_config.get('tokenizer_name', 'bert-base-uncased')
            )
        
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
    
    def alignment_loss(self, proj1, proj2, alpha=0.5):
        """对齐损失，结合L2距离和余弦相似度"""
        # 确保两个投影有相同的序列长度
        min_len = min(proj1.size(1), proj2.size(1))
        proj1 = proj1[:, :min_len, :]
        proj2 = proj2[:, :min_len, :]
        
        # L2距离损失
        l2_loss = F.mse_loss(proj1, proj2)
        
        # 余弦相似度损失
        cosine_sim = F.cosine_similarity(proj1, proj2, dim=-1)
        cosine_loss = 1 - cosine_sim.mean()
        
        # 组合损失
        total_loss = alpha * l2_loss + (1 - alpha) * cosine_loss
        
        return total_loss
    
    def train_with_dataset_selection(self, 
                                    dataset_names: List[str],
                                    epochs: int = 10,
                                    batch_size: int = 16,
                                    max_samples_per_dataset: int = 1000,
                                    task_sampling_strategy: str = 'balanced',
                                    validation_split: float = 0.2) -> Dict[str, Any]:
        """
        使用数据集选择参数进行训练
        
        Args:
            dataset_names: 选择的数据集名称列表
            epochs: 训练轮数
            batch_size: 批次大小
            max_samples_per_dataset: 每个数据集的最大样本数
            task_sampling_strategy: 任务采样策略
            validation_split: 验证集比例
            
        Returns:
            训练结果字典
        """
        if self.dataset_loader is None:
            raise ValueError("请在初始化时提供dataset_config参数以启用数据集选择功能")
        
        print(f"开始使用数据集选择训练: {dataset_names}")
        
        # 获取适配器训练数据集
        adapter_dataset = self.dataset_loader.get_adapter_training_dataset(
            datasets=dataset_names,
            max_samples=max_samples_per_dataset * len(dataset_names)
        )
        
        if len(adapter_dataset) == 0:
            raise ValueError("选择的数据集中没有找到有效数据")
        
        # 创建任务感知数据集
        task_aware_dataset = TaskAwareDataset(
            adapter_dataset.data,
            adapter_dataset.tokenizer,
            max_length=512,
            task_sampling_strategy=task_sampling_strategy,
            include_task_tokens=True
        )
        
        # 分割训练集和验证集
        train_dataset, val_dataset = task_aware_dataset.split_by_ratio(1 - validation_split)
        
        print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
        print(f"数据集统计: {train_dataset.get_task_statistics()}")
        
        # 创建数据加载器
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=train_dataset.collate_fn
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=val_dataset.collate_fn
        )
        
        # 训练过程
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # 训练阶段
            train_loss = self._train_epoch_with_dataloader(train_dataloader)
            train_losses.append(train_loss)
            
            # 验证阶段
            val_loss = self._validate_epoch_with_dataloader(val_dataloader)
            val_losses.append(val_loss)
            
            print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"发现更好的模型 (验证损失: {val_loss:.4f})")
        
        # 返回训练结果
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'final_train_loss': train_losses[-1],
            'dataset_info': {
                'selected_datasets': dataset_names,
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'task_distribution': train_dataset.get_task_statistics()['task_distribution']
            }
        }
    
    def _train_epoch_with_dataloader(self, dataloader: DataLoader) -> float:
        """使用DataLoader训练一个epoch"""
        self.collaborator.central_processor.train()
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            try:
                batch_loss = 0
                batch_size = len(batch['input_ids'])
                
                # 处理批次中的每个样本
                for i in range(batch_size):
                    # 获取单个样本的文本
                    input_ids = batch['input_ids'][i:i+1]  # 保持批次维度
                    
                    # 解码文本（用于获取hidden states）
                    if self.dataset_loader:
                        tokenizer = self.dataset_loader.tokenizer
                    else:
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                    
                    text = tokenizer.decode(
                        input_ids[0], 
                        skip_special_tokens=True
                    )
                    
                    # 获取两个模型的Hidden State
                    hidden1 = self.collaborator.get_hidden_states(text, 0)
                    hidden2 = self.collaborator.get_hidden_states(text, 1)
                    
                    # 投影到共享空间
                    projected_states = self.collaborator.central_processor.process([hidden1, hidden2])
                    proj1, proj2 = projected_states
                    
                    # 计算对齐损失
                    loss = self.alignment_loss(proj1, proj2)
                    batch_loss += loss
                
                # 平均批次损失
                batch_loss = batch_loss / batch_size
                
                # 反向传播
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                
                total_loss += batch_loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"处理批次时出错: {e}")
                continue
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def _validate_epoch_with_dataloader(self, dataloader: DataLoader) -> float:
        """使用DataLoader验证一个epoch"""
        self.collaborator.central_processor.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                try:
                    batch_loss = 0
                    batch_size = len(batch['input_ids'])
                    
                    # 处理批次中的每个样本
                    for i in range(batch_size):
                        # 获取单个样本的文本
                        input_ids = batch['input_ids'][i:i+1]
                        
                        # 解码文本
                        if self.dataset_loader:
                            tokenizer = self.dataset_loader.tokenizer
                        else:
                            from transformers import AutoTokenizer
                            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                            
                        text = tokenizer.decode(
                            input_ids[0], 
                            skip_special_tokens=True
                        )
                        
                        # 获取hidden states并计算损失
                        hidden1 = self.collaborator.get_hidden_states(text, 0)
                        hidden2 = self.collaborator.get_hidden_states(text, 1)
                        
                        projected_states = self.collaborator.central_processor.process([hidden1, hidden2])
                        proj1, proj2 = projected_states
                        
                        loss = self.alignment_loss(proj1, proj2)
                        batch_loss += loss
                    
                    # 平均批次损失
                    batch_loss = batch_loss / batch_size
                    total_loss += batch_loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"验证批次时出错: {e}")
                    continue
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def train_epoch(self, train_dataset):
        """训练一个epoch（兼容原有接口）"""
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
        
        train_losses = []
        val_results = []
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_dataset)
            train_losses.append(train_loss)
            
            # 验证
            epoch_val_results = None
            if validation_dataset is not None:
                epoch_val_results = self.evaluate_alignment(validation_dataset)
                val_results.append(epoch_val_results)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  训练损失: {train_loss:.4f}")
            if epoch_val_results:
                print(f"  验证余弦相似度: {epoch_val_results['cosine']:.4f}")
                print(f"  验证MMD损失: {epoch_val_results['mmd']:.4f}")
            print("-" * 50)
        
        return {
            'train_losses': train_losses,
            'val_results': val_results
        }
    
    def evaluate_alignment(self, dataset):
        """评估对齐效果"""
        self.collaborator.central_processor.eval()
        evaluator = AlignmentEvaluator(metrics=['cosine', 'mmd'])
        
        # 准备评估数据
        if isinstance(dataset, list):
            # 文本列表格式
            eval_texts = dataset[:min(50, len(dataset))]  # 限制评估数据量
        else:
            # 其他格式，尝试转换
            eval_texts = []
        
        all_proj1 = []
        all_proj2 = []
        
        with torch.no_grad():
            for text in eval_texts:
                try:
                    hidden1 = self.collaborator.get_hidden_states(text, 0)
                    hidden2 = self.collaborator.get_hidden_states(text, 1)
                    
                    projected_states = self.collaborator.central_processor.process([hidden1, hidden2])
                    proj1, proj2 = projected_states
                    
                    all_proj1.append(proj1)
                    all_proj2.append(proj2)
                    
                except Exception as e:
                    print(f"评估时处理文本出错: {e}")
                    continue
        
        if not all_proj1:
            return {'cosine': 0.0, 'mmd': float('inf')}
        
        # 拼接所有投影
        all_proj1 = torch.cat(all_proj1, dim=0)
        all_proj2 = torch.cat(all_proj2, dim=0)
        
        # 计算评估指标
        cosine_sim = evaluator.cosine_similarity(all_proj1, all_proj2)
        mmd_loss = evaluator.mmd_loss(
            all_proj1.view(-1, all_proj1.size(-1)),
            all_proj2.view(-1, all_proj2.size(-1))
        ).item()
        
        return {
            'cosine': cosine_sim,
            'mmd': mmd_loss
        }
