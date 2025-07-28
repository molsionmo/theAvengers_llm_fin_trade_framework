#!/usr/bin/env python3
"""
任务感知数据集
专门为任务感知训练设计的数据集类
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any, Union
import random
import numpy as np
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class TaskAwareDataset(Dataset):
    """
    任务感知数据集
    支持多任务训练和任务特定的数据采样
    """
    
    def __init__(self, 
                 data: Dict[str, List],
                 tokenizer: AutoTokenizer,
                 max_length: int = 512,
                 task_sampling_strategy: str = 'balanced',
                 include_task_tokens: bool = True):
        """
        初始化任务感知数据集
        
        Args:
            data: 包含所有训练数据的字典
            tokenizer: 分词器
            max_length: 最大序列长度
            task_sampling_strategy: 任务采样策略 ('balanced', 'proportional', 'random')
            include_task_tokens: 是否在输入中包含任务标识符
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_sampling_strategy = task_sampling_strategy
        self.include_task_tokens = include_task_tokens
        
        # 确保分词器有pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 创建任务映射
        self.unique_tasks = list(set(data['task_types']))
        self.task_to_id = {task: i for i, task in enumerate(self.unique_tasks)}
        self.id_to_task = {i: task for task, i in self.task_to_id.items()}
        
        # 按任务组织数据索引
        self.task_indices = {task: [] for task in self.unique_tasks}
        for i, task_type in enumerate(data['task_types']):
            self.task_indices[task_type].append(i)
        
        # 创建采样权重
        self.sampling_weights = self._calculate_sampling_weights()
        
        # 添加特殊任务token
        if self.include_task_tokens:
            self._add_task_tokens_to_tokenizer()
    
    def _calculate_sampling_weights(self) -> Dict[str, float]:
        """计算任务采样权重"""
        weights = {}
        total_samples = len(self.data['input_texts'])
        
        if self.task_sampling_strategy == 'balanced':
            # 平衡采样：每个任务权重相等
            weight_per_task = 1.0 / len(self.unique_tasks)
            for task in self.unique_tasks:
                weights[task] = weight_per_task
                
        elif self.task_sampling_strategy == 'proportional':
            # 比例采样：按原始数据分布采样
            for task in self.unique_tasks:
                task_count = len(self.task_indices[task])
                weights[task] = task_count / total_samples
                
        elif self.task_sampling_strategy == 'random':
            # 随机采样：完全随机
            for task in self.unique_tasks:
                weights[task] = random.random()
            # 归一化
            total_weight = sum(weights.values())
            weights = {task: w / total_weight for task, w in weights.items()}
        
        return weights
    
    def _add_task_tokens_to_tokenizer(self):
        """向分词器添加任务特定的特殊token"""
        task_tokens = [f"<{task}>" for task in self.unique_tasks]
        
        # 检查token是否已存在
        existing_tokens = set(self.tokenizer.get_vocab().keys())
        new_tokens = [token for token in task_tokens if token not in existing_tokens]
        
        if new_tokens:
            self.tokenizer.add_tokens(new_tokens)
            logger.info(f"添加了 {len(new_tokens)} 个任务标识符token")
    
    def __len__(self):
        return len(self.data['input_texts'])
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        # 获取基本数据
        input_text = self.data['input_texts'][idx]
        target_text = self.data['target_texts'][idx]
        task_type = self.data['task_types'][idx]
        dataset_source = self.data['dataset_sources'][idx]
        label = self.data['labels'][idx]
        
        # 添加任务标识符（如果启用）
        if self.include_task_tokens:
            task_token = f"<{task_type}>"
            input_text = f"{task_token} {input_text}"
        
        # 分词输入文本
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 分词目标文本
        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'input_attention_mask': input_encoding['attention_mask'].squeeze(),
            'target_ids': target_encoding['input_ids'].squeeze(),
            'target_attention_mask': target_encoding['attention_mask'].squeeze(),
            'task_id': torch.tensor(self.task_to_id[task_type], dtype=torch.long),
            'task_type': task_type,
            'dataset_source': dataset_source,
            'label': label
        }
    
    def get_task_batch(self, task_type: str, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """获取特定任务的批次数据"""
        if task_type not in self.task_indices:
            raise ValueError(f"未知任务类型: {task_type}")
        
        # 随机采样该任务的样本
        task_sample_indices = random.sample(
            self.task_indices[task_type],
            min(batch_size, len(self.task_indices[task_type]))
        )
        
        return [self[idx] for idx in task_sample_indices]
    
    def get_mixed_task_batch(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """获取多任务混合批次"""
        batch = []
        
        for _ in range(batch_size):
            # 根据采样策略选择任务
            task_type = self._sample_task()
            
            # 从该任务中随机选择一个样本
            task_idx = random.choice(self.task_indices[task_type])
            batch.append(self[task_idx])
        
        return batch
    
    def _sample_task(self) -> str:
        """根据采样策略选择任务"""
        if self.task_sampling_strategy == 'random':
            return random.choice(self.unique_tasks)
        else:
            # 使用权重采样
            tasks = list(self.sampling_weights.keys())
            weights = list(self.sampling_weights.values())
            return np.random.choice(tasks, p=weights)
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """获取任务统计信息"""
        stats = {
            'total_samples': len(self.data['input_texts']),
            'num_tasks': len(self.unique_tasks),
            'task_distribution': {},
            'sampling_weights': self.sampling_weights,
            'avg_input_length': 0,
            'avg_target_length': 0
        }
        
        # 计算每个任务的样本数
        for task in self.unique_tasks:
            task_count = len(self.task_indices[task])
            stats['task_distribution'][task] = {
                'count': task_count,
                'percentage': (task_count / stats['total_samples']) * 100
            }
        
        # 计算平均长度
        input_lengths = [len(text.split()) for text in self.data['input_texts']]
        target_lengths = [len(text.split()) for text in self.data['target_texts']]
        
        stats['avg_input_length'] = np.mean(input_lengths)
        stats['avg_target_length'] = np.mean(target_lengths)
        
        return stats
    
    def filter_by_tasks(self, task_types: List[str]) -> 'TaskAwareDataset':
        """根据任务类型过滤数据集"""
        filtered_indices = []
        for task_type in task_types:
            if task_type in self.task_indices:
                filtered_indices.extend(self.task_indices[task_type])
        
        # 创建过滤后的数据
        filtered_data = {
            'input_texts': [self.data['input_texts'][i] for i in filtered_indices],
            'target_texts': [self.data['target_texts'][i] for i in filtered_indices],
            'task_types': [self.data['task_types'][i] for i in filtered_indices],
            'dataset_sources': [self.data['dataset_sources'][i] for i in filtered_indices],
            'labels': [self.data['labels'][i] for i in filtered_indices]
        }
        
        return TaskAwareDataset(
            filtered_data,
            self.tokenizer,
            self.max_length,
            self.task_sampling_strategy,
            self.include_task_tokens
        )
    
    def split_by_ratio(self, train_ratio: float = 0.8) -> Tuple['TaskAwareDataset', 'TaskAwareDataset']:
        """按比例分割数据集"""
        total_samples = len(self.data['input_texts'])
        train_size = int(total_samples * train_ratio)
        
        # 随机打乱索引
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # 创建训练集
        train_data = {
            'input_texts': [self.data['input_texts'][i] for i in train_indices],
            'target_texts': [self.data['target_texts'][i] for i in train_indices],
            'task_types': [self.data['task_types'][i] for i in train_indices],
            'dataset_sources': [self.data['dataset_sources'][i] for i in train_indices],
            'labels': [self.data['labels'][i] for i in train_indices]
        }
        
        # 创建验证集
        val_data = {
            'input_texts': [self.data['input_texts'][i] for i in val_indices],
            'target_texts': [self.data['target_texts'][i] for i in val_indices],
            'task_types': [self.data['task_types'][i] for i in val_indices],
            'dataset_sources': [self.data['dataset_sources'][i] for i in val_indices],
            'labels': [self.data['labels'][i] for i in val_indices]
        }
        
        train_dataset = TaskAwareDataset(
            train_data, self.tokenizer, self.max_length,
            self.task_sampling_strategy, self.include_task_tokens
        )
        
        val_dataset = TaskAwareDataset(
            val_data, self.tokenizer, self.max_length,
            self.task_sampling_strategy, self.include_task_tokens
        )
        
        return train_dataset, val_dataset
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """批处理函数"""
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'input_attention_mask': torch.stack([item['input_attention_mask'] for item in batch]),
            'target_ids': torch.stack([item['target_ids'] for item in batch]),
            'target_attention_mask': torch.stack([item['target_attention_mask'] for item in batch]),
            'task_ids': torch.stack([item['task_id'] for item in batch]),
            'task_types': [item['task_type'] for item in batch],
            'dataset_sources': [item['dataset_source'] for item in batch],
            'labels': [item['label'] for item in batch]
        }


class TaskBalancedSampler:
    """任务平衡采样器"""
    
    def __init__(self, dataset: TaskAwareDataset, samples_per_task: int = 100):
        """
        初始化任务平衡采样器
        
        Args:
            dataset: 任务感知数据集
            samples_per_task: 每个任务的样本数
        """
        self.dataset = dataset
        self.samples_per_task = samples_per_task
        self.task_indices = dataset.task_indices
        
    def __iter__(self):
        """迭代器"""
        all_indices = []
        
        for task_type in self.dataset.unique_tasks:
            task_indices = self.task_indices[task_type]
            
            # 如果该任务样本数少于需要的数量，进行重复采样
            if len(task_indices) < self.samples_per_task:
                sampled_indices = np.random.choice(
                    task_indices, 
                    size=self.samples_per_task, 
                    replace=True
                ).tolist()
            else:
                sampled_indices = random.sample(task_indices, self.samples_per_task)
            
            all_indices.extend(sampled_indices)
        
        # 打乱所有索引
        random.shuffle(all_indices)
        return iter(all_indices)
    
    def __len__(self):
        return len(self.dataset.unique_tasks) * self.samples_per_task


def create_curriculum_dataset(base_dataset: TaskAwareDataset, 
                            difficulty_levels: int = 3) -> List[TaskAwareDataset]:
    """
    创建课程学习数据集
    根据文本长度和复杂度分层
    
    Args:
        base_dataset: 基础数据集
        difficulty_levels: 难度等级数量
        
    Returns:
        按难度分层的数据集列表
    """
    # 计算每个样本的难度分数（基于文本长度）
    difficulty_scores = []
    for i in range(len(base_dataset.data['input_texts'])):
        input_len = len(base_dataset.data['input_texts'][i].split())
        target_len = len(base_dataset.data['target_texts'][i].split())
        difficulty = (input_len + target_len) / 2  # 简单的难度度量
        difficulty_scores.append((i, difficulty))
    
    # 按难度排序
    difficulty_scores.sort(key=lambda x: x[1])
    
    # 分层
    samples_per_level = len(difficulty_scores) // difficulty_levels
    curriculum_datasets = []
    
    for level in range(difficulty_levels):
        start_idx = level * samples_per_level
        if level == difficulty_levels - 1:
            # 最后一层包含所有剩余样本
            end_idx = len(difficulty_scores)
        else:
            end_idx = (level + 1) * samples_per_level
        
        level_indices = [difficulty_scores[i][0] for i in range(start_idx, end_idx)]
        
        # 创建该难度级别的数据
        level_data = {
            'input_texts': [base_dataset.data['input_texts'][i] for i in level_indices],
            'target_texts': [base_dataset.data['target_texts'][i] for i in level_indices],
            'task_types': [base_dataset.data['task_types'][i] for i in level_indices],
            'dataset_sources': [base_dataset.data['dataset_sources'][i] for i in level_indices],
            'labels': [base_dataset.data['labels'][i] for i in level_indices]
        }
        
        level_dataset = TaskAwareDataset(
            level_data,
            base_dataset.tokenizer,
            base_dataset.max_length,
            base_dataset.task_sampling_strategy,
            base_dataset.include_task_tokens
        )
        
        curriculum_datasets.append(level_dataset)
    
    return curriculum_datasets


def main():
    """测试任务感知数据集"""
    from transformers import AutoTokenizer
    
    # 创建测试数据
    test_data = {
        'input_texts': [
            'What is the capital of France?',
            'I love this movie!',
            'Breaking news about technology',
            'How does machine learning work?',
            'This restaurant is terrible.',
            'Sports news update'
        ],
        'target_texts': [
            'Paris',
            'positive',
            'technology',
            'ML uses algorithms',
            'negative',
            'sports'
        ],
        'task_types': [
            'question_answering',
            'sentiment_analysis', 
            'text_classification',
            'question_answering',
            'sentiment_analysis',
            'text_classification'
        ],
        'dataset_sources': ['test'] * 6,
        'labels': [0, 1, 2, 0, 1, 2]
    }
    
    # 初始化分词器
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # 创建任务感知数据集
    dataset = TaskAwareDataset(
        test_data,
        tokenizer,
        max_length=128,
        task_sampling_strategy='balanced',
        include_task_tokens=True
    )
    
    print("任务感知数据集测试:")
    print(f"数据集大小: {len(dataset)}")
    print(f"任务类型: {dataset.unique_tasks}")
    
    # 测试单个样本
    sample = dataset[0]
    print(f"\n样本测试:")
    print(f"输入形状: {sample['input_ids'].shape}")
    print(f"任务ID: {sample['task_id']}")
    print(f"任务类型: {sample['task_type']}")
    
    # 测试任务特定批次
    qa_batch = dataset.get_task_batch('question_answering', 2)
    print(f"\n问答任务批次大小: {len(qa_batch)}")
    
    # 测试混合任务批次
    mixed_batch = dataset.get_mixed_task_batch(3)
    print(f"混合任务批次大小: {len(mixed_batch)}")
    
    # 测试统计信息
    stats = dataset.get_task_statistics()
    print(f"\n数据集统计: {stats}")
    
    # 测试数据集分割
    train_dataset, val_dataset = dataset.split_by_ratio(0.8)
    print(f"\n分割后 - 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    # 测试课程学习
    curriculum_datasets = create_curriculum_dataset(dataset, difficulty_levels=2)
    print(f"\n课程学习数据集: {[len(d) for d in curriculum_datasets]}")
    
    print("任务感知数据集测试完成!")


if __name__ == "__main__":
    main()
