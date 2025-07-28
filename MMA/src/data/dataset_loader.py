#!/usr/bin/env python3
"""
数据集加载器
提供统一的接口加载和管理训练数据
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
import random
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class DatasetLoader:
    """数据集加载器，提供统一的数据加载接口"""
    
    def __init__(self, data_dir: str, tokenizer_name: str = 'bert-base-uncased'):
        """
        初始化数据集加载器
        
        Args:
            data_dir: 数据目录路径
            tokenizer_name: 分词器名称
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载数据
        self.task_detection_data = self._load_task_detection_data()
        self.adapter_training_data = self._load_adapter_training_data()
        
    def _load_task_detection_data(self) -> List[Dict]:
        """加载任务检测数据"""
        file_path = self.data_dir / 'task_detection_data.json'
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _load_adapter_training_data(self) -> Dict[str, List]:
        """加载适配器训练数据"""
        file_path = self.data_dir / 'adapter_training_data.json'
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'input_texts': [], 'target_texts': [], 'labels': [], 'task_types': [], 'dataset_sources': []}
    
    def get_task_detection_dataset(self, task_types: Optional[List[str]] = None,
                                 max_samples: Optional[int] = None) -> 'TaskDetectionDataset':
        """
        获取任务检测数据集
        
        Args:
            task_types: 指定的任务类型列表，None表示包含所有类型
            max_samples: 最大样本数量
            
        Returns:
            TaskDetectionDataset对象
        """
        filtered_data = []
        
        for item in self.task_detection_data:
            if task_types is None or item['task_type'] in task_types:
                filtered_data.append(item)
        
        if max_samples and len(filtered_data) > max_samples:
            filtered_data = random.sample(filtered_data, max_samples)
        
        return TaskDetectionDataset(filtered_data, self.tokenizer)
    
    def get_adapter_training_dataset(self, task_types: Optional[List[str]] = None,
                                   datasets: Optional[List[str]] = None,
                                   max_samples: Optional[int] = None) -> 'AdapterTrainingDataset':
        """
        获取适配器训练数据集
        
        Args:
            task_types: 指定的任务类型列表
            datasets: 指定的数据集来源列表
            max_samples: 最大样本数量
            
        Returns:
            AdapterTrainingDataset对象
        """
        # 过滤数据
        indices = []
        for i in range(len(self.adapter_training_data['input_texts'])):
            include = True
            
            if task_types and self.adapter_training_data['task_types'][i] not in task_types:
                include = False
            
            if datasets and self.adapter_training_data['dataset_sources'][i] not in datasets:
                include = False
            
            if include:
                indices.append(i)
        
        if max_samples and len(indices) > max_samples:
            indices = random.sample(indices, max_samples)
        
        # 构建过滤后的数据
        filtered_data = {
            'input_texts': [self.adapter_training_data['input_texts'][i] for i in indices],
            'target_texts': [self.adapter_training_data['target_texts'][i] for i in indices],
            'labels': [self.adapter_training_data['labels'][i] for i in indices],
            'task_types': [self.adapter_training_data['task_types'][i] for i in indices],
            'dataset_sources': [self.adapter_training_data['dataset_sources'][i] for i in indices]
        }
        
        return AdapterTrainingDataset(filtered_data, self.tokenizer)
    
    def get_dataloader(self, dataset: Dataset, batch_size: int = 16, 
                      shuffle: bool = True, num_workers: int = 0) -> DataLoader:
        """
        创建DataLoader
        
        Args:
            dataset: 数据集对象
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 工作进程数
            
        Returns:
            DataLoader对象
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
        )
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        stats = {}
        
        # 任务检测数据统计
        if self.task_detection_data:
            task_counts = {}
            for item in self.task_detection_data:
                task_type = item['task_type']
                task_counts[task_type] = task_counts.get(task_type, 0) + 1
            
            stats['task_detection'] = {
                'total_samples': len(self.task_detection_data),
                'task_distribution': task_counts
            }
        
        # 适配器训练数据统计
        if self.adapter_training_data['input_texts']:
            task_counts = {}
            dataset_counts = {}
            
            for task_type in self.adapter_training_data['task_types']:
                task_counts[task_type] = task_counts.get(task_type, 0) + 1
            
            for dataset in self.adapter_training_data['dataset_sources']:
                dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
            
            stats['adapter_training'] = {
                'total_samples': len(self.adapter_training_data['input_texts']),
                'task_distribution': task_counts,
                'dataset_distribution': dataset_counts
            }
        
        return stats


class TaskDetectionDataset(Dataset):
    """任务检测数据集"""
    
    def __init__(self, data: List[Dict], tokenizer: AutoTokenizer, max_length: int = 512):
        """
        初始化任务检测数据集
        
        Args:
            data: 数据列表
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 创建任务类型到ID的映射
        unique_tasks = list(set(item['task_type'] for item in data))
        self.task_to_id = {task: i for i, task in enumerate(unique_tasks)}
        self.id_to_task = {i: task for task, i in self.task_to_id.items()}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 分词
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'task_id': torch.tensor(self.task_to_id[item['task_type']], dtype=torch.long),
            'task_type': item['task_type'],
            'text': item['text']
        }
    
    def collate_fn(self, batch):
        """批处理函数"""
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'task_ids': torch.stack([item['task_id'] for item in batch]),
            'task_types': [item['task_type'] for item in batch],
            'texts': [item['text'] for item in batch]
        }


class AdapterTrainingDataset(Dataset):
    """适配器训练数据集"""
    
    def __init__(self, data: Dict[str, List], tokenizer: AutoTokenizer, max_length: int = 512):
        """
        初始化适配器训练数据集
        
        Args:
            data: 数据字典
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 创建任务类型到ID的映射
        unique_tasks = list(set(data['task_types']))
        self.task_to_id = {task: i for i, task in enumerate(unique_tasks)}
        
    def __len__(self):
        return len(self.data['input_texts'])
    
    def __getitem__(self, idx):
        # 分词输入文本
        input_encoding = self.tokenizer(
            self.data['input_texts'][idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 分词目标文本
        target_encoding = self.tokenizer(
            self.data['target_texts'][idx],
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
            'task_id': torch.tensor(self.task_to_id[self.data['task_types'][idx]], dtype=torch.long),
            'task_type': self.data['task_types'][idx],
            'dataset_source': self.data['dataset_sources'][idx],
            'label': self.data['labels'][idx]
        }
    
    def collate_fn(self, batch):
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


def main():
    """测试数据集加载器"""
    import tempfile
    import json
    
    # 创建测试数据
    test_data_dir = tempfile.mkdtemp()
    
    # 创建测试的任务检测数据
    task_detection_data = [
        {'text': 'What is the capital of France?', 'task_type': 'question_answering', 'dataset_source': 'test'},
        {'text': 'I love this movie!', 'task_type': 'sentiment_analysis', 'dataset_source': 'test'},
        {'text': 'Breaking news about technology', 'task_type': 'text_classification', 'dataset_source': 'test'}
    ]
    
    # 创建测试的适配器训练数据
    adapter_training_data = {
        'input_texts': ['What is AI?', 'This movie is great', 'Technology news'],
        'target_texts': ['Artificial Intelligence', 'positive', 'technology'],
        'labels': [0, 1, 2],
        'task_types': ['question_answering', 'sentiment_analysis', 'text_classification'],
        'dataset_sources': ['test', 'test', 'test']
    }
    
    # 保存测试数据
    with open(f"{test_data_dir}/task_detection_data.json", 'w') as f:
        json.dump(task_detection_data, f)
    
    with open(f"{test_data_dir}/adapter_training_data.json", 'w') as f:
        json.dump(adapter_training_data, f)
    
    # 测试数据集加载器
    loader = DatasetLoader(test_data_dir)
    
    # 获取数据集
    task_dataset = loader.get_task_detection_dataset()
    adapter_dataset = loader.get_adapter_training_dataset()
    
    # 创建DataLoader
    task_dataloader = loader.get_dataloader(task_dataset, batch_size=2)
    adapter_dataloader = loader.get_dataloader(adapter_dataset, batch_size=2)
    
    print("任务检测数据集测试:")
    for batch in task_dataloader:
        print(f"  批次大小: {batch['input_ids'].shape}")
        print(f"  任务类型: {batch['task_types']}")
        break
    
    print("\n适配器训练数据集测试:")
    for batch in adapter_dataloader:
        print(f"  输入批次大小: {batch['input_ids'].shape}")
        print(f"  目标批次大小: {batch['target_ids'].shape}")
        print(f"  任务类型: {batch['task_types']}")
        break
    
    # 打印统计信息
    stats = loader.get_data_statistics()
    print(f"\n数据统计信息: {stats}")
    
    print("数据集加载器测试完成!")


if __name__ == "__main__":
    main()
