#!/usr/bin/env python3
"""
数据预处理器
支持从多种互联网数据集处理为适合任务感知和适配器训练的格式
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import urllib.request
import zipfile
import tarfile
import gzip
from pathlib import Path
import logging
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import torch
from sklearn.model_selection import train_test_split

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """数据预处理器，支持多种数据集格式和任务类型"""
    
    def __init__(self, cache_dir: str = "./data_cache", max_length: int = 512):
        """
        初始化数据预处理器
        
        Args:
            cache_dir: 数据缓存目录
            max_length: 文本最大长度
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_length = max_length
        
        # 支持的数据集配置
        self.dataset_configs = {
            # 问答任务
            'squad': {
                'task_type': 'question_answering',
                'huggingface_name': 'squad',
                'text_fields': ['question', 'context'],
                'label_field': 'answers'
            },
            'squad_v2': {
                'task_type': 'question_answering', 
                'huggingface_name': 'squad_v2',
                'text_fields': ['question', 'context'],
                'label_field': 'answers'
            },
            
            # 情感分析
            'imdb': {
                'task_type': 'sentiment_analysis',
                'huggingface_name': 'imdb',
                'text_fields': ['text'],
                'label_field': 'label'
            },
            'sst2': {
                'task_type': 'sentiment_analysis',
                'huggingface_name': 'glue',
                'subset': 'sst2',
                'text_fields': ['sentence'],
                'label_field': 'label'
            },
            
            # 文本分类
            'ag_news': {
                'task_type': 'text_classification',
                'huggingface_name': 'ag_news',
                'text_fields': ['text'],
                'label_field': 'label'
            },
            '20newsgroups': {
                'task_type': 'text_classification',
                'huggingface_name': '20_newsgroups',
                'text_fields': ['text'],
                'label_field': 'label'
            },
            
            # 文本相似度
            'sts_benchmark': {
                'task_type': 'text_similarity',
                'huggingface_name': 'stsb_multi_mt',
                'subset': 'en',
                'text_fields': ['sentence1', 'sentence2'],
                'label_field': 'similarity_score'
            },
            
            # 自然语言推理
            'snli': {
                'task_type': 'natural_language_inference',
                'huggingface_name': 'snli',
                'text_fields': ['premise', 'hypothesis'],
                'label_field': 'label'
            },
            'mnli': {
                'task_type': 'natural_language_inference',
                'huggingface_name': 'glue',
                'subset': 'mnli',
                'text_fields': ['premise', 'hypothesis'],
                'label_field': 'label'
            },
            
            # 文本生成/补全
            'wikitext': {
                'task_type': 'text_generation',
                'huggingface_name': 'wikitext',
                'subset': 'wikitext-2-raw-v1',
                'text_fields': ['text'],
                'label_field': None
            },
            'openwebtext': {
                'task_type': 'text_generation',
                'huggingface_name': 'openwebtext',
                'text_fields': ['text'],
                'label_field': None
            },
            
            # 对话任务
            'persona_chat': {
                'task_type': 'conversation',
                'huggingface_name': 'blended_skill_talk',
                'text_fields': ['personas', 'context'],
                'label_field': 'response'
            },
            
            # 总结任务
            'cnn_dailymail': {
                'task_type': 'summarization',
                'huggingface_name': 'cnn_dailymail',
                'subset': '3.0.0',
                'text_fields': ['article'],
                'label_field': 'highlights'
            },
            'xsum': {
                'task_type': 'summarization',
                'huggingface_name': 'xsum',
                'text_fields': ['document'],
                'label_field': 'summary'
            }
        }
    
    def list_available_datasets(self) -> Dict[str, Dict]:
        """列出所有可用的数据集"""
        return self.dataset_configs
    
    def download_dataset(self, dataset_name: str, split: str = 'train', 
                        max_samples: Optional[int] = None) -> Dataset:
        """
        下载并加载数据集
        
        Args:
            dataset_name: 数据集名称
            split: 数据集分割（train/validation/test）
            max_samples: 最大样本数量
            
        Returns:
            处理后的Dataset对象
        """
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        
        config = self.dataset_configs[dataset_name]
        logger.info(f"正在下载数据集: {dataset_name} ({split})")
        
        try:
            # 加载HuggingFace数据集
            if 'subset' in config:
                dataset = load_dataset(
                    config['huggingface_name'], 
                    config['subset'],
                    split=split,
                    cache_dir=str(self.cache_dir)
                )
            else:
                dataset = load_dataset(
                    config['huggingface_name'],
                    split=split,
                    cache_dir=str(self.cache_dir)
                )
            
            # 限制样本数量
            if max_samples and len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
            
            logger.info(f"成功加载数据集: {dataset_name}, 样本数: {len(dataset)}")
            return dataset
            
        except Exception as e:
            logger.error(f"加载数据集失败: {dataset_name}, 错误: {str(e)}")
            raise
    
    def preprocess_for_task_detection(self, dataset_name: str, 
                                    split: str = 'train',
                                    max_samples: Optional[int] = None) -> List[Dict]:
        """
        预处理数据用于任务检测训练
        
        Args:
            dataset_name: 数据集名称
            split: 数据集分割
            max_samples: 最大样本数量
            
        Returns:
            格式化的训练数据列表
        """
        dataset = self.download_dataset(dataset_name, split, max_samples)
        config = self.dataset_configs[dataset_name]
        
        processed_data = []
        
        for example in dataset:
            # 提取文本内容
            texts = []
            for field in config['text_fields']:
                if field in example and example[field]:
                    if isinstance(example[field], list):
                        texts.extend([str(item) for item in example[field]])
                    else:
                        texts.append(str(example[field]))
            
            # 合并文本
            combined_text = ' '.join(texts)
            
            # 截断过长文本
            if len(combined_text) > self.max_length * 4:  # 粗略估计token数
                combined_text = combined_text[:self.max_length * 4]
            
            processed_data.append({
                'text': combined_text,
                'task_type': config['task_type'],
                'dataset_source': dataset_name
            })
        
        logger.info(f"任务检测数据预处理完成: {len(processed_data)} 条记录")
        return processed_data
    
    def preprocess_for_adapter_training(self, dataset_name: str,
                                      split: str = 'train',
                                      max_samples: Optional[int] = None,
                                      tokenizer_name: str = 'bert-base-uncased') -> Dict[str, Any]:
        """
        预处理数据用于适配器训练
        
        Args:
            dataset_name: 数据集名称
            split: 数据集分割
            max_samples: 最大样本数量
            tokenizer_name: 分词器名称
            
        Returns:
            包含输入输出对的字典
        """
        dataset = self.download_dataset(dataset_name, split, max_samples)
        config = self.dataset_configs[dataset_name]
        
        # 初始化分词器
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        processed_data = {
            'input_texts': [],
            'target_texts': [],
            'labels': [],
            'task_type': config['task_type'],
            'dataset_source': dataset_name
        }
        
        for example in dataset:
            try:
                # 处理不同任务类型
                if config['task_type'] == 'question_answering':
                    input_text = f"问题: {example['question']} 上下文: {example['context']}"
                    if example['answers']['text']:
                        target_text = example['answers']['text'][0]
                    else:
                        continue  # 跳过没有答案的样本
                    
                elif config['task_type'] in ['sentiment_analysis', 'text_classification']:
                    input_text = example[config['text_fields'][0]]
                    target_text = str(example[config['label_field']])
                    
                elif config['task_type'] == 'text_similarity':
                    input_text = f"句子1: {example['sentence1']} 句子2: {example['sentence2']}"
                    target_text = str(example[config['label_field']])
                    
                elif config['task_type'] == 'natural_language_inference':
                    input_text = f"前提: {example['premise']} 假设: {example['hypothesis']}"
                    target_text = str(example[config['label_field']])
                    
                elif config['task_type'] == 'text_generation':
                    input_text = example['text'][:len(example['text'])//2]  # 前半部分作为输入
                    target_text = example['text'][len(example['text'])//2:]  # 后半部分作为目标
                    
                elif config['task_type'] == 'summarization':
                    input_text = example[config['text_fields'][0]]
                    target_text = example[config['label_field']]
                    
                elif config['task_type'] == 'conversation':
                    input_text = ' '.join(example['context']) if isinstance(example['context'], list) else example['context']
                    target_text = example[config['label_field']]
                    
                else:
                    # 默认处理方式
                    input_text = ' '.join([str(example[field]) for field in config['text_fields'] if field in example])
                    target_text = str(example[config['label_field']]) if config['label_field'] else input_text
                
                # 验证和清理文本
                if input_text and target_text and len(input_text.strip()) > 0 and len(target_text.strip()) > 0:
                    processed_data['input_texts'].append(input_text.strip())
                    processed_data['target_texts'].append(target_text.strip())
                    processed_data['labels'].append(example.get(config['label_field'], 0))
                    
            except KeyError as e:
                logger.warning(f"跳过缺少字段的样本: {e}")
                continue
            except Exception as e:
                logger.warning(f"处理样本时出错: {e}")
                continue
        
        logger.info(f"适配器训练数据预处理完成: {len(processed_data['input_texts'])} 条记录")
        return processed_data
    
    def create_mixed_dataset(self, dataset_names: List[str],
                           max_samples_per_dataset: int = 1000,
                           split: str = 'train') -> Dict[str, Any]:
        """
        创建混合数据集，包含多种任务类型
        
        Args:
            dataset_names: 数据集名称列表
            max_samples_per_dataset: 每个数据集的最大样本数
            split: 数据集分割
            
        Returns:
            混合的训练数据
        """
        all_task_data = []
        all_adapter_data = {
            'input_texts': [],
            'target_texts': [],
            'labels': [],
            'task_types': [],
            'dataset_sources': []
        }
        
        for dataset_name in dataset_names:
            try:
                # 任务检测数据
                task_data = self.preprocess_for_task_detection(
                    dataset_name, split, max_samples_per_dataset
                )
                all_task_data.extend(task_data)
                
                # 适配器训练数据
                adapter_data = self.preprocess_for_adapter_training(
                    dataset_name, split, max_samples_per_dataset
                )
                
                all_adapter_data['input_texts'].extend(adapter_data['input_texts'])
                all_adapter_data['target_texts'].extend(adapter_data['target_texts'])
                all_adapter_data['labels'].extend(adapter_data['labels'])
                all_adapter_data['task_types'].extend([adapter_data['task_type']] * len(adapter_data['input_texts']))
                all_adapter_data['dataset_sources'].extend([adapter_data['dataset_source']] * len(adapter_data['input_texts']))
                
                logger.info(f"已处理数据集: {dataset_name}")
                
            except Exception as e:
                logger.error(f"处理数据集 {dataset_name} 时出错: {e}")
                continue
        
        return {
            'task_detection_data': all_task_data,
            'adapter_training_data': all_adapter_data
        }
    
    def save_processed_data(self, data: Dict[str, Any], output_dir: str):
        """
        保存处理后的数据
        
        Args:
            data: 处理后的数据
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 保存任务检测数据
        if 'task_detection_data' in data:
            task_detection_file = output_path / 'task_detection_data.json'
            with open(task_detection_file, 'w', encoding='utf-8') as f:
                json.dump(data['task_detection_data'], f, ensure_ascii=False, indent=2)
            logger.info(f"任务检测数据已保存到: {task_detection_file}")
        
        # 保存适配器训练数据
        if 'adapter_training_data' in data:
            adapter_training_file = output_path / 'adapter_training_data.json'
            with open(adapter_training_file, 'w', encoding='utf-8') as f:
                json.dump(data['adapter_training_data'], f, ensure_ascii=False, indent=2)
            logger.info(f"适配器训练数据已保存到: {adapter_training_file}")
        
        # 保存数据统计信息
        stats = self.generate_data_statistics(data)
        stats_file = output_path / 'data_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"数据统计信息已保存到: {stats_file}")
    
    def generate_data_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成数据统计信息
        
        Args:
            data: 处理后的数据
            
        Returns:
            统计信息字典
        """
        stats = {}
        
        if 'task_detection_data' in data:
            task_data = data['task_detection_data']
            task_counts = {}
            dataset_counts = {}
            
            for item in task_data:
                task_type = item['task_type']
                dataset_source = item['dataset_source']
                
                task_counts[task_type] = task_counts.get(task_type, 0) + 1
                dataset_counts[dataset_source] = dataset_counts.get(dataset_source, 0) + 1
            
            stats['task_detection'] = {
                'total_samples': len(task_data),
                'task_type_distribution': task_counts,
                'dataset_distribution': dataset_counts
            }
        
        if 'adapter_training_data' in data:
            adapter_data = data['adapter_training_data']
            
            stats['adapter_training'] = {
                'total_samples': len(adapter_data['input_texts']),
                'avg_input_length': np.mean([len(text.split()) for text in adapter_data['input_texts']]),
                'avg_target_length': np.mean([len(text.split()) for text in adapter_data['target_texts']]),
                'unique_task_types': len(set(adapter_data['task_types'])),
                'unique_datasets': len(set(adapter_data['dataset_sources']))
            }
        
        return stats
    
    def load_processed_data(self, data_dir: str) -> Dict[str, Any]:
        """
        加载已处理的数据
        
        Args:
            data_dir: 数据目录
            
        Returns:
            加载的数据字典
        """
        data_path = Path(data_dir)
        loaded_data = {}
        
        # 加载任务检测数据
        task_detection_file = data_path / 'task_detection_data.json'
        if task_detection_file.exists():
            with open(task_detection_file, 'r', encoding='utf-8') as f:
                loaded_data['task_detection_data'] = json.load(f)
        
        # 加载适配器训练数据
        adapter_training_file = data_path / 'adapter_training_data.json'
        if adapter_training_file.exists():
            with open(adapter_training_file, 'r', encoding='utf-8') as f:
                loaded_data['adapter_training_data'] = json.load(f)
        
        return loaded_data


def main():
    """测试数据预处理器功能"""
    # 创建预处理器
    preprocessor = DataPreprocessor()
    
    # 列出可用数据集
    print("可用数据集:")
    for name, config in preprocessor.list_available_datasets().items():
        print(f"  {name}: {config['task_type']}")
    
    # 创建小规模混合数据集进行测试
    test_datasets = ['imdb', 'ag_news', 'squad']
    print(f"\n正在处理测试数据集: {test_datasets}")
    
    mixed_data = preprocessor.create_mixed_dataset(
        dataset_names=test_datasets,
        max_samples_per_dataset=100,
        split='train'
    )
    
    # 保存数据
    output_dir = "./processed_data_test"
    preprocessor.save_processed_data(mixed_data, output_dir)
    
    print(f"\n数据处理完成，已保存到: {output_dir}")


if __name__ == "__main__":
    main()
