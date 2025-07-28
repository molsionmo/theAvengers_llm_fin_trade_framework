#!/usr/bin/env python3
"""
数据预处理CLI脚本
用于下载和预处理各种数据集
"""

import argparse
import os
import sys
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessor import DataPreprocessor
from src.data.dataset_loader import DatasetLoader
from src.data.task_dataset import TaskAwareDataset


def main():
    parser = argparse.ArgumentParser(description='数据预处理工具')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用的命令')
    
    # 列出可用数据集
    list_parser = subparsers.add_parser('list', help='列出所有可用的数据集')
    
    # 下载单个数据集
    download_parser = subparsers.add_parser('download', help='下载单个数据集')
    download_parser.add_argument('dataset', help='数据集名称')
    download_parser.add_argument('--split', default='train', help='数据集分割 (train/validation/test)')
    download_parser.add_argument('--max-samples', type=int, help='最大样本数量')
    download_parser.add_argument('--output-dir', default='./processed_data', help='输出目录')
    
    # 创建混合数据集
    mixed_parser = subparsers.add_parser('create-mixed', help='创建混合数据集')
    mixed_parser.add_argument('datasets', nargs='+', help='数据集名称列表')
    mixed_parser.add_argument('--max-samples-per-dataset', type=int, default=1000, 
                            help='每个数据集的最大样本数')
    mixed_parser.add_argument('--split', default='train', help='数据集分割')
    mixed_parser.add_argument('--output-dir', default='./processed_data', help='输出目录')
    mixed_parser.add_argument('--cache-dir', default='./data_cache', help='数据缓存目录')
    
    # 预处理用于任务检测
    task_parser = subparsers.add_parser('preprocess-task', help='预处理数据用于任务检测')
    task_parser.add_argument('datasets', nargs='+', help='数据集名称列表')
    task_parser.add_argument('--max-samples-per-dataset', type=int, default=500, 
                           help='每个数据集的最大样本数')
    task_parser.add_argument('--output-dir', default='./task_detection_data', help='输出目录')
    
    # 预处理用于适配器训练
    adapter_parser = subparsers.add_parser('preprocess-adapter', help='预处理数据用于适配器训练')
    adapter_parser.add_argument('datasets', nargs='+', help='数据集名称列表')
    adapter_parser.add_argument('--max-samples-per-dataset', type=int, default=1000, 
                              help='每个数据集的最大样本数')
    adapter_parser.add_argument('--tokenizer', default='bert-base-uncased', help='分词器名称')
    adapter_parser.add_argument('--output-dir', default='./adapter_training_data', help='输出目录')
    
    # 验证数据集
    validate_parser = subparsers.add_parser('validate', help='验证处理后的数据集')
    validate_parser.add_argument('data_dir', help='数据目录路径')
    
    # 统计信息
    stats_parser = subparsers.add_parser('stats', help='显示数据集统计信息')
    stats_parser.add_argument('data_dir', help='数据目录路径')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_datasets()
    elif args.command == 'download':
        download_dataset(args)
    elif args.command == 'create-mixed':
        create_mixed_dataset(args)
    elif args.command == 'preprocess-task':
        preprocess_for_task_detection(args)
    elif args.command == 'preprocess-adapter':
        preprocess_for_adapter_training(args)
    elif args.command == 'validate':
        validate_dataset(args)
    elif args.command == 'stats':
        show_statistics(args)
    else:
        parser.print_help()


def list_datasets():
    """列出所有可用的数据集"""
    preprocessor = DataPreprocessor()
    datasets = preprocessor.list_available_datasets()
    
    print("可用数据集:")
    print("=" * 60)
    
    # 按任务类型分组
    task_groups = {}
    for name, config in datasets.items():
        task_type = config['task_type']
        if task_type not in task_groups:
            task_groups[task_type] = []
        task_groups[task_type].append(name)
    
    for task_type, dataset_names in task_groups.items():
        print(f"\n📋 {task_type.replace('_', ' ').title()}:")
        for dataset_name in sorted(dataset_names):
            config = datasets[dataset_name]
            print(f"  • {dataset_name}")
            print(f"    HuggingFace: {config['huggingface_name']}")
            if 'subset' in config:
                print(f"    子集: {config['subset']}")


def download_dataset(args):
    """下载单个数据集"""
    print(f"下载数据集: {args.dataset}")
    
    preprocessor = DataPreprocessor()
    
    try:
        # 任务检测数据
        task_data = preprocessor.preprocess_for_task_detection(
            args.dataset, args.split, args.max_samples
        )
        
        # 适配器训练数据
        adapter_data = preprocessor.preprocess_for_adapter_training(
            args.dataset, args.split, args.max_samples
        )
        
        # 保存数据
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 保存任务检测数据
        task_file = output_dir / f"{args.dataset}_task_detection.json"
        with open(task_file, 'w', encoding='utf-8') as f:
            json.dump(task_data, f, ensure_ascii=False, indent=2)
        
        # 保存适配器训练数据
        adapter_file = output_dir / f"{args.dataset}_adapter_training.json"
        with open(adapter_file, 'w', encoding='utf-8') as f:
            json.dump(adapter_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 数据集 {args.dataset} 下载完成")
        print(f"   任务检测数据: {len(task_data)} 条记录 -> {task_file}")
        print(f"   适配器训练数据: {len(adapter_data['input_texts'])} 条记录 -> {adapter_file}")
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")


def create_mixed_dataset(args):
    """创建混合数据集"""
    print(f"创建混合数据集: {args.datasets}")
    
    preprocessor = DataPreprocessor(cache_dir=args.cache_dir)
    
    try:
        mixed_data = preprocessor.create_mixed_dataset(
            args.datasets,
            max_samples_per_dataset=args.max_samples_per_dataset,
            split=args.split
        )
        
        # 保存数据
        preprocessor.save_processed_data(mixed_data, args.output_dir)
        
        print(f"✅ 混合数据集创建完成")
        print(f"   输出目录: {args.output_dir}")
        print(f"   任务检测数据: {len(mixed_data['task_detection_data'])} 条记录")
        print(f"   适配器训练数据: {len(mixed_data['adapter_training_data']['input_texts'])} 条记录")
        
    except Exception as e:
        print(f"❌ 创建失败: {e}")


def preprocess_for_task_detection(args):
    """预处理数据用于任务检测"""
    print(f"预处理任务检测数据: {args.datasets}")
    
    preprocessor = DataPreprocessor()
    all_task_data = []
    
    for dataset_name in args.datasets:
        try:
            task_data = preprocessor.preprocess_for_task_detection(
                dataset_name, 'train', args.max_samples_per_dataset
            )
            all_task_data.extend(task_data)
            print(f"  ✅ {dataset_name}: {len(task_data)} 条记录")
            
        except Exception as e:
            print(f"  ❌ {dataset_name}: {e}")
    
    # 保存数据
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_file = output_dir / 'task_detection_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_task_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 任务检测数据预处理完成")
    print(f"   总计: {len(all_task_data)} 条记录")
    print(f"   输出文件: {output_file}")


def preprocess_for_adapter_training(args):
    """预处理数据用于适配器训练"""
    print(f"预处理适配器训练数据: {args.datasets}")
    
    preprocessor = DataPreprocessor()
    all_adapter_data = {
        'input_texts': [],
        'target_texts': [],
        'labels': [],
        'task_types': [],
        'dataset_sources': []
    }
    
    for dataset_name in args.datasets:
        try:
            adapter_data = preprocessor.preprocess_for_adapter_training(
                dataset_name, 'train', args.max_samples_per_dataset, args.tokenizer
            )
            
            all_adapter_data['input_texts'].extend(adapter_data['input_texts'])
            all_adapter_data['target_texts'].extend(adapter_data['target_texts'])
            all_adapter_data['labels'].extend(adapter_data['labels'])
            all_adapter_data['task_types'].extend([adapter_data['task_type']] * len(adapter_data['input_texts']))
            all_adapter_data['dataset_sources'].extend([adapter_data['dataset_source']] * len(adapter_data['input_texts']))
            
            print(f"  ✅ {dataset_name}: {len(adapter_data['input_texts'])} 条记录")
            
        except Exception as e:
            print(f"  ❌ {dataset_name}: {e}")
    
    # 保存数据
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_file = output_dir / 'adapter_training_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_adapter_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 适配器训练数据预处理完成")
    print(f"   总计: {len(all_adapter_data['input_texts'])} 条记录")
    print(f"   输出文件: {output_file}")


def validate_dataset(args):
    """验证处理后的数据集"""
    print(f"验证数据集: {args.data_dir}")
    
    try:
        loader = DatasetLoader(args.data_dir)
        
        # 检查任务检测数据
        task_dataset = loader.get_task_detection_dataset()
        print(f"✅ 任务检测数据集: {len(task_dataset)} 条记录")
        
        # 检查适配器训练数据
        adapter_dataset = loader.get_adapter_training_dataset()
        print(f"✅ 适配器训练数据集: {len(adapter_dataset)} 条记录")
        
        # 测试数据加载
        task_dataloader = loader.get_dataloader(task_dataset, batch_size=2)
        adapter_dataloader = loader.get_dataloader(adapter_dataset, batch_size=2)
        
        # 测试第一个批次
        for batch in task_dataloader:
            print(f"✅ 任务检测批次形状: {batch['input_ids'].shape}")
            break
        
        for batch in adapter_dataloader:
            print(f"✅ 适配器训练批次形状: {batch['input_ids'].shape}")
            break
        
        print("✅ 数据集验证通过")
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")


def show_statistics(args):
    """显示数据集统计信息"""
    print(f"数据集统计信息: {args.data_dir}")
    
    try:
        loader = DatasetLoader(args.data_dir)
        stats = loader.get_data_statistics()
        
        print("\n" + "=" * 60)
        print("📊 数据集统计")
        print("=" * 60)
        
        if 'task_detection' in stats:
            print("\n🔍 任务检测数据:")
            td_stats = stats['task_detection']
            print(f"   总样本数: {td_stats['total_samples']}")
            print(f"   任务分布:")
            for task, count in td_stats['task_distribution'].items():
                percentage = (count / td_stats['total_samples']) * 100
                print(f"     • {task}: {count} ({percentage:.1f}%)")
        
        if 'adapter_training' in stats:
            print("\n🔧 适配器训练数据:")
            at_stats = stats['adapter_training']
            print(f"   总样本数: {at_stats['total_samples']}")
            print(f"   任务分布:")
            for task, count in at_stats['task_distribution'].items():
                percentage = (count / at_stats['total_samples']) * 100
                print(f"     • {task}: {count} ({percentage:.1f}%)")
            print(f"   数据集分布:")
            for dataset, count in at_stats['dataset_distribution'].items():
                percentage = (count / at_stats['total_samples']) * 100
                print(f"     • {dataset}: {count} ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"❌ 获取统计信息失败: {e}")


if __name__ == "__main__":
    main()
