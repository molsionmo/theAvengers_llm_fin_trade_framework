#!/usr/bin/env python3
"""
数据预处理示例脚本
演示如何使用数据预处理功能
"""

import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessor import DataPreprocessor
from src.data.dataset_loader import DatasetLoader
from src.data.task_dataset import TaskAwareDataset


def demo_dataset_preprocessing():
    """演示数据集预处理"""
    print("🎯 数据预处理演示")
    print("=" * 60)
    
    # 1. 创建预处理器
    print("\n1️⃣ 创建数据预处理器...")
    preprocessor = DataPreprocessor(cache_dir="./demo_cache")
    
    # 2. 列出可用数据集
    print("\n2️⃣ 可用数据集:")
    datasets = preprocessor.list_available_datasets()
    
    # 按任务类型分组显示
    task_groups = {}
    for name, config in datasets.items():
        task_type = config['task_type']
        if task_type not in task_groups:
            task_groups[task_type] = []
        task_groups[task_type].append(name)
    
    for task_type, dataset_names in task_groups.items():
        print(f"\n   📋 {task_type.replace('_', ' ').title()}:")
        for dataset_name in sorted(dataset_names)[:3]:  # 只显示前3个
            print(f"     • {dataset_name}")
    
    # 3. 选择少量数据集进行演示
    demo_datasets = ['imdb', 'ag_news']  # 选择两个较小的数据集
    print(f"\n3️⃣ 选择演示数据集: {demo_datasets}")
    
    try:
        # 4. 创建混合数据集
        print("\n4️⃣ 创建混合数据集...")
        mixed_data = preprocessor.create_mixed_dataset(
            dataset_names=demo_datasets,
            max_samples_per_dataset=50,  # 每个数据集只取50个样本用于演示
            split='train'
        )
        
        print(f"   ✅ 任务检测数据: {len(mixed_data['task_detection_data'])} 条记录")
        print(f"   ✅ 适配器训练数据: {len(mixed_data['adapter_training_data']['input_texts'])} 条记录")
        
        # 5. 保存数据
        print("\n5️⃣ 保存处理后的数据...")
        output_dir = "./demo_processed_data"
        preprocessor.save_processed_data(mixed_data, output_dir)
        print(f"   📁 数据已保存到: {output_dir}")
        
        # 6. 加载和验证数据
        print("\n6️⃣ 加载和验证数据...")
        loader = DatasetLoader(output_dir)
        
        # 获取任务检测数据集
        task_dataset = loader.get_task_detection_dataset()
        print(f"   🔍 任务检测数据集: {len(task_dataset)} 条记录")
        
        # 测试第一个样本
        sample = task_dataset[0]
        print(f"   📝 示例样本:")
        print(f"      任务类型: {sample['task_type']}")
        print(f"      文本: {sample['text'][:100]}...")
        
        # 获取适配器训练数据集
        adapter_dataset = loader.get_adapter_training_dataset()
        print(f"   🔧 适配器训练数据集: {len(adapter_dataset)} 条记录")
        
        # 7. 创建任务感知数据集
        print("\n7️⃣ 创建任务感知数据集...")
        task_aware_dataset = TaskAwareDataset(
            adapter_dataset.data,
            adapter_dataset.tokenizer,
            max_length=128,  # 短序列用于演示
            task_sampling_strategy='balanced',
            include_task_tokens=True
        )
        
        # 显示统计信息
        stats = task_aware_dataset.get_task_statistics()
        print(f"   📊 数据集统计:")
        print(f"      总样本数: {stats['total_samples']}")
        print(f"      任务类型数: {stats['num_tasks']}")
        print(f"      平均输入长度: {stats['avg_input_length']:.1f} 词")
        print(f"      平均目标长度: {stats['avg_target_length']:.1f} 词")
        
        print(f"   📈 任务分布:")
        for task, info in stats['task_distribution'].items():
            print(f"      • {task}: {info['count']} ({info['percentage']:.1f}%)")
        
        # 8. 测试任务特定批次
        print("\n8️⃣ 测试任务特定采样...")
        if 'sentiment_analysis' in task_aware_dataset.task_indices:
            sentiment_batch = task_aware_dataset.get_task_batch('sentiment_analysis', 2)
            print(f"   😊 情感分析批次: {len(sentiment_batch)} 个样本")
        
        # 9. 测试混合任务批次
        mixed_batch = task_aware_dataset.get_mixed_task_batch(3)
        print(f"   🔀 混合任务批次: {len(mixed_batch)} 个样本")
        task_types = [item['task_type'] for item in mixed_batch]
        print(f"      任务类型: {task_types}")
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")
        print("   💡 提示: 确保网络连接正常，某些数据集需要从HuggingFace下载")
        import traceback
        traceback.print_exc()


def demo_training_with_datasets():
    """演示使用数据集进行训练"""
    print("\n" + "🚀 数据集训练演示")
    print("=" * 60)
    
    try:
        from src.core.collaborator import MultiModelCollaborator
        from src.training.alignment_trainer import AlignmentTrainer
        from transformers import AutoModel
        
        # 检查是否有处理好的数据
        data_dir = "./demo_processed_data"
        if not os.path.exists(data_dir):
            print("❌ 未找到处理后的数据，请先运行数据预处理演示")
            return
        
        print("\n1️⃣ 初始化模型...")
        model1 = AutoModel.from_pretrained("bert-base-uncased")
        model2 = AutoModel.from_pretrained("gpt2")
        collaborator = MultiModelCollaborator([model1, model2])
        
        print("\n2️⃣ 配置训练器...")
        dataset_config = {
            'data_dir': data_dir,
            'tokenizer_name': 'bert-base-uncased'
        }
        
        trainer = AlignmentTrainer(
            collaborator, 
            learning_rate=1e-4,
            dataset_config=dataset_config
        )
        
        print("\n3️⃣ 开始训练...")
        results = trainer.train_with_dataset_selection(
            dataset_names=['imdb', 'ag_news'],
            epochs=3,  # 只训练3个epoch用于演示
            batch_size=4,  # 小批次
            max_samples_per_dataset=20,  # 每个数据集只用20个样本
            task_sampling_strategy='balanced',
            validation_split=0.3
        )
        
        print(f"\n✅ 训练完成!")
        print(f"   最终训练损失: {results['final_train_loss']:.4f}")
        print(f"   最佳验证损失: {results['best_val_loss']:.4f}")
        print(f"   训练样本数: {results['dataset_info']['train_samples']}")
        print(f"   验证样本数: {results['dataset_info']['val_samples']}")
        
    except Exception as e:
        print(f"❌ 训练演示失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("🌟 数据预处理和训练演示")
    print("这个演示将展示如何:")
    print("  1. 下载和预处理互联网数据集")
    print("  2. 创建任务感知数据集")
    print("  3. 使用数据集进行适配器训练")
    print("\n⚠️  注意: 首次运行需要下载数据集，可能需要一些时间")
    
    try:
        # 演示数据预处理
        demo_dataset_preprocessing()
        
        # 演示训练
        demo_training_with_datasets()
        
        print("\n" + "🎉 演示完成!")
        print("=" * 60)
        print("您可以通过以下命令使用完整功能:")
        print("  python main.py data list                    # 查看所有可用数据集")
        print("  python main.py data create-mixed imdb squad # 创建混合数据集")
        print("  python main.py train-with-data imdb squad   # 使用数据集训练")
        
    except KeyboardInterrupt:
        print("\n⏹️ 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")


if __name__ == "__main__":
    main()
