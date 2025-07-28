#!/usr/bin/env python3
"""
任务感知多模型协作框架主入口

使用方法:
    python main.py demo basic           # 基础演示
    python main.py demo training       # 训练演示
    python main.py demo comparison     # 对比演示
    python main.py demo interactive    # 交互式演示
    python main.py test                # 运行测试
    python main.py quick-start         # 快速开始
    python main.py data list           # 列出可用数据集
    python main.py data create-mixed dataset1 dataset2  # 创建混合数据集
    python main.py train-with-data dataset1 dataset2 --epochs 10  # 使用数据集训练
"""

import sys
import os
import argparse

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def run_demo(mode):
    """运行演示"""
    from examples.demo import main as demo_main
    sys.argv = ['demo.py', mode]
    demo_main()


def run_quick_start():
    """运行快速开始"""
    from examples.quick_start import quick_start
    quick_start()


def run_tests():
    """运行测试"""
    import subprocess
    test_files = [
        'tests/task_aware_test.py',
        'tests/test_collaboration.py',
        'tests/quick_test.py'
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n🧪 运行测试: {test_file}")
            try:
                subprocess.run([sys.executable, test_file], check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ 测试失败: {test_file}")
                print(f"错误: {e}")
            except KeyboardInterrupt:
                print("\n⏹️ 测试被用户中断")
                break
        else:
            print(f"⚠️ 测试文件不存在: {test_file}")


def show_project_info():
    """显示项目信息"""
    print("🤖 任务感知多模型协作框架")
    print("=" * 50)
    print("版本: 1.0.0")
    print("作者: Task-Aware Collaboration Team")
    print()
    print("📋 项目结构:")
    
    # 显示项目结构
    def show_tree(path, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
        
        items = []
        try:
            for item in sorted(os.listdir(path)):
                if not item.startswith('.') and item != '__pycache__':
                    items.append(item)
        except PermissionError:
            return
        
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            item_path = os.path.join(path, item)
            
            if is_last:
                print(f"{prefix}└── {item}")
                new_prefix = prefix + "    "
            else:
                print(f"{prefix}├── {item}")
                new_prefix = prefix + "│   "
            
            if os.path.isdir(item_path) and not item.startswith('.'):
                show_tree(item_path, new_prefix, max_depth, current_depth + 1)
    
    show_tree(".")
    
    print("\n📦 核心组件:")
    print("  • MultiModelCollaborator: 多模型协作系统")
    print("  • TaskDetector: 任务类型检测器")
    print("  • TaskAwareAdapter: 任务感知适配器")
    print("  • TaskAwareTrainer: 任务感知训练器")
    
    print("\n🎯 支持的任务类型:")
    print("  • 问答 (QA)")
    print("  • 情感分析 (Sentiment)")
    print("  • 文本生成 (Generation)")
    print("  • 对话 (Conversation)")
    print("  • 通用任务 (General)")


def run_data_command(args):
    """运行数据相关命令"""
    import subprocess
    
    # 构建命令
    cmd = [sys.executable, 'scripts/preprocess_data.py', args.data_command]
    
    if hasattr(args, 'datasets') and args.datasets:
        cmd.extend(args.datasets)
    
    # 只为需要这些参数的命令添加参数
    if args.data_command in ['create-mixed', 'preprocess-task', 'preprocess-adapter']:
        if hasattr(args, 'max_samples_per_dataset') and args.max_samples_per_dataset:
            cmd.extend(['--max-samples-per-dataset', str(args.max_samples_per_dataset)])
        
        if hasattr(args, 'output_dir') and args.output_dir:
            cmd.extend(['--output-dir', args.output_dir])
        
        if hasattr(args, 'split') and args.split:
            cmd.extend(['--split', args.split])
    
    # 执行命令
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 数据处理命令执行失败: {e}")


def train_with_datasets(args):
    """使用数据集进行训练"""
    print(f"🚀 开始使用数据集训练: {args.datasets}")
    
    from src.core.collaborator import MultiModelCollaborator
    from src.training.alignment_trainer import AlignmentTrainer
    from transformers import AutoModel
    
    try:
        # 初始化模型
        print("初始化模型...")
        model1 = AutoModel.from_pretrained("bert-base-uncased")
        model2 = AutoModel.from_pretrained("gpt2")
        collaborator = MultiModelCollaborator([model1, model2])
        
        # 配置数据集
        dataset_config = {
            'data_dir': args.data_dir,
            'tokenizer_name': args.tokenizer
        }
        
        # 初始化训练器
        trainer = AlignmentTrainer(
            collaborator, 
            learning_rate=args.learning_rate,
            dataset_config=dataset_config
        )
        
        # 开始训练
        results = trainer.train_with_dataset_selection(
            dataset_names=args.datasets,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_samples_per_dataset=args.max_samples_per_dataset,
            task_sampling_strategy=args.sampling_strategy,
            validation_split=args.validation_split
        )
        
        print(f"\n✅ 训练完成!")
        print(f"   最终训练损失: {results['final_train_loss']:.4f}")
        print(f"   最佳验证损失: {results['best_val_loss']:.4f}")
        print(f"   训练样本数: {results['dataset_info']['train_samples']}")
        print(f"   验证样本数: {results['dataset_info']['val_samples']}")
        
        # 保存训练结果
        import json
        from pathlib import Path
        
        results_dir = Path('./results')
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / 'dataset_training_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"   训练结果已保存到: {results_file}")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="任务感知多模型协作框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py info                    # 显示项目信息
  python main.py demo basic             # 基础演示
  python main.py demo training          # 训练演示
  python main.py demo comparison        # 对比演示
  python main.py demo interactive       # 交互式演示
  python main.py test                   # 运行测试
  python main.py quick-start            # 快速开始
  python main.py data list              # 列出可用数据集
  python main.py data create-mixed imdb ag_news --output-dir ./data  # 创建混合数据集
  python main.py train-with-data imdb ag_news --epochs 10            # 使用数据集训练
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # info命令
    subparsers.add_parser('info', help='显示项目信息')
    
    # demo命令
    demo_parser = subparsers.add_parser('demo', help='运行演示')
    demo_parser.add_argument(
        'mode', 
        choices=['basic', 'training', 'comparison', 'interactive'],
        help='演示模式'
    )
    
    # test命令
    subparsers.add_parser('test', help='运行测试')
    
    # quick-start命令
    subparsers.add_parser('quick-start', help='快速开始')
    
    # data命令
    data_parser = subparsers.add_parser('data', help='数据处理命令')
    data_parser.add_argument(
        'data_command',
        choices=['list', 'create-mixed', 'preprocess-task', 'preprocess-adapter', 'validate', 'stats'],
        help='数据处理操作'
    )
    data_parser.add_argument('datasets', nargs='*', help='数据集名称列表')
    data_parser.add_argument('--max-samples-per-dataset', type=int, default=1000, help='每个数据集的最大样本数')
    data_parser.add_argument('--output-dir', default='./processed_data', help='输出目录')
    data_parser.add_argument('--split', default='train', help='数据集分割')
    
    # train-with-data命令
    train_parser = subparsers.add_parser('train-with-data', help='使用数据集进行训练')
    train_parser.add_argument('datasets', nargs='+', help='数据集名称列表')
    train_parser.add_argument('--data-dir', default='./processed_data', help='数据目录')
    train_parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    train_parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    train_parser.add_argument('--learning-rate', type=float, default=1e-4, help='学习率')
    train_parser.add_argument('--max-samples-per-dataset', type=int, default=1000, help='每个数据集的最大样本数')
    train_parser.add_argument('--sampling-strategy', default='balanced', 
                            choices=['balanced', 'proportional', 'random'], help='任务采样策略')
    train_parser.add_argument('--validation-split', type=float, default=0.2, help='验证集比例')
    train_parser.add_argument('--tokenizer', default='bert-base-uncased', help='分词器名称')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'info':
            show_project_info()
        elif args.command == 'demo':
            run_demo(args.mode)
        elif args.command == 'test':
            run_tests()
        elif args.command == 'quick-start':
            run_quick_start()
        elif args.command == 'data':
            run_data_command(args)
        elif args.command == 'train-with-data':
            train_with_datasets(args)
            
    except KeyboardInterrupt:
        print("\n⏹️ 操作被用户中断")
    except Exception as e:
        print(f"❌ 执行命令时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
