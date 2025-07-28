#!/usr/bin/env python3
"""
任务感知多模型协作系统演示

展示如何使用任务感知的多模型协作框架
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from transformers import AutoModel
from src.core.collaborator import MultiModelCollaborator
from src.tasks.detector import TaskType
from src.training.task_aware_trainer import TaskAwareTrainer


def basic_demo():
    """基础演示：任务检测和协作"""
    print("🚀 任务感知多模型协作基础演示")
    print("=" * 60)
    
    # 加载模型
    print("正在加载模型...")
    model1 = AutoModel.from_pretrained("bert-base-uncased")
    model2 = AutoModel.from_pretrained("gpt2")
    
    # 创建协作系统
    collaborator = MultiModelCollaborator([model1, model2])
    
    # 测试文本
    test_texts = [
        "What is machine learning?",
        "I love this product!",
        "Write a story about space",
        "Hello, how are you?",
        "The weather is nice today."
    ]
    
    print("\n📋 任务检测演示:")
    for text in test_texts:
        detected_task = collaborator.detect_task_for_text(text)
        confidence = collaborator.get_task_confidence(text)
        
        print(f"文本: '{text}'")
        print(f"  检测任务: {detected_task.value}")
        print(f"  置信度: {max(confidence.values()):.3f}")
        print()
    
    print("📊 协作演示:")
    test_text = "What is the capital of France?"
    result = collaborator.collaborate(test_text, 0, 1)
    print(f"文本: '{test_text}'")
    print(f"适配后hidden shape: {result['adapted_hidden'].shape}")
    print(f"原始hidden shape: {result['normal_hidden'].shape}")
    
    print("\n✅ 基础演示完成!")


def training_demo():
    """训练演示：任务感知训练"""
    print("🔧 任务感知训练演示")
    print("=" * 60)
    
    # 加载模型
    print("正在加载模型...")
    model1 = AutoModel.from_pretrained("bert-base-uncased")
    model2 = AutoModel.from_pretrained("gpt2")
    
    # 创建协作系统
    collaborator = MultiModelCollaborator([model1, model2])
    
    # 创建训练器
    trainer = TaskAwareTrainer(collaborator, learning_rate=1e-4)
    
    # 准备训练数据
    train_texts = [
        "What is artificial intelligence?",
        "How does neural network work?",
        "I love machine learning!",
        "This framework is amazing!",
        "Write a poem about technology",
        "Create a story about robots",
        "Hello there!",
        "Good morning everyone!",
        "The sky is blue today",
        "Technology advances rapidly"
    ]
    
    print("📈 数据集任务分布:")
    distribution = trainer.get_task_distribution(train_texts)
    for task, ratio in distribution.items():
        print(f"  {task}: {ratio:.2%}")
    
    print("\n🏋️ 开始训练...")
    results = trainer.train_with_task_awareness(train_texts, epochs=3)
    
    print("\n📊 训练结果摘要:")
    summary = trainer.get_task_performance_summary()
    for task, metrics in summary.items():
        print(f"{task}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\n✅ 训练演示完成!")


def comparison_demo():
    """对比演示：任务感知 vs 通用协作"""
    print("⚖️ 任务感知 vs 通用协作对比")
    print("=" * 60)
    
    # 加载模型
    print("正在加载模型...")
    model1 = AutoModel.from_pretrained("bert-base-uncased")
    model2 = AutoModel.from_pretrained("gpt2")
    
    # 创建协作系统
    collaborator = MultiModelCollaborator([model1, model2])
    
    # 测试用例
    test_cases = [
        ("What is the capital of Japan?", TaskType.QUESTION_ANSWERING),
        ("I hate this movie!", TaskType.SENTIMENT_ANALYSIS),
        ("Generate a creative story", TaskType.TEXT_GENERATION),
    ]
    
    print("\n🔍 协作效果对比:")
    for text, task_type in test_cases:
        print(f"\n文本: '{text}' (任务: {task_type.value})")
        
        # 通用协作
        general_result = collaborator.collaborate(text, 0, 1, task_type=None)
        
        # 任务感知协作
        task_aware_result = collaborator.collaborate(text, 0, 1, task_type=task_type)
        
        # 计算差异
        import torch.nn.functional as F
        similarity_general = F.cosine_similarity(
            general_result['source_hidden'].mean(dim=1),
            general_result['adapted_hidden'].mean(dim=1)
        ).item()
        
        similarity_task_aware = F.cosine_similarity(
            task_aware_result['source_hidden'].mean(dim=1),
            task_aware_result['adapted_hidden'].mean(dim=1)
        ).item()
        
        print(f"  通用协作相似度: {similarity_general:.4f}")
        print(f"  任务感知相似度: {similarity_task_aware:.4f}")
        print(f"  改善程度: {similarity_task_aware - similarity_general:+.4f}")
    
    print("\n✅ 对比演示完成!")


def interactive_demo():
    """交互式演示"""
    print("🎮 交互式任务感知协作演示")
    print("=" * 60)
    
    # 加载模型
    print("正在加载模型...")
    model1 = AutoModel.from_pretrained("bert-base-uncased")
    model2 = AutoModel.from_pretrained("gpt2")
    
    # 创建协作系统
    collaborator = MultiModelCollaborator([model1, model2])
    
    print("✨ 输入文本来测试任务检测和协作效果")
    print("输入 'quit' 退出演示")
    print()
    
    while True:
        try:
            user_input = input("请输入文本: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            # 检测任务
            detected_task = collaborator.detect_task_for_text(user_input)
            confidence = collaborator.get_task_confidence(user_input)
            
            print(f"🎯 检测结果:")
            print(f"  任务类型: {detected_task.value}")
            print(f"  置信度: {max(confidence.values()):.3f}")
            
            # 协作测试
            result = collaborator.collaborate(user_input, 0, 1, task_type=detected_task)
            print(f"📊 协作结果:")
            print(f"  适配维度: {result['adapted_hidden'].shape}")
            print(f"  任务类型: {result['task_type'].value if result['task_type'] else 'None'}")
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")
            print()
    
    print("👋 再见!")


def main():
    """主函数"""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "basic":
            basic_demo()
        elif mode == "training":
            training_demo()
        elif mode == "comparison":
            comparison_demo()
        elif mode == "interactive":
            interactive_demo()
        elif mode == "help":
            print("🔧 任务感知多模型协作演示")
            print("使用方法:")
            print("  python demo.py basic        - 基础功能演示")
            print("  python demo.py training     - 训练过程演示")
            print("  python demo.py comparison   - 效果对比演示")
            print("  python demo.py interactive  - 交互式演示")
            print("  python demo.py help         - 显示此帮助信息")
        else:
            print(f"未知模式: {mode}")
            print("使用 'python demo.py help' 查看可用模式")
    else:
        # 默认运行基础演示
        basic_demo()


if __name__ == "__main__":
    main()
