#!/usr/bin/env python3
"""
任务感知多模型协作系统测试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.tasks.detector import TaskType, TaskDetector
from src.core.adapters import TaskAwareAdapter
from src.core.collaborator import MultiModelCollaborator
from src.training.task_aware_trainer import TaskAwareTrainer
from transformers import AutoModel

def test_task_detector():
    """测试任务检测器"""
    print("=" * 50)
    print("测试任务检测器")
    print("=" * 50)
    
    detector = TaskDetector()
    
    test_texts = [
        "What is the capital of France?",
        "I love this movie!",
        "Generate a story about dragons.",
        "Hello, how are you?",
        "The weather is nice today.",
        "How does machine learning work?",
        "This product is terrible.",
        "Write a poem about love.",
        "Good morning everyone!",
        "Python is a programming language."
    ]
    
    for text in test_texts:
        detected_task = detector.detect_task(text)
        print(f"Text: '{text}'")
        print(f"Detected Task: {detected_task.value}")
        print("-" * 30)

def test_task_aware_adapter():
    """测试任务感知适配器"""
    print("=" * 50)
    print("测试任务感知适配器")
    print("=" * 50)
    
    # 创建测试适配器
    task_types = [TaskType.QUESTION_ANSWERING, TaskType.SENTIMENT_ANALYSIS, TaskType.GENERAL]
    adapter = TaskAwareAdapter(768, 512, task_types)
    
    # 创建测试数据
    hidden_state = torch.randn(1, 10, 768)  # batch_size=1, seq_len=10, hidden_dim=768
    
    print(f"输入hidden state shape: {hidden_state.shape}")
    
    # 测试不同任务类型的适配
    for task_type in task_types:
        output = adapter(hidden_state, task_type)
        print(f"任务 {task_type.value} 输出 shape: {output.shape}")
    
    # 测试自动混合
    mixed_output = adapter(hidden_state, None)
    print(f"自动混合输出 shape: {mixed_output.shape}")

def test_minimal_collaboration():
    """测试最小化的协作功能（使用小模型）"""
    print("=" * 50)
    print("测试最小化任务感知协作")
    print("=" * 50)
    
    try:
        # 使用较小的模型进行快速测试
        print("正在加载模型...")
        model1 = AutoModel.from_pretrained("distilbert-base-uncased")
        model2 = AutoModel.from_pretrained("distilbert-base-uncased")  # 使用相同模型简化测试
        
        print("创建协作系统...")
        collaborator = MultiModelCollaborator([model1, model2])
        
        # 测试文本
        test_texts = [
            "What is machine learning?",
            "I love programming!",
            "Hello there!"
        ]
        
        print("测试任务感知协作...")
        for text in test_texts:
            print(f"\n处理文本: '{text}'")
            
            # 检测任务类型
            detected_task = collaborator.central_processor.task_detector.detect_task(text)
            print(f"检测到任务类型: {detected_task.value}")
            
            # 进行协作
            try:
                outputs = collaborator.collaborate(text, 0, 1, task_type=detected_task)
                print(f"协作成功！输出shape: {outputs['adapted_hidden'].shape}")
            except Exception as e:
                print(f"协作失败: {e}")
        
        print("\n最小化测试完成！")
        
    except Exception as e:
        print(f"模型加载失败，跳过协作测试: {e}")

def main():
    """主测试函数"""
    print("开始任务感知多模型协作系统测试")
    print("=" * 60)
    
    # 1. 测试任务检测器
    test_task_detector()
    
    # 2. 测试任务感知适配器
    test_task_aware_adapter()
    
    # 3. 测试最小化协作功能
    test_minimal_collaboration()
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("\n如需运行完整演示，请使用:")
    print("  python Multi.py task_aware")

if __name__ == "__main__":
    main()
