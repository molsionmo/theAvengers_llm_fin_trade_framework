#!/usr/bin/env python3
"""
快速开始示例

展示如何快速使用任务感知多模型协作框架
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from transformers import AutoModel
from src.core.collaborator import MultiModelCollaborator
from src.tasks.detector import TaskType


def quick_start():
    """快速开始示例"""
    
    # 1. 加载预训练模型
    print("📦 加载模型...")
    bert_model = AutoModel.from_pretrained("bert-base-uncased")
    gpt2_model = AutoModel.from_pretrained("gpt2")
    
    # 2. 创建多模型协作系统
    print("🔧 创建协作系统...")
    collaborator = MultiModelCollaborator([bert_model, gpt2_model])
    
    # 3. 任务检测示例
    print("\n🎯 任务检测示例:")
    texts = [
        "What is machine learning?",      # 问答
        "I love this product!",           # 情感分析
        "Write a creative story",         # 文本生成
    ]
    
    for text in texts:
        task = collaborator.detect_task_for_text(text)
        print(f"'{text}' -> {task.value}")
    
    # 4. 模型协作示例
    print("\n🤝 模型协作示例:")
    text = "What is the capital of France?"
    
    # 从BERT获取信息，传递给GPT-2
    result = collaborator.collaborate(text, source_model_idx=0, target_model_idx=1)
    print(f"文本: '{text}'")
    print(f"源模型(BERT) hidden shape: {result['source_hidden'].shape}")
    print(f"适配后 hidden shape: {result['adapted_hidden'].shape}")
    print(f"目标模型(GPT-2) hidden shape: {result['normal_hidden'].shape}")
    
    # 5. 任务感知协作
    print("\n🧠 任务感知协作:")
    qa_result = collaborator.collaborate(
        text, 
        source_model_idx=0, 
        target_model_idx=1, 
        task_type=TaskType.QUESTION_ANSWERING
    )
    print(f"任务感知适配后 shape: {qa_result['adapted_hidden'].shape}")
    print(f"检测到的任务类型: {qa_result['task_type'].value}")
    
    print("\n✅ 快速开始完成!")


if __name__ == "__main__":
    quick_start()
