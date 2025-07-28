#!/usr/bin/env python3
"""
任务感知协作测试脚本：专门测试不同任务类型下的协作效果
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel
from Multi import (
    MultiModelCollaborator, TaskType, TaskDetector, TaskAwareTrainer
)

def test_task_detection(test_texts):
    """测试任务检测功能"""
    print("\n🎯 任务检测测试")
    print("=" * 50)
    
    detector = TaskDetector()
    detection_results = {}
    
    for text, expected_task in test_texts:
        detected_task = detector.detect_task(text)
        detection_results[text] = {
            'expected': expected_task,
            'detected': detected_task,
            'correct': detected_task == expected_task
        }
        
        status = "✅" if detected_task == expected_task else "❌"
        print(f"{status} '{text}'")
        print(f"    预期: {expected_task.value}")
        print(f"    检测: {detected_task.value}")
    
    accuracy = sum(1 for r in detection_results.values() if r['correct']) / len(detection_results)
    print(f"\n📊 检测准确率: {accuracy:.2%}")
    
    return detection_results

def test_task_aware_collaboration(collaborator, text, task_type):
    """测试特定任务类型下的协作效果"""
    print(f"\n🔍 任务感知协作测试: {task_type.value}")
    print(f"文本: '{text}'")
    
    # 获取原始hidden states
    hidden1 = collaborator.get_hidden_states(text, 0)  # BERT
    hidden2 = collaborator.get_hidden_states(text, 1)  # GPT-2
    
    # 通用协作（不指定任务类型）
    general_output = collaborator.collaborate(text, 0, 1, task_type=None)
    
    # 任务感知协作
    task_aware_output = collaborator.collaborate(text, 0, 1, task_type=task_type)
    
    # 分析协作效果
    general_hidden = general_output['adapted_hidden']
    task_aware_hidden = task_aware_output['adapted_hidden']
    
    # 为了比较，我们需要将原始BERT hidden states投影到共享空间
    projected_states = collaborator.central_processor.semantic_projector([hidden1])
    projected_bert = projected_states[0]
    
    # 计算与投影后BERT的相似性
    bert_to_general_sim = F.cosine_similarity(
        projected_bert.mean(dim=1), general_hidden.mean(dim=1)
    ).item()
    
    bert_to_task_sim = F.cosine_similarity(
        projected_bert.mean(dim=1), task_aware_hidden.mean(dim=1)
    ).item()
    
    # 计算适配后的信息密度变化
    general_info_density = torch.std(general_hidden).item()
    task_aware_info_density = torch.std(task_aware_hidden).item()
    
    print(f"  📊 相似性分析:")
    print(f"    投影BERT → 通用适配: {bert_to_general_sim:.4f}")
    print(f"    投影BERT → 任务适配: {bert_to_task_sim:.4f}")
    print(f"  📈 信息密度:")
    print(f"    通用适配: {general_info_density:.4f}")
    print(f"    任务适配: {task_aware_info_density:.4f}")
    
    return {
        'task_type': task_type,
        'general_similarity': bert_to_general_sim,
        'task_aware_similarity': bert_to_task_sim,
        'general_info_density': general_info_density,
        'task_aware_info_density': task_aware_info_density,
        'adaptation_difference': abs(bert_to_task_sim - bert_to_general_sim)
    }

def test_task_specific_adaptation(collaborator, task_test_cases):
    """测试不同任务类型的适配效果"""
    print("\n🔧 任务特定适配测试")
    print("=" * 50)
    
    adaptation_results = []
    
    for text, task_type in task_test_cases:
        result = test_task_aware_collaboration(collaborator, text, task_type)
        adaptation_results.append(result)
    
    # 分析任务特定的适配效果
    print(f"\n📊 任务适配效果汇总:")
    task_performance = {}
    
    for result in adaptation_results:
        task = result['task_type'].value
        if task not in task_performance:
            task_performance[task] = []
        task_performance[task].append(result['adaptation_difference'])
    
    for task, differences in task_performance.items():
        avg_difference = np.mean(differences)
        print(f"  {task}: 平均适配差异 = {avg_difference:.4f}")
    
    return adaptation_results

def task_aware_generation_test(collaborator, text, task_type):
    """任务感知的文本生成测试"""
    if not hasattr(collaborator, 'gpt2_generator'):
        collaborator.gpt2_generator = GPT2LMHeadModel.from_pretrained("gpt2")
    
    tokenizer = collaborator.tokenizers[1]  # GPT-2 tokenizer
    
    # 通用协作生成
    general_collab = collaborator.collaborate(text, 0, 1, task_type=None)
    
    # 任务感知协作生成
    task_aware_collab = collaborator.collaborate(text, 0, 1, task_type=task_type)
    
    # 模拟使用不同协作结果进行生成（简化版本）
    inputs = tokenizer.encode_plus(text, return_tensors="pt", padding=True, truncation=True)
    
    # 设置不同的生成参数基于任务类型
    generation_params = {
        TaskType.QUESTION_ANSWERING: {'temperature': 0.7, 'top_p': 0.9},
        TaskType.TEXT_GENERATION: {'temperature': 0.9, 'top_p': 0.95},
        TaskType.CONVERSATION: {'temperature': 0.8, 'top_p': 0.9},
        TaskType.SENTIMENT_ANALYSIS: {'temperature': 0.6, 'top_p': 0.8},
        TaskType.GENERAL: {'temperature': 0.8, 'top_p': 0.9}
    }
    
    params = generation_params.get(task_type, generation_params[TaskType.GENERAL])
    
    with torch.no_grad():
        general_output = collaborator.gpt2_generator.generate(
            inputs['input_ids'],
            max_length=len(inputs['input_ids'][0]) + 15,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        task_aware_output = collaborator.gpt2_generator.generate(
            inputs['input_ids'],
            max_length=len(inputs['input_ids'][0]) + 15,
            num_return_sequences=1,
            temperature=params['temperature'],
            top_p=params['top_p'],
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    general_text = tokenizer.decode(general_output[0], skip_special_tokens=True)
    task_aware_text = tokenizer.decode(task_aware_output[0], skip_special_tokens=True)
    
    return {
        'input': text,
        'task_type': task_type.value,
        'general_generation': general_text,
        'task_aware_generation': task_aware_text,
        'generation_params': params
    }

def task_aware_classification_test(collaborator, text, task_type):
    """任务感知的分类测试"""
    # 通用协作
    general_collab = collaborator.collaborate(text, 0, 1, task_type=None)
    general_hidden = general_collab['adapted_hidden']
    
    # 任务感知协作
    task_aware_collab = collaborator.collaborate(text, 0, 1, task_type=task_type)
    task_aware_hidden = task_aware_collab['adapted_hidden']
    
    # 创建简单分类器（如果不存在）
    if not hasattr(collaborator, 'task_classifiers'):
        collaborator.task_classifiers = {}
    
    if task_type not in collaborator.task_classifiers:
        hidden_size = task_aware_hidden.size(-1)
        if task_type == TaskType.SENTIMENT_ANALYSIS:
            num_classes = 3  # 正面、中性、负面
            class_labels = ['负面', '中性', '正面']
        elif task_type == TaskType.QUESTION_ANSWERING:
            num_classes = 2  # 是问题、不是问题
            class_labels = ['非问题', '问题']
        else:
            num_classes = 2  # 通用二分类
            class_labels = ['类别A', '类别B']
        
        collaborator.task_classifiers[task_type] = {
            'classifier': torch.nn.Linear(hidden_size, num_classes),
            'labels': class_labels
        }
    
    classifier_info = collaborator.task_classifiers[task_type]
    classifier = classifier_info['classifier']
    labels = classifier_info['labels']
    
    with torch.no_grad():
        # 通用协作的分类结果
        general_pooled = general_hidden.mean(dim=1)
        general_logits = classifier(general_pooled)
        general_probs = F.softmax(general_logits, dim=-1)
        general_pred = int(torch.argmax(general_probs, dim=-1).item())
        
        # 任务感知协作的分类结果
        task_aware_pooled = task_aware_hidden.mean(dim=1)
        task_aware_logits = classifier(task_aware_pooled)
        task_aware_probs = F.softmax(task_aware_logits, dim=-1)
        task_aware_pred = int(torch.argmax(task_aware_probs, dim=-1).item())
    
    return {
        'text': text,
        'task_type': task_type.value,
        'general_prediction': {
            'label': labels[general_pred],
            'confidence': general_probs[0][general_pred].item(),
            'all_probs': {labels[i]: general_probs[0][i].item() for i in range(len(labels))}
        },
        'task_aware_prediction': {
            'label': labels[task_aware_pred],
            'confidence': task_aware_probs[0][task_aware_pred].item(),
            'all_probs': {labels[i]: task_aware_probs[0][i].item() for i in range(len(labels))}
        }
    }

def task_aware_quick_test():
    """任务感知的快速测试函数"""
    print("🚀 任务感知多模型协作测试")
    print("=" * 60)
    
    # 初始化模型
    print("正在加载模型...")
    model1 = AutoModel.from_pretrained("bert-base-uncased")
    model2 = AutoModel.from_pretrained("gpt2")
    collaborator = MultiModelCollaborator([model1, model2])
    
    # 定义测试用例：(文本, 预期任务类型)
    task_test_cases = [
        ("What is the capital of France?", TaskType.QUESTION_ANSWERING),
        ("I love this movie, it's amazing!", TaskType.SENTIMENT_ANALYSIS),
        ("Generate a story about dragons.", TaskType.TEXT_GENERATION),
        ("Hello, how are you today?", TaskType.CONVERSATION),
        ("The weather is nice today.", TaskType.GENERAL),
        ("How does machine learning work?", TaskType.QUESTION_ANSWERING),
        ("This book is terrible.", TaskType.SENTIMENT_ANALYSIS),
        ("Write a poem about love.", TaskType.TEXT_GENERATION),
        ("Good morning everyone!", TaskType.CONVERSATION)
    ]
    
    # 第一阶段：任务检测测试
    print("\n📋 第一阶段：任务检测能力测试")
    detection_results = test_task_detection(task_test_cases)
    
    # 第二阶段：任务感知协作测试（训练前）
    print("\n🔍 第二阶段：训练前任务感知协作测试")
    before_adaptation_results = test_task_specific_adaptation(collaborator, task_test_cases[:5])
    
    # 第三阶段：文本生成任务感知测试
    print("\n📝 第三阶段：任务感知文本生成测试")
    generation_test_cases = [
        ("The weather is", TaskType.GENERAL),
        ("What is the answer to", TaskType.QUESTION_ANSWERING),
        ("I feel", TaskType.SENTIMENT_ANALYSIS),
        ("Once upon a time", TaskType.TEXT_GENERATION)
    ]
    
    generation_results = []
    for text, task_type in generation_test_cases:
        print(f"\n测试: '{text}' (任务: {task_type.value})")
        result = task_aware_generation_test(collaborator, text, task_type)
        generation_results.append(result)
        print(f"  通用生成: {result['general_generation']}")
        print(f"  任务感知生成: {result['task_aware_generation']}")
    
    # 第四阶段：分类任务感知测试
    print("\n🏷️ 第四阶段：任务感知分类测试")
    classification_test_cases = [
        ("I love programming!", TaskType.SENTIMENT_ANALYSIS),
        ("This is a terrible movie", TaskType.SENTIMENT_ANALYSIS),
        ("How do you solve this problem?", TaskType.QUESTION_ANSWERING),
        ("Tell me about the weather", TaskType.QUESTION_ANSWERING)
    ]
    
    classification_results = []
    for text, task_type in classification_test_cases:
        print(f"\n测试: '{text}' (任务: {task_type.value})")
        result = task_aware_classification_test(collaborator, text, task_type)
        classification_results.append(result)
        print(f"  通用分类: {result['general_prediction']['label']} "
              f"(置信度: {result['general_prediction']['confidence']:.3f})")
        print(f"  任务感知分类: {result['task_aware_prediction']['label']} "
              f"(置信度: {result['task_aware_prediction']['confidence']:.3f})")
    
    # 第五阶段：任务感知训练
    print("\n🔧 第五阶段：任务感知训练")
    trainer = TaskAwareTrainer(collaborator, learning_rate=1e-4)
    
    # 准备多样化的训练数据
    train_texts = [text for text, _ in task_test_cases] + [
        "Machine learning is powerful",
        "The ocean is vast and deep",
        "Music brings joy to people",
        "How does AI work?",
        "I'm feeling great today!",
        "Continue this story: A brave knight...",
        "Good evening everyone"
    ]
    
    print("开始任务感知训练...")
    trainer.train_with_task_awareness(train_texts, epochs=3)
    
    # 第六阶段：训练后效果对比
    print("\n📊 第六阶段：训练后效果对比")
    after_adaptation_results = test_task_specific_adaptation(collaborator, task_test_cases[:5])
    
    # 生成对比报告
    print("\n📋 第七阶段：任务感知效果总结")
    print("=" * 60)
    
    # 检测准确率
    detection_accuracy = sum(1 for r in detection_results.values() if r['correct']) / len(detection_results)
    print(f"🎯 任务检测准确率: {detection_accuracy:.2%}")
    
    # 适配效果对比
    print(f"\n🔧 任务感知适配效果:")
    task_improvements = {}
    
    for i, (before, after) in enumerate(zip(before_adaptation_results, after_adaptation_results)):
        task_type = before['task_type'].value
        improvement = after['adaptation_difference'] - before['adaptation_difference']
        
        if task_type not in task_improvements:
            task_improvements[task_type] = []
        task_improvements[task_type].append(improvement)
        
        print(f"  {task_type}: 适配差异 {before['adaptation_difference']:.4f} → "
              f"{after['adaptation_difference']:.4f} (变化: {improvement:+.4f})")
    
    # 整体评估
    overall_improvement = np.mean([
        np.mean(improvements) for improvements in task_improvements.values()
    ])
    
    print(f"\n✨ 整体任务感知改善: {overall_improvement:+.4f}")
    
    if overall_improvement > 0.01:
        conclusion = "🎉 任务感知训练显著提升了协作效果！"
    elif overall_improvement > 0.001:
        conclusion = "✅ 任务感知训练有一定改善效果"
    else:
        conclusion = "📈 任务感知训练效果有限，可能需要更多数据或调参"
    
    print(f"  {conclusion}")
    
    # 保存结果
    results = {
        'detection_accuracy': detection_accuracy,
        'detection_results': detection_results,
        'before_adaptation': before_adaptation_results,
        'after_adaptation': after_adaptation_results,
        'generation_results': generation_results,
        'classification_results': classification_results,
        'task_improvements': task_improvements,
        'overall_improvement': overall_improvement,
        'conclusion': conclusion
    }
    
    return results

def save_task_aware_results(results):
    """保存任务感知测试结果"""
    import json
    
    # 准备保存数据（处理不可序列化的对象）
    save_data = {
        'summary': {
            'detection_accuracy': results['detection_accuracy'],
            'overall_improvement': results['overall_improvement'],
            'conclusion': results['conclusion']
        },
        'detection_details': {
            text: {
                'expected': result['expected'].value,
                'detected': result['detected'].value,
                'correct': result['correct']
            }
            for text, result in results['detection_results'].items()
        },
        'generation_examples': [
            {
                'input': r['input'],
                'task_type': r['task_type'],
                'general_generation': r['general_generation'],
                'task_aware_generation': r['task_aware_generation']
            }
            for r in results['generation_results']
        ],
        'classification_examples': [
            {
                'text': r['text'],
                'task_type': r['task_type'],
                'general_prediction': r['general_prediction'],
                'task_aware_prediction': r['task_aware_prediction']
            }
            for r in results['classification_results']
        ],
        'adaptation_improvements': {
            task: improvements for task, improvements in results['task_improvements'].items()
        }
    }
    
    # 保存到JSON文件
    with open('task_aware_results.json', 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    # 生成文本报告
    with open('task_aware_report.txt', 'w', encoding='utf-8') as f:
        f.write("🤖 任务感知多模型协作详细报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"📊 核心指标:\n")
        f.write(f"  任务检测准确率: {results['detection_accuracy']:.2%}\n")
        f.write(f"  整体改善程度: {results['overall_improvement']:+.4f}\n")
        f.write(f"  结论: {results['conclusion']}\n\n")
        
        f.write("🎯 任务检测详情:\n")
        for text, result in results['detection_results'].items():
            status = "✅" if result['correct'] else "❌"
            f.write(f"{status} '{text}'\n")
            f.write(f"    预期: {result['expected'].value} | 检测: {result['detected'].value}\n")
        
        f.write(f"\n📝 生成任务示例:\n")
        for r in results['generation_results']:
            f.write(f"输入: '{r['input']}' (任务: {r['task_type']})\n")
            f.write(f"  通用生成: {r['general_generation']}\n")
            f.write(f"  任务感知生成: {r['task_aware_generation']}\n\n")
        
        f.write(f"🏷️ 分类任务示例:\n")
        for r in results['classification_results']:
            f.write(f"文本: '{r['text']}' (任务: {r['task_type']})\n")
            f.write(f"  通用分类: {r['general_prediction']['label']} "
                   f"(置信度: {r['general_prediction']['confidence']:.3f})\n")
            f.write(f"  任务感知分类: {r['task_aware_prediction']['label']} "
                   f"(置信度: {r['task_aware_prediction']['confidence']:.3f})\n\n")
    
    print(f"\n💾 任务感知测试结果已保存:")
    print(f"  📄 详细数据: task_aware_results.json")
    print(f"  📋 文本报告: task_aware_report.txt")

if __name__ == "__main__":
    """运行任务感知测试"""
    print("🎯 开始任务感知多模型协作测试")
    print("=" * 60)
    
    try:
        results = task_aware_quick_test()
        save_task_aware_results(results)
        
        print("\n🎉 任务感知测试完成！")
        print("查看生成的报告文件了解详细结果。")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        print("请检查模型加载和依赖项是否正确安装。")
        import traceback
        traceback.print_exc()
