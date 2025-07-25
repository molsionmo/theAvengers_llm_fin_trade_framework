#!/usr/bin/env python3
"""
在更多不同的金融/情绪数据集上测试泛化能力
Test generalization on more diverse financial/sentiment datasets
"""

import unsloth
from unsloth import FastLanguageModel
import os
import json
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import GenerationConfig
from tqdm import tqdm
from utils import *

def test_more_datasets():
    """测试更多数据集"""
    
    # 更多测试数据集
    test_datasets = [
        # 通用情绪分析数据集
        {
            'name': 'stanfordnlp/imdb',
            'split': 'test',
            'text_field': 'text',
            'label_field': 'label',
            'label_mapping': {0: 'negative', 1: 'positive'},
            'max_samples': 100  # IMDB数据集很大，只取100个样本
        },
        # Twitter情绪
        {
            'name': 'tweet_eval', 
            'config': 'sentiment',
            'split': 'test',
            'text_field': 'text',
            'label_field': 'label',
            'label_mapping': {0: 'negative', 1: 'neutral', 2: 'positive'},
            'max_samples': 200
        },
        # 新闻情绪
        {
            'name': 'SetFit/20_newsgroups_sentiment',
            'split': 'test',
            'text_field': 'text', 
            'label_field': 'label',
            'label_mapping': {0: 'negative', 1: 'positive'},
            'max_samples': 200
        }
    ]
    
    base_model = "Qwen/Qwen2.5-1.5B-Instruct"
    checkpoint_path = "./experiments/Qwen2.5-1.5B-Instruct-gsm8k-group4-lora32-rmin0.98-temp0.5/checkpoint-2385"
    
    all_results = []
    
    for dataset_info in test_datasets:
        print(f"\n{'='*60}")
        print(f"测试数据集: {dataset_info['name']}")
        print(f"{'='*60}")
        
        try:
            # 加载数据集
            if 'config' in dataset_info:
                dataset = load_dataset(dataset_info['name'], dataset_info['config'])[dataset_info['split']]
            else:
                dataset = load_dataset(dataset_info['name'])[dataset_info['split']]
            
            # 限制样本数量
            max_samples = dataset_info.get('max_samples', 200)
            if len(dataset) > max_samples:
                dataset = dataset.shuffle(seed=42).select(range(max_samples))
            
            print(f"数据集大小: {len(dataset)}")
            
            # 测试基础模型
            print(f"\n--- 基础模型测试 ---")
            base_metrics = evaluate_on_general_dataset(
                model_path=base_model,
                adapter_path=None,
                dataset=dataset,
                dataset_info=dataset_info,
                use_base_model=True
            )
            
            # 测试训练后模型
            print(f"\n--- 训练模型测试 ---")
            trained_metrics = evaluate_on_general_dataset(
                model_path=base_model,
                adapter_path=checkpoint_path,
                dataset=dataset,
                dataset_info=dataset_info,
                use_base_model=False
            )
            
            # 对比结果
            improvement = trained_metrics['accuracy'] - base_metrics['accuracy']
            print(f"\n--- 结果对比 ---")
            print(f"数据集: {dataset_info['name']}")
            print(f"基础模型准确率: {base_metrics['accuracy']:.3f} ({base_metrics['correct']}/{base_metrics['total']})")
            print(f"训练模型准确率: {trained_metrics['accuracy']:.3f} ({trained_metrics['correct']}/{trained_metrics['total']})")
            print(f"提升: {improvement:.3f} ({improvement*100:.1f}%)")
            
            # 记录结果
            result = {
                'dataset': dataset_info['name'],
                'base_accuracy': base_metrics['accuracy'],
                'trained_accuracy': trained_metrics['accuracy'],
                'improvement': improvement,
                'base_correct': base_metrics['correct'],
                'trained_correct': trained_metrics['correct'],
                'total_samples': base_metrics['total']
            }
            all_results.append(result)
            
        except Exception as e:
            print(f"处理数据集 {dataset_info['name']} 时出错: {e}")
            continue
    
    # 汇总所有结果
    print(f"\n{'='*60}")
    print("汇总结果")
    print(f"{'='*60}")
    
    total_base_correct = sum(r['base_correct'] for r in all_results)
    total_trained_correct = sum(r['trained_correct'] for r in all_results)
    total_samples = sum(r['total_samples'] for r in all_results)
    
    overall_base_acc = total_base_correct / total_samples
    overall_trained_acc = total_trained_correct / total_samples
    overall_improvement = overall_trained_acc - overall_base_acc
    
    print(f"总体基础模型准确率: {overall_base_acc:.3f} ({total_base_correct}/{total_samples})")
    print(f"总体训练模型准确率: {overall_trained_acc:.3f} ({total_trained_correct}/{total_samples})")
    print(f"总体提升: {overall_improvement:.3f} ({overall_improvement*100:.1f}%)")
    
    print(f"\n各数据集详细结果:")
    for result in all_results:
        print(f"  {result['dataset']}: {result['base_accuracy']:.3f} → {result['trained_accuracy']:.3f} (+{result['improvement']:.3f})")
    
    # 保存汇总结果
    summary = {
        'individual_results': all_results,
        'overall_summary': {
            'base_accuracy': overall_base_acc,
            'trained_accuracy': overall_trained_acc,
            'improvement': overall_improvement,
            'total_base_correct': total_base_correct,
            'total_trained_correct': total_trained_correct,
            'total_samples': total_samples
        },
        'timestamp': datetime.now().isoformat()
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"comprehensive_generalization_test_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n汇总结果保存到: {filename}")

def evaluate_on_general_dataset(model_path, adapter_path, dataset, dataset_info, use_base_model=False):
    """在通用情绪数据集上评估模型"""
    
    # 加载模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=1024,
        load_in_4bit=False,
        fast_inference=False,
    )
    model.answer_start = ANSWER_START
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    if not use_base_model and adapter_path:
        model.load_adapter(adapter_path)
    model = FastLanguageModel.for_inference(model)
    
    results = []
    correct = 0
    total = 0
    
    batch_size = 4
    total_samples = len(dataset)
    
    progress_bar = tqdm(
        total=total_samples,
        desc=f"评估 {'基础' if use_base_model else '训练'}模型",
        unit="examples",
        dynamic_ncols=True,
    )
    
    # 批处理
    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        batch = dataset[i:end_idx]
        
        # 处理批次数据
        if isinstance(batch[dataset_info['text_field']], list):
            texts = batch[dataset_info['text_field']]
            labels = batch[dataset_info['label_field']]
        else:
            texts = [batch[dataset_info['text_field']]]
            labels = [batch[dataset_info['label_field']]]
        
        # 准备prompts - 适配不同类型的数据
        prompts = []
        for text in texts:
            # 截断过长的文本
            if len(text) > 500:
                text = text[:500] + "..."
                
            if 'finance' in dataset_info['name'].lower() or 'financial' in dataset_info['name'].lower():
                task_desc = "What is the sentiment of this financial text?"
            elif 'twitter' in dataset_info['name'].lower() or 'tweet' in dataset_info['name'].lower():
                task_desc = "What is the sentiment of this tweet?"
            elif 'imdb' in dataset_info['name'].lower():
                task_desc = "What is the sentiment of this movie review?"
            else:
                task_desc = "What is the sentiment of this text?"
            
            # 根据标签映射确定选项
            options = list(set(dataset_info['label_mapping'].values()))
            options_str = "{" + "/".join(sorted(options)) + "}"
            
            prompt = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f"{task_desc} Please choose an answer from {options_str}.\nInput: {text.strip()}"},
            ]
            prompts.append(prompt)
        
        # 格式化prompts
        formatted_prompts = [
            tokenizer.apply_chat_template(
                p,
                tokenize=False,
                add_generation_prompt=True
            )
            for p in prompts
        ]
        
        # Tokenize
        prompt_inputs = tokenizer(
            formatted_prompts, 
            return_tensors="pt", 
            padding=True, 
            padding_side="left", 
            add_special_tokens=False,
            truncation=True,
            max_length=800  # 限制长度防止内存问题
        )
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        prompt_ids = prompt_ids.to(model.device)
        prompt_mask = prompt_mask.to(model.device)
        prompt_length = prompt_ids.size(1)
        
        # 生成回复
        with torch.no_grad():
            outputs = model.generate(
                prompt_ids, 
                attention_mask=prompt_mask, 
                generation_config=GenerationConfig(
                    do_sample=True,
                    temperature=0.1,
                    max_new_tokens=50,  # 减少生成长度
                ),
                processing_class=tokenizer,
                is_inference=True,
            )
        
        # 处理结果
        for j, output in enumerate(outputs):
            response = tokenizer.decode(output[prompt_length:])
            response = response.split(tokenizer.special_tokens_map['eos_token'])[0]
            
            # 提取答案 - 更通用的答案提取
            extracted = extract_from_response(response)
            generated_answer = extract_sentiment_answer(extracted, list(dataset_info['label_mapping'].values()))
            
            # 获取真实标签
            true_label = labels[j]
            if isinstance(true_label, int):
                true_answer = dataset_info['label_mapping'][true_label]
            else:
                true_answer = str(true_label).lower()
            
            is_correct = generated_answer == true_answer
            
            # 记录结果
            result = {
                'text': texts[j][:200] + "..." if len(texts[j]) > 200 else texts[j],  # 截断存储的文本
                'true_answer': true_answer,
                'generated_answer': generated_answer,
                'full_response': response,
                'correct': is_correct
            }
            results.append(result)
            
            if is_correct:
                correct += 1
            total += 1
        
        progress_bar.update(len(texts))
        progress_bar.set_postfix({
            'acc': f'{(correct/total)*100:.1f}%',
            'correct': f'{correct}/{total}',
        })
    
    progress_bar.close()
    
    # 计算最终指标
    accuracy = correct / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results
    }

def extract_sentiment_answer(text, valid_options):
    """更通用的情绪答案提取函数"""
    text = text.lower().strip()
    
    # 首先尝试直接匹配
    for option in valid_options:
        if option.lower() in text:
            return option.lower()
    
    # 如果没有直接匹配，使用关键词
    if 'positive' in valid_options and any(word in text for word in ['positive', 'good', 'great', 'excellent', 'amazing']):
        return 'positive'
    if 'negative' in valid_options and any(word in text for word in ['negative', 'bad', 'terrible', 'awful', 'horrible']):
        return 'negative'
    if 'neutral' in valid_options and any(word in text for word in ['neutral', 'okay', 'average', 'normal']):
        return 'neutral'
    
    # 默认返回第一个选项
    return valid_options[0].lower() if valid_options else 'unknown'

if __name__ == "__main__":
    test_more_datasets()
