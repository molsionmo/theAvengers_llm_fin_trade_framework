#!/usr/bin/env python3
"""
在不同的金融情绪数据集上评估模型泛化能力
Evaluate model generalization on different financial sentiment datasets
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

def test_on_different_datasets():
    """在多个不同的金融情绪数据集上测试"""
    
    # 可用的金融情绪数据集
    test_datasets = [
        # {
        #     'name': 'zeroshot/twitter-financial-news-sentiment',
        #     'split': 'validation',
        #     'text_field': 'text',
        #     'label_field': 'label',
        #     'label_mapping': {0: 'negative', 1: 'neutral', 2: 'positive'}
        # },
        {
            'name': 'financial_phrasebank',
            'config': 'sentences_allagree',
            'split': 'train',
            'text_field': 'sentence',
            'label_field': 'label',
            'label_mapping': {0: 'negative', 1: 'neutral', 2: 'positive'}
        },
        # {
        #     'name': 'Sahrmann/finance-sentiment-extraction',
        #     'split': 'train',
        #     'text_field': 'text',
        #     'label_field': 'sentiment',
        #     'label_mapping': {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'}
        # }
    ]
    
    base_model = "Qwen/Qwen2.5-1.5B-Instruct"
    checkpoint_path = "./experiments/Qwen2.5-1.5B-Instruct-gsm8k-group4-lora32-rmin0.98-temp0.5/checkpoint-2385"
    
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
            if len(dataset) > 200:
                dataset = dataset.shuffle(seed=42).select(range(200))
            
            print(f"数据集大小: {len(dataset)}")
            
            # 测试基础模型
            print(f"\n--- 基础模型测试 ---")
            base_metrics = evaluate_on_dataset(
                model_path=base_model,
                adapter_path=None,
                dataset=dataset,
                dataset_info=dataset_info,
                use_base_model=True
            )
            
            # 测试训练后模型
            print(f"\n--- 训练模型测试 ---")
            trained_metrics = evaluate_on_dataset(
                model_path=base_model,
                adapter_path=checkpoint_path,
                dataset=dataset,
                dataset_info=dataset_info,
                use_base_model=False
            )
            
            # 对比结果
            print(f"\n--- 结果对比 ---")
            print(f"数据集: {dataset_info['name']}")
            print(f"基础模型准确率: {base_metrics['accuracy']:.3f} ({base_metrics['correct']}/{base_metrics['total']})")
            print(f"训练模型准确率: {trained_metrics['accuracy']:.3f} ({trained_metrics['correct']}/{trained_metrics['total']})")
            improvement = trained_metrics['accuracy'] - base_metrics['accuracy']
            print(f"提升: {improvement:.3f} ({improvement*100:.1f}%)")
            
            # 保存结果
            results = {
                'dataset': dataset_info['name'],
                'base_model_metrics': base_metrics,
                'trained_model_metrics': trained_metrics,
                'improvement': improvement
            }
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"generalization_test_{dataset_info['name'].replace('/', '_')}_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"结果保存到: {filename}")
            
        except Exception as e:
            print(f"处理数据集 {dataset_info['name']} 时出错: {e}")
            continue

def evaluate_on_dataset(model_path, adapter_path, dataset, dataset_info, use_base_model=False):
    """在指定数据集上评估模型"""
    
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
        
        # 准备prompts
        prompts = []
        for text in texts:
            prompt = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f"What is the sentiment of this financial text? Please choose an answer from {{negative/neutral/positive}}.\nInput: {text.strip()}"},
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
            add_special_tokens=False
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
                    max_new_tokens=100,
                ),
                processing_class=tokenizer,
                is_inference=True,
            )
        
        # 处理结果
        for j, output in enumerate(outputs):
            response = tokenizer.decode(output[prompt_length:])
            response = response.split(tokenizer.special_tokens_map['eos_token'])[0]
            
            # 提取答案
            extracted = extract_from_response(response)
            generated_answer = process_finance_sentiment_answer(extracted)
            
            # 获取真实标签
            true_label = labels[j]
            if isinstance(true_label, int):
                true_answer = dataset_info['label_mapping'][true_label]
            else:
                true_answer = str(true_label).lower()
            
            is_correct = generated_answer == true_answer
            
            # 记录结果
            result = {
                'text': texts[j],
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

if __name__ == "__main__":
    test_on_different_datasets()
