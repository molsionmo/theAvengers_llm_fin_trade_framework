#!/usr/bin/env python3
"""
改进的金融情感分析评估脚本
解决基础模型重复生成问题，提供更公平的比较
"""

import argparse
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel
import re
from datetime import datetime
from tqdm import tqdm
import time

def load_model(model_name, adapter_path=None, max_seq_length=2048):
    """加载模型，处理适配器"""
    print(f"Loading model: {model_name}")
    
    if adapter_path:
        print(f"Loading with adapter: {adapter_path}")
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
            # 尝试加载适配器
            model = FastLanguageModel.from_pretrained(
                model_name=adapter_path,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )[0]
            print("Successfully loaded model with adapter")
        except Exception as e:
            print(f"Error loading adapter: {e}")
            print("Loading base model only...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
    
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def clean_response(text):
    """清理模型响应，去除重复字符"""
    # 去除连续的重复字符（如 !!!!! 等）
    text = re.sub(r'(.)\1{5,}', r'\1\1\1', text)
    
    # 去除奇怪的开头标记
    text = re.sub(r'^[#{}\[\]_\-!@#$%^&*()+=|\\`~]*', '', text)
    
    # 如果整个响应都是重复字符，标记为无效
    if len(set(text.replace(' ', '').replace('\n', ''))) <= 3:
        return "[INVALID_REPETITIVE_OUTPUT]"
    
    return text.strip()

def extract_sentiment_robust(response):
    """更鲁棒的情感提取，处理各种格式"""
    if "[INVALID_REPETITIVE_OUTPUT]" in response:
        return "invalid_output"
    
    # 移除响应中的清理标记
    response = response.replace("[INVALID_REPETITIVE_OUTPUT]", "")
    
    # 转换为小写进行匹配
    response_lower = response.lower()
    
    # 寻找明确的情感词汇
    if "positive" in response_lower:
        return "positive"
    elif "negative" in response_lower:
        return "negative"
    elif "neutral" in response_lower:
        return "neutral"
    
    # 如果没有找到明确的情感词，尝试从上下文推断
    positive_indicators = ["good", "great", "excellent", "up", "rise", "gain", "bull", "optimistic"]
    negative_indicators = ["bad", "poor", "terrible", "down", "fall", "loss", "bear", "pessimistic"]
    
    pos_count = sum(1 for word in positive_indicators if word in response_lower)
    neg_count = sum(1 for word in negative_indicators if word in response_lower)
    
    if pos_count > neg_count and pos_count > 0:
        return "positive"
    elif neg_count > pos_count and neg_count > 0:
        return "negative"
    else:
        return "neutral"  # 默认为neutral

def evaluate_model_improved(model, tokenizer, dataset, num_samples=100, batch_size=8):
    """改进的模型评估函数"""
    results = []
    correct = 0
    invalid_outputs = 0
    
    # 准备批次数据
    samples = dataset.select(range(min(num_samples, len(dataset))))
    
    print(f"Evaluating on {len(samples)} samples...")
    
    for i in tqdm(range(0, len(samples), batch_size)):
        batch_end = min(i + batch_size, len(samples))
        batch_samples = samples.select(range(i, batch_end))
        
        batch_prompts = []
        batch_true_answers = []
        
        for sample in batch_samples:
            prompt = f"Instruction: What is the sentiment of this tweet? Please choose an answer from {{negative/neutral/positive}}.\nInput: {sample['sentence']}\nOutput:"
            batch_prompts.append(prompt)
            batch_true_answers.append(sample['label'].lower())
        
        # 批量推理
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,  # 减少输出长度避免重复
                temperature=0.1,     # 降低随机性
                do_sample=True,
                repetition_penalty=1.2,  # 增加重复惩罚
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        # 处理每个输出
        for j, (prompt, true_answer) in enumerate(zip(batch_prompts, batch_true_answers)):
            generated_tokens = outputs[j][len(inputs['input_ids'][j]):]
            full_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # 清理响应
            cleaned_response = clean_response(full_response)
            
            # 提取情感
            generated_answer = extract_sentiment_robust(cleaned_response)
            
            # 检查是否为无效输出
            if generated_answer == "invalid_output":
                invalid_outputs += 1
                is_correct = False
            else:
                is_correct = generated_answer == true_answer
                if is_correct:
                    correct += 1
            
            results.append({
                "context": prompt,
                "true_answer": true_answer,
                "generated_answer": generated_answer,
                "full_response": cleaned_response,
                "correct": is_correct
            })
        
        # 添加延迟避免GPU过载
        time.sleep(0.1)
    
    accuracy = correct / len(results) if len(results) > 0 else 0
    valid_accuracy = correct / (len(results) - invalid_outputs) if (len(results) - invalid_outputs) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "valid_accuracy": valid_accuracy,
        "correct": correct,
        "total": len(results),
        "invalid_outputs": invalid_outputs,
        "results": results
    }

def main():
    parser = argparse.ArgumentParser(description="改进的金融情感分析评估")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct", help="基础模型名称")
    parser.add_argument("--adapter_path", help="适配器路径")
    parser.add_argument("--num_samples", type=int, default=100, help="评估样本数量")
    parser.add_argument("--compare", action="store_true", help="比较基础模型和训练模型")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    
    args = parser.parse_args()
    
    # 加载数据集
    print("Loading dataset...")
    dataset = load_dataset("Balaji173/finance_news_sentiment", split="train")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.compare and args.adapter_path:
        print("=" * 60)
        print("IMPROVED COMPARISON EVALUATION")
        print("=" * 60)
        
        # 评估基础模型
        print("\n1. Evaluating BASE model...")
        base_model, base_tokenizer = load_model(args.base_model)
        base_results = evaluate_model_improved(base_model, base_tokenizer, dataset, args.num_samples, args.batch_size)
        
        # 保存基础模型结果
        base_filename = f"improved_finance_base_{timestamp}.json"
        with open(base_filename, 'w', encoding='utf-8') as f:
            json.dump({
                "model_path": args.base_model,
                "adapter_path": None,
                "total_samples": args.num_samples,
                "timestamp": timestamp,
                **base_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Base model results saved to: {base_filename}")
        
        # 清理内存
        del base_model, base_tokenizer
        torch.cuda.empty_cache()
        
        # 评估训练模型
        print("\n2. Evaluating TRAINED model...")
        trained_model, trained_tokenizer = load_model(args.base_model, args.adapter_path)
        trained_results = evaluate_model_improved(trained_model, trained_tokenizer, dataset, args.num_samples, args.batch_size)
        
        # 保存训练模型结果
        model_name = args.adapter_path.split('/')[-1] if '/' in args.adapter_path else args.adapter_path
        trained_filename = f"improved_finance_{model_name}_{timestamp}.json"
        with open(trained_filename, 'w', encoding='utf-8') as f:
            json.dump({
                "model_path": args.base_model,
                "adapter_path": args.adapter_path,
                "total_samples": args.num_samples,
                "timestamp": timestamp,
                **trained_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Trained model results saved to: {trained_filename}")
        
        # 显示比较结果
        print("\n" + "=" * 60)
        print("IMPROVED COMPARISON RESULTS")
        print("=" * 60)
        print(f"Base Model:")
        print(f"  Total Accuracy:     {base_results['accuracy']:.4f} ({base_results['accuracy']*100:.2f}%)")
        print(f"  Valid Accuracy:     {base_results['valid_accuracy']:.4f} ({base_results['valid_accuracy']*100:.2f}%)")
        print(f"  Invalid Outputs:    {base_results['invalid_outputs']}/{base_results['total']}")
        
        print(f"\nTrained Model:")
        print(f"  Total Accuracy:     {trained_results['accuracy']:.4f} ({trained_results['accuracy']*100:.2f}%)")
        print(f"  Valid Accuracy:     {trained_results['valid_accuracy']:.4f} ({trained_results['valid_accuracy']*100:.2f}%)")
        print(f"  Invalid Outputs:    {trained_results['invalid_outputs']}/{trained_results['total']}")
        
        # 计算改进
        valid_improvement = trained_results['valid_accuracy'] - base_results['valid_accuracy']
        total_improvement = trained_results['accuracy'] - base_results['accuracy']
        
        print(f"\nImprovement (Valid Outputs):")
        print(f"  Absolute:           {valid_improvement:+.4f} ({valid_improvement*100:+.2f} percentage points)")
        if base_results['valid_accuracy'] > 0:
            relative_improvement = (valid_improvement / base_results['valid_accuracy']) * 100
            print(f"  Relative:           {relative_improvement:+.2f}%")
        
        print(f"\nImprovement (Total):")
        print(f"  Absolute:           {total_improvement:+.4f} ({total_improvement*100:+.2f} percentage points)")
        if base_results['accuracy'] > 0:
            relative_improvement_total = (total_improvement / base_results['accuracy']) * 100
            print(f"  Relative:           {relative_improvement_total:+.2f}%")
        
    else:
        # 单模型评估
        model, tokenizer = load_model(args.base_model, args.adapter_path)
        results = evaluate_model_improved(model, tokenizer, dataset, args.num_samples, args.batch_size)
        
        model_name = args.adapter_path.split('/')[-1] if args.adapter_path and '/' in args.adapter_path else "base"
        filename = f"improved_finance_{model_name}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "model_path": args.base_model,
                "adapter_path": args.adapter_path,
                "total_samples": args.num_samples,
                "timestamp": timestamp,
                **results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {filename}")
        print(f"Total Accuracy:     {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"Valid Accuracy:     {results['valid_accuracy']:.4f} ({results['valid_accuracy']*100:.2f}%)")
        print(f"Invalid Outputs:    {results['invalid_outputs']}/{results['total']}")

if __name__ == "__main__":
    main()
