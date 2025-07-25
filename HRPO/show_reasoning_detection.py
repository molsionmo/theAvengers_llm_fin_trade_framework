#!/usr/bin/env python3
"""
展示推理检测的具体例子
"""

import json

def show_reasoning_examples():
    print("🔍 推理检测的具体例子")
    print("=" * 50)
    
    # 读取两个结果文件
    with open("./eval_results_base_model_200samples.json", 'r') as f:
        base_results = json.load(f)
    
    with open("./experiments/Qwen2.5-1.5B-Instruct-gsm8k-group4-lora32-rmin0.98-temp0.5/checkpoint-2000/eval_results.json", 'r') as f:
        trained_results = json.load(f)
    
    # 推理检测函数
    def has_reasoning(response):
        reasoning_indicators = ['analyze', 'determine', 'based on', 'therefore', 'considering', 'context']
        return any(indicator in response.lower() for indicator in reasoning_indicators)
    
    def find_reasoning_words(response):
        reasoning_indicators = ['analyze', 'determine', 'based on', 'therefore', 'considering', 'context']
        found = [indicator for indicator in reasoning_indicators if indicator in response.lower()]
        return found
    
    print("📝 基础模型例子:")
    print("-" * 30)
    
    for i, result in enumerate(base_results['results'][:5]):
        reasoning_words = find_reasoning_words(result['full_response'])
        has_reason = has_reasoning(result['full_response'])
        
        print(f"\n样本 {i+1}:")
        print(f"输入: {result['context'][:60]}...")
        print(f"完整回复: {result['full_response'][:120]}...")
        print(f"包含推理关键词: {reasoning_words}")
        print(f"判定为有推理: {'✅' if has_reason else '❌'}")
    
    print("\n" + "=" * 50)
    print("📝 训练模型例子:")
    print("-" * 30)
    
    for i, result in enumerate(trained_results['results'][:5]):
        reasoning_words = find_reasoning_words(result['full_response'])
        has_reason = has_reasoning(result['full_response'])
        
        print(f"\n样本 {i+1}:")
        print(f"输入: {result['context'][:60]}...")
        print(f"完整回复: {result['full_response'][:120]}...")
        print(f"包含推理关键词: {reasoning_words}")
        print(f"判定为有推理: {'✅' if has_reason else '❌'}")
    
    # 统计分析
    base_with_reasoning = sum(1 for r in base_results['results'] if has_reasoning(r['full_response']))
    trained_with_reasoning = sum(1 for r in trained_results['results'] if has_reasoning(r['full_response']))
    
    print(f"\n📊 统计结果:")
    print(f"基础模型有推理的样本: {base_with_reasoning}/{len(base_results['results'])}")
    print(f"训练模型有推理的样本: {trained_with_reasoning}/{len(trained_results['results'])}")
    
    # 展示一个完整的对比
    print(f"\n🔍 完整对比例子 (同一个问题):")
    print(f"问题: {base_results['results'][0]['context'][:80]}...")
    
    print(f"\n🤖 基础模型完整回答:")
    print(f"{base_results['results'][0]['full_response']}")
    print(f"推理关键词: {find_reasoning_words(base_results['results'][0]['full_response'])}")
    
    print(f"\n🧠 训练模型完整回答:")
    print(f"{trained_results['results'][0]['full_response']}")
    print(f"推理关键词: {find_reasoning_words(trained_results['results'][0]['full_response'])}")

if __name__ == "__main__":
    show_reasoning_examples()
