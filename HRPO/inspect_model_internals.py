#!/usr/bin/env python3
"""
检查训练后模型的内部机制
Inspect the internal mechanisms of the trained model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel
import os
import json

def inspect_thinking_residual_mechanism():
    """检查thinking residual机制的具体实现"""
    
    print("=== 加载基础模型 ===")
    base_model, base_tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    print("=== 加载训练后的模型 ===")
    # 找到实验目录和最新checkpoint
    exp_dirs = [d for d in os.listdir("/workspace/HRPO/experiments/")]
    if exp_dirs:
        latest_exp = sorted(exp_dirs)[-1]
        exp_path = f"/workspace/HRPO/experiments/{latest_exp}"
        
        # 找到最新的checkpoint
        checkpoints = [d for d in os.listdir(exp_path) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1]
            model_path = f"{exp_path}/{latest_checkpoint}"
            print(f"使用模型路径: {model_path}")
            
            trained_model, trained_tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
        else:
            print("未找到checkpoint")
            return
    else:
        print("未找到训练后的模型")
        return
    
    print("\n=== 检查模型结构 ===")
    
    # 检查基础模型是否有thinking_residual
    base_has_thinking = hasattr(base_model.model.model, 'thinking_residual')
    trained_has_thinking = hasattr(trained_model.model.model, 'thinking_residual')
    
    print(f"基础模型有thinking_residual方法: {base_has_thinking}")
    print(f"训练模型有thinking_residual方法: {trained_has_thinking}")
    
    # 检查thinking_residual相关参数
    if trained_has_thinking:
        print("\n=== 训练模型的thinking_residual参数 ===")
        model = trained_model.model.model
        
        print(f"thinking_residual_gate_r: {model.thinking_residual_gate_r}")
        print(f"thinking_residual_gate_i: {model.thinking_residual_gate_i}")
        print(f"thinking_residual_Lambda: {model.thinking_residual_Lambda}")
        
        # 检查Lambda参数的值
        lambda_params = model.thinking_residual_Lambda.Lambda.data
        print(f"Lambda参数统计:")
        print(f"  平均值: {lambda_params.mean().item():.6f}")
        print(f"  标准差: {lambda_params.std().item():.6f}")
        print(f"  最小值: {lambda_params.min().item():.6f}")
        print(f"  最大值: {lambda_params.max().item():.6f}")
        
        # 检查门控参数
        gate_r_weight = model.thinking_residual_gate_r.weight.data
        gate_i_weight = model.thinking_residual_gate_i.weight.data
        
        print(f"\ngate_r权重统计:")
        print(f"  平均值: {gate_r_weight.mean().item():.6f}")
        print(f"  标准差: {gate_r_weight.std().item():.6f}")
        
        print(f"\ngate_i权重统计:")
        print(f"  平均值: {gate_i_weight.mean().item():.6f}")
        print(f"  标准差: {gate_i_weight.std().item():.6f}")
    
    print("\n=== 测试thinking_residual函数 ===")
    if trained_has_thinking:
        # 创建测试输入
        test_input = "Classify the sentiment of this news: The company's stock price surged after announcing record profits."
        
        # Tokenize
        inputs = trained_tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            # 获取嵌入
            embeds = trained_model.model.model.embed_tokens(inputs.input_ids)
            
            # 创建假的残差（通常来自前一层）
            residual = torch.randn_like(embeds) * 0.1
            
            # 调用thinking_residual
            output, a_t = trained_model.model.model.thinking_residual(embeds, residual)
            
            print(f"输入嵌入shape: {embeds.shape}")
            print(f"残差shape: {residual.shape}")
            print(f"输出shape: {output.shape}")
            print(f"a_t (混合系数) shape: {a_t.shape}")
            print(f"a_t平均值: {a_t.mean().item():.6f}")
            print(f"a_t标准差: {a_t.std().item():.6f}")
            print(f"a_t最小值: {a_t.min().item():.6f}")
            print(f"a_t最大值: {a_t.max().item():.6f}")
            
            # 检查输出是否真的是嵌入和残差的混合
            pure_embed_norm = torch.norm(embeds).item()
            pure_residual_norm = torch.norm(residual).item()
            output_norm = torch.norm(output).item()
            
            print(f"\n范数比较:")
            print(f"  原始嵌入范数: {pure_embed_norm:.6f}")
            print(f"  残差范数: {pure_residual_norm:.6f}")
            print(f"  混合输出范数: {output_norm:.6f}")
    
    print("\n=== 比较生成输出 ===")
    test_prompts = [
        "Classify the sentiment of this news: The company announced massive layoffs.",
        "Classify the sentiment of this news: The company reported record-breaking revenue growth.",
        "Classify the sentiment of this news: The company's CEO resigned unexpectedly."
    ]
    
    print("基础模型生成:")
    base_model = FastLanguageModel.for_inference(base_model)
    for i, prompt in enumerate(test_prompts):
        inputs = base_tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = base_model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True,
                pad_token_id=base_tokenizer.eos_token_id
            )
        response = base_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  {i+1}. {response.strip()}")
    
    print("\n训练模型生成:")
    trained_model = FastLanguageModel.for_inference(trained_model)
    for i, prompt in enumerate(test_prompts):
        inputs = trained_tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = trained_model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=True,
                pad_token_id=trained_tokenizer.eos_token_id
            )
        response = trained_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  {i+1}. {response.strip()}")

if __name__ == "__main__":
    inspect_thinking_residual_mechanism()
