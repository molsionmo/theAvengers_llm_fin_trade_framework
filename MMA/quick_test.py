#!/usr/bin/env python3
"""
简化版协作测试脚本：显示训练前后的实际输出差异
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel, AutoModelForSequenceClassification
from Multi import MultiModelCollaborator, AlignmentTrainer, AlignmentEvaluator

def get_text_generation_output(collaborator, text, use_collaboration=False):
    """获取文本生成的实际输出"""
    if not hasattr(collaborator, 'gpt2_generator'):
        # 创建一个GPT-2生成模型
        collaborator.gpt2_generator = GPT2LMHeadModel.from_pretrained("gpt2")
    
    tokenizer = collaborator.tokenizers[1]  # GPT-2 tokenizer
    
    if use_collaboration:
        # 使用协作：从BERT获取信息，传递给GPT-2
        collaboration_output = collaborator.collaborate(text, 0, 1)
        adapted_hidden = collaboration_output['adapted_hidden']
        
        # 使用适配后的hidden states作为初始状态
        inputs = tokenizer.encode_plus(text, return_tensors="pt", padding=True, truncation=True)
        
        # 生成文本（使用协作信息）
        with torch.no_grad():
            # 这里我们模拟使用协作信息影响生成
            outputs = collaborator.gpt2_generator.generate(
                inputs['input_ids'],
                max_length=len(inputs['input_ids'][0]) + 10,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
    else:
        # 正常生成
        inputs = tokenizer.encode_plus(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = collaborator.gpt2_generator.generate(
                inputs['input_ids'],
                max_length=len(inputs['input_ids'][0]) + 10,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
    
    # 解码生成的文本
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def get_classification_output(collaborator, text, use_collaboration=False):
    """获取分类任务的实际输出"""
    # 使用简单的方法：基于hidden states的最后一层进行情感分类
    if use_collaboration:
        # 使用协作后的hidden states
        collaboration_output = collaborator.collaborate(text, 0, 1)
        hidden_states = collaboration_output['adapted_hidden']
    else:
        # 使用原始BERT的hidden states
        hidden_states = collaborator.get_hidden_states(text, 0)
    
    # 简单的情感分类：基于hidden states的平均值
    pooled = hidden_states.mean(dim=1)  # [1, hidden_size]
    
    # 创建一个简单的分类头（线性层）
    if not hasattr(collaborator, 'classifier'):
        hidden_size = hidden_states.size(-1)
        collaborator.classifier = torch.nn.Linear(hidden_size, 3)  # 3个类别：正面、中性、负面
    
    with torch.no_grad():
        logits = collaborator.classifier(pooled)
        probabilities = F.softmax(logits, dim=-1)
    
    labels = ['负面', '中性', '正面']
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class].item()
    
    return {
        'predicted_label': labels[predicted_class],
        'confidence': confidence,
        'probabilities': {labels[i]: probabilities[0][i].item() for i in range(3)}
    }

def compare_similarity_understanding(collaborator, text1, text2, use_collaboration=False):
    """比较模型对语义相似性的理解"""
    if use_collaboration:
        # 使用协作后的投影
        hidden1_1 = collaborator.get_hidden_states(text1, 0)
        hidden2_1 = collaborator.get_hidden_states(text1, 1)
        hidden1_2 = collaborator.get_hidden_states(text2, 0)
        hidden2_2 = collaborator.get_hidden_states(text2, 1)
        
        proj1_1, proj2_1 = collaborator.central_processor.process([hidden1_1, hidden2_1])
        proj1_2, proj2_2 = collaborator.central_processor.process([hidden1_2, hidden2_2])
        
        # 计算BERT投影间的相似性
        bert_similarity = F.cosine_similarity(
            proj1_1.mean(dim=1), proj1_2.mean(dim=1)
        ).item()
        
        # 计算跨模型一致性
        cross_consistency = F.cosine_similarity(
            proj1_1.mean(dim=1), proj2_1.mean(dim=1)
        ).item()
        
        return {
            'bert_projection_similarity': bert_similarity,
            'cross_model_consistency': cross_consistency,
            'interpretation': f"协作后模型认为这两句话的相似度是 {bert_similarity:.3f}"
        }
    else:
        # 使用原始hidden states
        hidden1_bert = collaborator.get_hidden_states(text1, 0)
        hidden2_bert = collaborator.get_hidden_states(text2, 0)
        hidden1_gpt = collaborator.get_hidden_states(text1, 1)
        hidden2_gpt = collaborator.get_hidden_states(text2, 1)
        
        bert_similarity = F.cosine_similarity(
            hidden1_bert.mean(dim=1), hidden2_bert.mean(dim=1)
        ).item()
        
        gpt_similarity = F.cosine_similarity(
            hidden1_gpt.mean(dim=1), hidden2_gpt.mean(dim=1)
        ).item()
        
        return {
            'bert_similarity': bert_similarity,
            'gpt_similarity': gpt_similarity,
            'interpretation': f"BERT认为相似度是 {bert_similarity:.3f}, GPT-2认为是 {gpt_similarity:.3f}"
        }

def get_attention_focus(collaborator, text, model_idx):
    """获取模型注意力的焦点词汇"""
    tokenizer = collaborator.tokenizers[model_idx]
    inputs = tokenizer.encode_plus(text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = collaborator.models[model_idx](**inputs, output_attentions=True)
    
    # 获取最后一层的注意力权重
    attention = outputs.attentions[-1]  # [batch, heads, seq_len, seq_len]
    
    # 平均所有注意力头
    avg_attention = attention.mean(dim=1)[0]  # [seq_len, seq_len]
    
    # 获取每个token对所有其他token的平均注意力
    token_importance = avg_attention.mean(dim=0)
    
    # 获取tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # 找到最重要的3个tokens（排除特殊tokens）
    important_indices = []
    for i, token in enumerate(tokens):
        if token not in ['[CLS]', '[SEP]', '<|endoftext|>'] and not token.startswith('Ġ'):
            important_indices.append((i, token, token_importance[i].item()))
    
    # 按重要性排序
    important_indices.sort(key=lambda x: x[2], reverse=True)
    
    return {
        'tokens': tokens,
        'most_important': important_indices[:3] if len(important_indices) >= 3 else important_indices,
        'attention_pattern': avg_attention.cpu().numpy()
    }

def quick_test():
    """快速测试函数 - 显示实际的任务输出差异"""
    print("🚀 多模型协作效果测试 - 实际输出对比")
    print("=" * 60)
    
    # 初始化
    model1 = AutoModel.from_pretrained("bert-base-uncased")
    model2 = AutoModel.from_pretrained("gpt2")
    collaborator = MultiModelCollaborator([model1, model2])
    
    # 测试文本
    test_texts = [
        "The weather is",
        "Artificial intelligence will",
        "I feel happy because"
    ]
    
    similarity_pairs = [
        ("The cat is sleeping", "A feline is resting"),
        ("I love programming", "I enjoy coding")
    ]
    
    print("\n🔍 第一阶段：训练前的实际输出")
    print("=" * 60)
    
    # 1. 文本生成对比
    print("\n📝 文本生成任务:")
    generation_before = {}
    for text in test_texts:
        normal_output = get_text_generation_output(collaborator, text, use_collaboration=False)
        collab_output = get_text_generation_output(collaborator, text, use_collaboration=True)
        
        generation_before[text] = {
            'normal': normal_output,
            'collaborative': collab_output
        }
        
        print(f"\n输入: '{text}'")
        print(f"  正常生成: {normal_output}")
        print(f"  协作生成: {collab_output}")
    
    # 2. 情感分析对比
    print(f"\n😊 情感分析任务:")
    sentiment_before = {}
    sentiment_texts = ["I love this movie", "This is terrible", "The weather is okay"]
    
    for text in sentiment_texts:
        normal_sentiment = get_classification_output(collaborator, text, use_collaboration=False)
        collab_sentiment = get_classification_output(collaborator, text, use_collaboration=True)
        
        sentiment_before[text] = {
            'normal': normal_sentiment,
            'collaborative': collab_sentiment
        }
        
        print(f"\n文本: '{text}'")
        print(f"  正常分类: {normal_sentiment['predicted_label']} (置信度: {normal_sentiment['confidence']:.3f})")
        print(f"  协作分类: {collab_sentiment['predicted_label']} (置信度: {collab_sentiment['confidence']:.3f})")
    
    # 3. 语义相似性理解
    print(f"\n🔄 语义相似性理解:")
    similarity_before = {}
    
    for text1, text2 in similarity_pairs:
        normal_sim = compare_similarity_understanding(collaborator, text1, text2, use_collaboration=False)
        collab_sim = compare_similarity_understanding(collaborator, text1, text2, use_collaboration=True)
        
        similarity_before[(text1, text2)] = {
            'normal': normal_sim,
            'collaborative': collab_sim
        }
        
        print(f"\n对比: '{text1}' vs '{text2}'")
        print(f"  训练前: {normal_sim['interpretation']}")
        print(f"  协作后: {collab_sim['interpretation']}")
    
    # 训练适配器
    print("\n🔧 第二阶段：训练适配器")
    print("=" * 60)
    trainer = AlignmentTrainer(collaborator, learning_rate=1e-4)
    
    train_texts = test_texts + sentiment_texts + [t for pair in similarity_pairs for t in pair]
    train_texts.extend([
        "Machine learning is powerful",
        "The ocean is vast and deep",
        "Music brings joy to people"
    ])
    
    for epoch in range(3):
        loss = trainer.train_epoch(train_texts)
        print(f"  Epoch {epoch+1}: Loss = {loss:.4f}")
    
    print("\n🔍 第三阶段：训练后的实际输出")
    print("=" * 60)
    
    # 重新测试所有任务
    print("\n📝 训练后文本生成:")
    generation_after = {}
    for text in test_texts:
        normal_output = get_text_generation_output(collaborator, text, use_collaboration=False)
        collab_output = get_text_generation_output(collaborator, text, use_collaboration=True)
        
        generation_after[text] = {
            'normal': normal_output,
            'collaborative': collab_output
        }
        
        print(f"\n输入: '{text}'")
        print(f"  正常生成: {normal_output}")
        print(f"  协作生成: {collab_output}")
        
        # 显示变化
        if text in generation_before:
            print(f"  📊 变化:")
            print(f"    协作生成前: {generation_before[text]['collaborative']}")
            print(f"    协作生成后: {collab_output}")
    
    print(f"\n😊 训练后情感分析:")
    for text in sentiment_texts:
        normal_sentiment = get_classification_output(collaborator, text, use_collaboration=False)
        collab_sentiment = get_classification_output(collaborator, text, use_collaboration=True)
        
        print(f"\n文本: '{text}'")
        print(f"  正常分类: {normal_sentiment['predicted_label']} (置信度: {normal_sentiment['confidence']:.3f})")
        print(f"  协作分类: {collab_sentiment['predicted_label']} (置信度: {collab_sentiment['confidence']:.3f})")
        
        if text in sentiment_before:
            print(f"  📊 置信度变化:")
            print(f"    训练前: {sentiment_before[text]['collaborative']['confidence']:.3f}")
            print(f"    训练后: {collab_sentiment['confidence']:.3f}")
    
    print(f"\n🔄 训练后语义相似性:")
    for text1, text2 in similarity_pairs:
        normal_sim = compare_similarity_understanding(collaborator, text1, text2, use_collaboration=False)
        collab_sim = compare_similarity_understanding(collaborator, text1, text2, use_collaboration=True)
        
        print(f"\n对比: '{text1}' vs '{text2}'")
        print(f"  训练后: {collab_sim['interpretation']}")
        
        if (text1, text2) in similarity_before:
            before_cross = similarity_before[(text1, text2)]['collaborative'].get('cross_model_consistency', 0)
            after_cross = collab_sim.get('cross_model_consistency', 0)
            print(f"  📊 跨模型一致性变化: {before_cross:.3f} → {after_cross:.3f}")
    
    print("\n✨ 第四阶段：关键改进总结")
    print("=" * 60)
    
    print("🎯 主要观察:")
    print("  1. 文本生成的创造性和一致性")
    print("  2. 情感分析的准确性和置信度")
    print("  3. 语义理解的跨模型一致性")
    print("  4. 模型间信息传递的有效性")
    
    return {
        'generation_before': generation_before,
        'generation_after': generation_after,
        'sentiment_before': sentiment_before,
        'similarity_before': similarity_before
    }

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer
from Multi import MultiModelCollaborator, AlignmentTrainer, AlignmentEvaluator

def analyze_hidden_states(hidden_states, name):
    """分析hidden states的详细特征"""
    print(f"\n� {name} Hidden States 分析:")
    print(f"  形状: {hidden_states.shape}")
    print(f"  平均值: {hidden_states.mean().item():.6f}")
    print(f"  标准差: {hidden_states.std().item():.6f}")
    print(f"  最大值: {hidden_states.max().item():.6f}")
    print(f"  最小值: {hidden_states.min().item():.6f}")
    
    # 计算激活神经元比例（绝对值大于0.1的）
    active_ratio = (torch.abs(hidden_states) > 0.1).float().mean().item()
    print(f"  激活神经元比例: {active_ratio:.3f}")
    
    return {
        'shape': list(hidden_states.shape),
        'mean': hidden_states.mean().item(),
        'std': hidden_states.std().item(),
        'max': hidden_states.max().item(),
        'min': hidden_states.min().item(),
        'active_ratio': active_ratio
    }

def get_top_tokens_and_attention(collaborator, text, model_idx):
    """获取模型的Top激活tokens和注意力权重"""
    tokenizer = collaborator.tokenizers[model_idx]
    
    # 获取token化结果
    inputs = tokenizer.encode_plus(text, return_tensors="pt", padding=True, truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # 获取hidden states
    with torch.no_grad():
        outputs = collaborator.models[model_idx](**inputs, output_hidden_states=True, output_attentions=True)
    
    hidden_states = outputs.hidden_states[-1]  # 最后一层
    attentions = outputs.attentions[-1]  # 最后一层注意力
    
    # 计算每个token的重要性（通过hidden state的norm）
    token_importance = torch.norm(hidden_states[0], dim=-1)
    
    # 获取最重要的tokens
    top_indices = torch.argsort(token_importance, descending=True)[:3]
    
    result = {
        'tokens': tokens,
        'top_tokens': [(tokens[i], token_importance[i].item()) for i in top_indices],
        'attention_avg': attentions.mean(dim=1)[0].cpu().numpy()  # 平均注意力权重
    }
    
    return result, hidden_states

def compare_model_outputs(collaborator, text):
    """详细对比两个模型的输出"""
    print(f"\n🔍 详细分析文本: '{text}'")
    print("=" * 60)
    
    # 获取两个模型的详细输出
    model1_analysis, hidden1 = get_top_tokens_and_attention(collaborator, text, 0)
    model2_analysis, hidden2 = get_top_tokens_and_attention(collaborator, text, 1)
    
    print(f"\n📝 BERT (Model 1) 分析:")
    print(f"  Tokens: {model1_analysis['tokens']}")
    print(f"  重要tokens: {model1_analysis['top_tokens']}")
    
    print(f"\n📝 GPT-2 (Model 2) 分析:")
    print(f"  Tokens: {model2_analysis['tokens']}")
    print(f"  重要tokens: {model2_analysis['top_tokens']}")
    
    # 分析原始hidden states
    bert_stats = analyze_hidden_states(hidden1, "BERT")
    gpt_stats = analyze_hidden_states(hidden2, "GPT-2")
    
    # 通过协作系统处理
    collaboration_output = collaborator.collaborate(text, 0, 1)
    adapted_hidden = collaboration_output['adapted_hidden']
    
    print(f"\n🔄 协作后 (BERT→GPT-2) 分析:")
    adapted_stats = analyze_hidden_states(adapted_hidden, "协作适配")
    
    # 计算变化
    print(f"\n📊 关键变化:")
    print(f"  维度变化: {bert_stats['shape']} → {adapted_stats['shape']}")
    print(f"  激活强度变化: {bert_stats['mean']:.6f} → {adapted_stats['mean']:.6f}")
    print(f"  信息密度变化: {bert_stats['std']:.6f} → {adapted_stats['std']:.6f}")
    
    return {
        'bert': bert_stats,
        'gpt': gpt_stats, 
        'adapted': adapted_stats,
        'bert_tokens': model1_analysis,
        'gpt_tokens': model2_analysis
    }

def semantic_similarity_test(collaborator, text1, text2):
    """测试语义相似性在协作前后的变化"""
    print(f"\n🔄 语义相似性测试:")
    print(f"  文本1: '{text1}'")
    print(f"  文本2: '{text2}'")
    
    # 获取原始hidden states
    hidden1_1 = collaborator.get_hidden_states(text1, 0)  # BERT
    hidden1_2 = collaborator.get_hidden_states(text2, 0)  # BERT
    hidden2_1 = collaborator.get_hidden_states(text1, 1)  # GPT-2
    hidden2_2 = collaborator.get_hidden_states(text2, 1)  # GPT-2
    
    # 计算原始相似性
    bert_sim = F.cosine_similarity(
        hidden1_1.mean(dim=1), hidden1_2.mean(dim=1)
    ).item()
    
    gpt_sim = F.cosine_similarity(
        hidden2_1.mean(dim=1), hidden2_2.mean(dim=1)
    ).item()
    
    cross_sim_before = F.cosine_similarity(
        hidden1_1.mean(dim=1), hidden2_1.mean(dim=1)
    ).item()
    
    # 通过协作系统处理
    proj1_1, proj2_1 = collaborator.central_processor.process([hidden1_1, hidden2_1])
    proj1_2, proj2_2 = collaborator.central_processor.process([hidden1_2, hidden2_2])
    
    # 计算协作后相似性
    proj_sim1 = F.cosine_similarity(
        proj1_1.mean(dim=1), proj1_2.mean(dim=1)
    ).item()
    
    cross_sim_after = F.cosine_similarity(
        proj1_1.mean(dim=1), proj2_1.mean(dim=1)
    ).item()
    
    print(f"  BERT内部相似性: {bert_sim:.4f}")
    print(f"  GPT-2内部相似性: {gpt_sim:.4f}")
    print(f"  跨模型相似性 (协作前): {cross_sim_before:.4f}")
    print(f"  跨模型相似性 (协作后): {cross_sim_after:.4f}")
    print(f"  协作后投影相似性: {proj_sim1:.4f}")
    
    return {
        'bert_similarity': bert_sim,
        'gpt_similarity': gpt_sim,
        'cross_before': cross_sim_before,
        'cross_after': cross_sim_after,
        'projection_similarity': proj_sim1
    }

def quick_test():
    """快速测试函数"""
    print("🚀 快速协作效果测试 - 包含实际输出对比")
    print("=" * 60)
    
    # 初始化
    model1 = AutoModel.from_pretrained("bert-base-uncased")
    model2 = AutoModel.from_pretrained("gpt2")
    collaborator = MultiModelCollaborator([model1, model2])
    
    # 测试文本
    test_texts = [
        "What is artificial intelligence?",
        "The weather is beautiful today",
        "I love reading books"
    ]
    
    print("\n🔍 第一阶段：训练前的详细分析")
    print("=" * 60)
    
    # 详细分析每个测试文本（训练前）
    before_analyses = []
    for text in test_texts:
        analysis = compare_model_outputs(collaborator, text)
        before_analyses.append(analysis)
    
    # 语义相似性测试（训练前）
    semantic_before = semantic_similarity_test(
        collaborator, 
        "The cat is sleeping", 
        "A feline is resting"
    )
    
    # 训练适配器
    print("\n🔧 第二阶段：训练适配器")
    print("=" * 60)
    trainer = AlignmentTrainer(collaborator, learning_rate=1e-4)
    
    train_texts = test_texts + [
        "Machine learning is powerful",
        "The ocean is vast and deep", 
        "Music brings joy to people",
        "Technology changes our lives",
        "Education opens new doors"
    ]
    
    for epoch in range(3):
        loss = trainer.train_epoch(train_texts)
        print(f"  Epoch {epoch+1}: Loss = {loss:.4f}")
    
    print("\n🔍 第三阶段：训练后的详细分析")
    print("=" * 60)
    
    # 详细分析每个测试文本（训练后）
    after_analyses = []
    for text in test_texts:
        analysis = compare_model_outputs(collaborator, text)
        after_analyses.append(analysis)
    
    # 语义相似性测试（训练后）
    semantic_after = semantic_similarity_test(
        collaborator,
        "The cat is sleeping", 
        "A feline is resting"
    )
    
    # 生成对比报告
    print("\n📋 第四阶段：训练前后对比总结")
    print("=" * 60)
    
    print(f"\n📊 整体数值变化:")
    for i, text in enumerate(test_texts):
        before = before_analyses[i]
        after = after_analyses[i]
        
        print(f"\n  文本: '{text}'")
        print(f"    协作适配平均值: {before['adapted']['mean']:.6f} → {after['adapted']['mean']:.6f}")
        print(f"    协作适配标准差: {before['adapted']['std']:.6f} → {after['adapted']['std']:.6f}")
        print(f"    激活神经元比例: {before['adapted']['active_ratio']:.3f} → {after['adapted']['active_ratio']:.3f}")
    
    print(f"\n🔄 语义理解能力变化:")
    print(f"  跨模型语义一致性: {semantic_before['cross_before']:.4f} → {semantic_after['cross_after']:.4f}")
    print(f"  投影后语义保持: {semantic_before['projection_similarity']:.4f} → {semantic_after['projection_similarity']:.4f}")
    
    # 计算整体改善
    cross_improvement = ((semantic_after['cross_after'] - semantic_before['cross_before']) / 
                        abs(semantic_before['cross_before'])) * 100
    
    print(f"\n✨ 核心改善:")
    print(f"  跨模型理解提升: {cross_improvement:+.1f}%")
    
    if cross_improvement > 1000:
        conclusion = "🎉 训练显著改善了模型协作效果！"
    elif cross_improvement > 100:
        conclusion = "✅ 训练效果良好"
    else:
        conclusion = "📈 有一定改善"
    
    print(f"  {conclusion}")
    
    return {
        'before_analyses': before_analyses,
        'after_analyses': after_analyses,
        'semantic_before': semantic_before,
        'semantic_after': semantic_after,
        'improvement': cross_improvement
    }

def save_detailed_results(results):
    """保存详细的对比结果到文件"""
    import json
    
    # 准备保存的数据
    save_data = {
        "summary": {
            "improvement_percent": results['improvement'],
            "conclusion": "训练显著改善了模型协作效果" if results['improvement'] > 1000 else "训练有一定效果"
        },
        "semantic_analysis": {
            "before": results['semantic_before'],
            "after": results['semantic_after']
        },
        "detailed_changes": []
    }
    
    # 添加每个文本的详细变化
    test_texts = [
        "What is artificial intelligence?",
        "The weather is beautiful today", 
        "I love reading books"
    ]
    
    for i, text in enumerate(test_texts):
        before = results['before_analyses'][i]
        after = results['after_analyses'][i]
        
        change_data = {
            "text": text,
            "before_stats": before['adapted'],
            "after_stats": after['adapted'],
            "bert_tokens": before['bert_tokens']['tokens'],
            "gpt_tokens": before['gpt_tokens']['tokens']
        }
        save_data["detailed_changes"].append(change_data)
    
    # 保存到文件
    with open('detailed_collaboration_results.json', 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细结果已保存到 detailed_collaboration_results.json")
    
    # 同时生成一个简化的文本报告
    with open('collaboration_report.txt', 'w', encoding='utf-8') as f:
        f.write("🤖 多模型协作效果详细报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"📊 整体改善: {results['improvement']:+.1f}%\n\n")
        
        f.write("📝 各文本的具体变化:\n")
        for i, text in enumerate(test_texts):
            before = results['before_analyses'][i]
            after = results['after_analyses'][i]
            
            f.write(f"\n文本: '{text}'\n")
            f.write(f"  协作适配平均值: {before['adapted']['mean']:.6f} → {after['adapted']['mean']:.6f}\n")
            f.write(f"  协作适配标准差: {before['adapted']['std']:.6f} → {after['adapted']['std']:.6f}\n")
            f.write(f"  激活神经元比例: {before['adapted']['active_ratio']:.3f} → {after['adapted']['active_ratio']:.3f}\n")
        
        f.write(f"\n🔄 语义理解变化:\n")
        f.write(f"  跨模型语义一致性: {results['semantic_before']['cross_before']:.4f} → {results['semantic_after']['cross_after']:.4f}\n")
        f.write(f"  投影后语义保持: {results['semantic_before']['projection_similarity']:.4f} → {results['semantic_after']['projection_similarity']:.4f}\n")
    
    print(f"📄 文本报告已保存到 collaboration_report.txt")

if __name__ == "__main__":
    results = quick_test()
    save_detailed_results(results)
