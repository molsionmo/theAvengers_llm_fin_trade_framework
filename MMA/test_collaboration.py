#!/usr/bin/env python3
"""
测试脚本：比较训练前后的模型协作效果
评估在真实任务中，适配器训练是否能改善跨模型协作的性能
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
import json
import time
from Multi import MultiModelCollaborator, AlignmentTrainer, AlignmentEvaluator

class CollaborationTester:
    """协作效果测试器"""
    
    def __init__(self, model1_name="bert-base-uncased", model2_name="gpt2"):
        print(f"初始化测试器: {model1_name} + {model2_name}")
        self.model1 = AutoModel.from_pretrained(model1_name)
        self.model2 = AutoModel.from_pretrained(model2_name)
        self.collaborator = MultiModelCollaborator([self.model1, self.model2])
        
        # 准备测试数据集
        self.test_scenarios = self._prepare_test_scenarios()
        
    def _prepare_test_scenarios(self):
        """准备多种测试场景"""
        return {
            "question_answering": [
                "What is the capital of France?",
                "Who wrote Romeo and Juliet?", 
                "What is the largest planet in our solar system?",
                "When did World War II end?",
                "What is the chemical symbol for gold?"
            ],
            "sentiment_analysis": [
                "I love this movie, it's amazing!",
                "This product is terrible and doesn't work.",
                "The weather today is quite pleasant.",
                "I'm feeling a bit sad about the news.",
                "This restaurant serves excellent food."
            ],
            "text_completion": [
                "The future of artificial intelligence",
                "Climate change is a serious issue",
                "Space exploration has always been",
                "Education in the digital age",
                "The importance of renewable energy"
            ],
            "semantic_similarity": [
                ("The cat is sleeping", "A feline is resting"),
                ("I love programming", "I enjoy coding"),
                ("The weather is cold", "It's freezing outside"),
                ("She is reading a book", "The woman is studying literature"),
                ("The car is fast", "The vehicle has high speed")
            ]
        }
    
    def test_hidden_state_quality(self, scenario_name, texts):
        """测试hidden state的质量"""
        print(f"\n测试场景: {scenario_name}")
        results = {
            'untrained': {'cosine_similarities': [], 'mmd_losses': []},
            'trained': {'cosine_similarities': [], 'mmd_losses': []}
        }
        
        evaluator = AlignmentEvaluator(metrics=['cosine', 'mmd'])
        
        for text in texts:
            if isinstance(text, tuple):
                text = text[0]  # 对于语义相似性测试，只取第一个句子
                
            # 获取原始hidden states
            hidden1 = self.collaborator.get_hidden_states(text, 0)
            hidden2 = self.collaborator.get_hidden_states(text, 1)
            
            # 未训练的对齐效果
            untrained_cosine = evaluator.cosine_similarity(hidden1, hidden2)
            untrained_mmd = evaluator.mmd_loss(
                hidden1.view(-1, hidden1.size(-1)), 
                hidden2.view(-1, hidden2.size(-1))
            ).item()
            
            results['untrained']['cosine_similarities'].append(untrained_cosine)
            results['untrained']['mmd_losses'].append(untrained_mmd)
            
            # 通过中心处理层处理后的对齐效果
            projected_states = self.collaborator.central_processor.process([hidden1, hidden2])
            proj1, proj2 = projected_states
            
            trained_cosine = evaluator.cosine_similarity(proj1, proj2)
            trained_mmd = evaluator.mmd_loss(
                proj1.view(-1, proj1.size(-1)), 
                proj2.view(-1, proj2.size(-1))
            ).item()
            
            results['trained']['cosine_similarities'].append(trained_cosine)
            results['trained']['mmd_losses'].append(trained_mmd)
        
        return results
    
    def test_semantic_similarity_preservation(self):
        """测试语义相似性保持能力"""
        print("\n=== 语义相似性保持测试 ===")
        
        similarity_pairs = self.test_scenarios["semantic_similarity"]
        results = {'untrained': [], 'trained': []}
        
        for text1, text2 in similarity_pairs:
            print(f"测试对: '{text1}' vs '{text2}'")
            
            # 获取原始hidden states
            hidden1_1 = self.collaborator.get_hidden_states(text1, 0)
            hidden1_2 = self.collaborator.get_hidden_states(text2, 0)
            hidden2_1 = self.collaborator.get_hidden_states(text1, 1)
            hidden2_2 = self.collaborator.get_hidden_states(text2, 1)
            
            # 未训练情况：直接计算相似性
            # 使用平均池化获得句子级表示
            sent1_bert = hidden1_1.mean(dim=1)
            sent2_bert = hidden1_2.mean(dim=1)
            sent1_gpt = hidden2_1.mean(dim=1)
            sent2_gpt = hidden2_2.mean(dim=1)
            
            untrained_bert_sim = F.cosine_similarity(sent1_bert, sent2_bert).item()
            untrained_gpt_sim = F.cosine_similarity(sent1_gpt, sent2_gpt).item()
            untrained_cross_sim = F.cosine_similarity(sent1_bert, sent1_gpt).item()
            
            # 训练后情况：通过中心处理层
            proj1_1, proj2_1 = self.collaborator.central_processor.process([hidden1_1, hidden2_1])
            proj1_2, proj2_2 = self.collaborator.central_processor.process([hidden1_2, hidden2_2])
            
            sent1_proj1 = proj1_1.mean(dim=1)
            sent2_proj1 = proj1_2.mean(dim=1)
            sent1_proj2 = proj2_1.mean(dim=1)
            sent2_proj2 = proj2_2.mean(dim=1)
            
            trained_proj1_sim = F.cosine_similarity(sent1_proj1, sent2_proj1).item()
            trained_proj2_sim = F.cosine_similarity(sent1_proj2, sent2_proj2).item()
            trained_cross_sim = F.cosine_similarity(sent1_proj1, sent1_proj2).item()
            
            results['untrained'].append({
                'bert_similarity': untrained_bert_sim,
                'gpt_similarity': untrained_gpt_sim,
                'cross_model_similarity': untrained_cross_sim
            })
            
            results['trained'].append({
                'proj1_similarity': trained_proj1_sim,
                'proj2_similarity': trained_proj2_sim,
                'cross_model_similarity': trained_cross_sim
            })
            
            print(f"  未训练 - BERT相似性: {untrained_bert_sim:.3f}, GPT相似性: {untrained_gpt_sim:.3f}, 跨模型: {untrained_cross_sim:.3f}")
            print(f"  训练后 - 投影1相似性: {trained_proj1_sim:.3f}, 投影2相似性: {trained_proj2_sim:.3f}, 跨模型: {trained_cross_sim:.3f}")
        
        return results
    
    def test_information_transfer_quality(self):
        """测试信息传递质量"""
        print("\n=== 信息传递质量测试 ===")
        
        test_texts = self.test_scenarios["question_answering"]
        results = {'untrained': [], 'trained': []}
        
        for text in test_texts:
            print(f"测试文本: '{text}'")
            
            # 获取协作结果
            collaboration_output = self.collaborator.collaborate(text, 0, 1)
            adapted_hidden = collaboration_output['adapted_hidden']
            normal_hidden = collaboration_output['normal_hidden']
            
            # 计算信息保持度（通过比较维度和激活模式）
            # 1. 激活强度比较
            adapted_activation_strength = torch.norm(adapted_hidden, dim=-1).mean().item()
            normal_activation_strength = torch.norm(normal_hidden, dim=-1).mean().item()
            
            # 2. 信息密度比较（通过方差衡量）
            adapted_variance = torch.var(adapted_hidden, dim=-1).mean().item()
            normal_variance = torch.var(normal_hidden, dim=-1).mean().item()
            
            # 3. 跨模型一致性
            # 由于adapted_hidden和normal_hidden维度不同，需要先统一维度
            adapted_mean = adapted_hidden.mean(dim=1)  # [1, 512]
            normal_mean = normal_hidden.mean(dim=1)    # [1, 768]
            
            # 将两者投影到相同维度进行比较
            min_dim = min(adapted_mean.size(-1), normal_mean.size(-1))
            adapted_reduced = adapted_mean[:, :min_dim]
            normal_reduced = normal_mean[:, :min_dim]
            
            cross_model_consistency = F.cosine_similarity(
                adapted_reduced, 
                normal_reduced
            ).item()
            
            result = {
                'activation_strength_ratio': adapted_activation_strength / normal_activation_strength,
                'variance_ratio': adapted_variance / normal_variance,
                'cross_model_consistency': cross_model_consistency
            }
            
            results['trained'].append(result)
            
            print(f"  激活强度比: {result['activation_strength_ratio']:.3f}")
            print(f"  方差比: {result['variance_ratio']:.3f}")
            print(f"  跨模型一致性: {result['cross_model_consistency']:.3f}")
        
        return results
    
    def train_adapter(self, epochs=5):
        """训练适配器"""
        print(f"\n=== 开始训练适配器 ({epochs} epochs) ===")
        
        trainer = AlignmentTrainer(self.collaborator, learning_rate=1e-4)
        
        # 准备训练数据
        train_texts = []
        for scenario_texts in self.test_scenarios.values():
            if isinstance(scenario_texts[0], tuple):
                # 对于语义相似性数据，提取所有文本
                for pair in scenario_texts:
                    train_texts.extend(pair)
            else:
                train_texts.extend(scenario_texts)
        
        # 添加更多训练数据
        additional_texts = [
            "Machine learning is transforming technology",
            "The ocean is vast and mysterious", 
            "Books contain endless knowledge",
            "Music brings people together",
            "Travel broadens one's perspective",
            "Friendship is valuable beyond measure",
            "Science helps us understand the world",
            "Art expresses human creativity"
        ]
        train_texts.extend(additional_texts)
        
        print(f"训练数据量: {len(train_texts)} 条文本")
        
        # 评估训练前效果
        val_texts = train_texts[:5]
        pre_train_results = trainer.evaluate_alignment(val_texts)
        print(f"训练前对齐效果 - 余弦相似度: {pre_train_results['cosine']:.4f}, MMD: {pre_train_results['mmd']:.4f}")
        
        # 开始训练
        start_time = time.time()
        trainer.train(train_texts, epochs=epochs, validation_dataset=val_texts)
        training_time = time.time() - start_time
        
        # 评估训练后效果
        post_train_results = trainer.evaluate_alignment(val_texts)
        print(f"训练后对齐效果 - 余弦相似度: {post_train_results['cosine']:.4f}, MMD: {post_train_results['mmd']:.4f}")
        print(f"训练耗时: {training_time:.2f} 秒")
        
        return pre_train_results, post_train_results
    
    def run_comprehensive_test(self, train_adapter=True):
        """运行综合测试"""
        print("=" * 60)
        print("🚀 开始多模型协作效果综合测试")
        print("=" * 60)
        
        # 第一阶段：测试未训练的协作效果
        print("\n📊 阶段1: 测试未训练的基线效果")
        baseline_results = {}
        
        for scenario_name, texts in self.test_scenarios.items():
            if scenario_name != "semantic_similarity":
                results = self.test_hidden_state_quality(scenario_name, texts)
                baseline_results[scenario_name] = results['untrained']
        
        baseline_semantic = self.test_semantic_similarity_preservation()
        baseline_transfer = self.test_information_transfer_quality()
        
        # 第二阶段：训练适配器（如果需要）
        if train_adapter:
            training_results = self.train_adapter(epochs=5)
        
        # 第三阶段：测试训练后的协作效果
        print("\n📊 阶段2: 测试训练后的协作效果")
        trained_results = {}
        
        for scenario_name, texts in self.test_scenarios.items():
            if scenario_name != "semantic_similarity":
                results = self.test_hidden_state_quality(scenario_name, texts)
                trained_results[scenario_name] = results['trained']
        
        trained_semantic = self.test_semantic_similarity_preservation()
        trained_transfer = self.test_information_transfer_quality()
        
        # 第四阶段：生成对比报告
        self._generate_comparison_report(
            baseline_results, trained_results, 
            baseline_semantic, trained_semantic,
            baseline_transfer, trained_transfer,
            training_results if train_adapter else None
        )
    
    def _generate_comparison_report(self, baseline_results, trained_results, 
                                   baseline_semantic, trained_semantic,
                                   baseline_transfer, trained_transfer,
                                   training_results):
        """生成详细的对比报告"""
        print("\n" + "=" * 60)
        print("📋 协作效果对比报告")
        print("=" * 60)
        
        # 1. Hidden State对齐效果对比
        print("\n1️⃣  Hidden State对齐效果对比:")
        print("-" * 40)
        
        for scenario in baseline_results.keys():
            baseline_cosine = np.mean(baseline_results[scenario]['cosine_similarities'])
            trained_cosine = np.mean(trained_results[scenario]['cosine_similarities'])
            baseline_mmd = np.mean(baseline_results[scenario]['mmd_losses'])
            trained_mmd = np.mean(trained_results[scenario]['mmd_losses'])
            
            cosine_improvement = ((trained_cosine - baseline_cosine) / abs(baseline_cosine)) * 100
            mmd_improvement = ((baseline_mmd - trained_mmd) / baseline_mmd) * 100
            
            print(f"\n📝 {scenario.replace('_', ' ').title()}:")
            print(f"   余弦相似度: {baseline_cosine:.4f} → {trained_cosine:.4f} (改善: {cosine_improvement:+.1f}%)")
            print(f"   MMD损失:    {baseline_mmd:.4f} → {trained_mmd:.4f} (改善: {mmd_improvement:+.1f}%)")
        
        # 2. 语义相似性保持对比
        print("\n2️⃣  语义相似性保持对比:")
        print("-" * 40)
        
        baseline_cross = np.mean([r['cross_model_similarity'] for r in baseline_semantic['untrained']])
        trained_cross = np.mean([r['cross_model_similarity'] for r in trained_semantic['trained']])
        cross_improvement = ((trained_cross - baseline_cross) / abs(baseline_cross)) * 100
        
        print(f"   跨模型语义一致性: {baseline_cross:.4f} → {trained_cross:.4f} (改善: {cross_improvement:+.1f}%)")
        
        # 3. 信息传递质量
        print("\n3️⃣  信息传递质量分析:")
        print("-" * 40)
        
        avg_activation_ratio = np.mean([r['activation_strength_ratio'] for r in trained_transfer['trained']])
        avg_variance_ratio = np.mean([r['variance_ratio'] for r in trained_transfer['trained']])
        avg_consistency = np.mean([r['cross_model_consistency'] for r in trained_transfer['trained']])
        
        print(f"   平均激活强度比: {avg_activation_ratio:.3f}")
        print(f"   平均方差比:     {avg_variance_ratio:.3f}")
        print(f"   平均跨模型一致性: {avg_consistency:.3f}")
        
        # 4. 训练效果总结
        if training_results:
            print("\n4️⃣  训练效果总结:")
            print("-" * 40)
            pre_cosine = training_results[0]['cosine']
            post_cosine = training_results[1]['cosine']
            pre_mmd = training_results[0]['mmd']
            post_mmd = training_results[1]['mmd']
            
            cosine_train_improvement = ((post_cosine - pre_cosine) / abs(pre_cosine)) * 100
            mmd_train_improvement = ((pre_mmd - post_mmd) / pre_mmd) * 100
            
            print(f"   训练期间余弦相似度提升: {cosine_train_improvement:+.1f}%")
            print(f"   训练期间MMD损失降低:   {mmd_train_improvement:+.1f}%")
        
        # 5. 结论和建议
        print("\n5️⃣  结论和建议:")
        print("-" * 40)
        
        overall_improvement = (cross_improvement + cosine_improvement) / 2
        
        if overall_improvement > 10:
            conclusion = "🎉 训练显著改善了模型协作效果！"
        elif overall_improvement > 5:
            conclusion = "✅ 训练适度改善了模型协作效果。"
        elif overall_improvement > 0:
            conclusion = "📈 训练略微改善了模型协作效果。"
        else:
            conclusion = "⚠️  训练效果不明显，建议调整训练策略。"
        
        print(f"   {conclusion}")
        print(f"   总体改善度: {overall_improvement:+.1f}%")
        
        if avg_consistency > 0.5:
            print("   ✅ 跨模型信息传递质量良好")
        else:
            print("   ⚠️  跨模型信息传递需要进一步优化")
        
        print("\n" + "=" * 60)


def main():
    """主函数"""
    print("🤖 多模型协作效果测试脚本")
    print("此脚本将比较训练前后的协作效果")
    
    # 创建测试器
    tester = CollaborationTester()
    
    # 运行综合测试
    tester.run_comprehensive_test(train_adapter=True)


if __name__ == "__main__":
    main()
