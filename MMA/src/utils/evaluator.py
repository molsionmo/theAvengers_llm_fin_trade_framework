"""
对齐评估器模块

用于评估Hidden State对齐效果
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import AutoTokenizer


class AlignmentEvaluator:
    """Hidden State对齐评估器"""
    def __init__(self, metrics=['cosine', 'mmd', 'tsne']):
        self.metrics = metrics
    
    def cosine_similarity(self, hidden_states_1, hidden_states_2):
        """计算余弦相似度"""
        # 确保两个hidden states有相同的序列长度
        min_len = min(hidden_states_1.size(1), hidden_states_2.size(1))
        hidden_states_1 = hidden_states_1[:, :min_len, :]
        hidden_states_2 = hidden_states_2[:, :min_len, :]
        
        # 如果hidden states的特征维度不同，使用平均池化
        if hidden_states_1.size(-1) != hidden_states_2.size(-1):
            # 使用平均池化来标准化特征维度
            hidden_states_1 = hidden_states_1.mean(dim=-1, keepdim=True)
            hidden_states_2 = hidden_states_2.mean(dim=-1, keepdim=True)
        
        sim_matrix = F.cosine_similarity(hidden_states_1, hidden_states_2, dim=-1)
        return sim_matrix.mean().item()
    
    def mmd_loss(self, x, y, kernel='rbf'):
        """计算MMD损失评估分布匹配程度"""
        # 确保两个张量有相同的特征维度
        if x.size(-1) != y.size(-1):
            # 如果维度不同，将它们投影到较小的维度
            min_dim = min(x.size(-1), y.size(-1))
            x = x[:, :min_dim]
            y = y[:, :min_dim]
        
        # 确保有足够的样本
        min_samples = min(x.size(0), y.size(0))
        if min_samples < 2:
            return torch.tensor(0.0)
        
        x = x[:min_samples]
        y = y[:min_samples]
        
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
        
        dxx = rx.t() + rx - 2. * xx
        dyy = ry.t() + ry - 2. * yy
        dxy = rx.t() + ry - 2. * zz
        
        gamma = 1.0 / x.size(1) if x.size(1) > 0 else 1.0
        XX = torch.exp(-gamma * dxx)
        YY = torch.exp(-gamma * dyy)
        XY = torch.exp(-gamma * dxy)
        
        return torch.mean(XX + YY - 2. * XY)
    
    def visualize_tsne(self, hidden_states_1, hidden_states_2, labels=None, save_path=None):
        """使用t-SNE可视化对齐效果"""
        combined = torch.cat([hidden_states_1, hidden_states_2], dim=0).cpu().numpy()
        
        tsne = TSNE(n_components=2, random_state=42)
        reduced = tsne.fit_transform(combined)
        
        n1 = hidden_states_1.shape[0]
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced[:n1, 0], reduced[:n1, 1], c='blue', label='Model 1')
        plt.scatter(reduced[n1:, 0], reduced[n1:, 1], c='red', label='Model 2')
        
        if labels is not None:
            pass  # 可根据标签进一步着色
        
        plt.legend()
        plt.title('Hidden State Alignment Visualization')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def evaluate(self, model_A, model_B, dataset, central_processor=None):
        """评估两个模型之间的Hidden State对齐效果"""
        results = {}
        
        for text, labels in dataset:
            # 为两个模型准备输入
            tokenizer_A = AutoTokenizer.from_pretrained(model_A.config._name_or_path)
            tokenizer_B = AutoTokenizer.from_pretrained(model_B.config._name_or_path)
            
            # 设置padding token
            if tokenizer_A.pad_token is None:
                tokenizer_A.pad_token = tokenizer_A.eos_token
            if tokenizer_B.pad_token is None:
                tokenizer_B.pad_token = tokenizer_B.eos_token
            
            inputs_A = tokenizer_A.encode_plus(text, return_tensors="pt", padding=True, truncation=True)
            inputs_B = tokenizer_B.encode_plus(text, return_tensors="pt", padding=True, truncation=True)
            
            # 获取原始Hidden State
            with torch.no_grad():
                hidden_A = model_A(**inputs_A, output_hidden_states=True).hidden_states[-1]
                hidden_B = model_B(**inputs_B, output_hidden_states=True).hidden_states[-1]
            
            # 如果提供了中心处理层，使用对齐后的Hidden State
            if central_processor:
                projected_states = central_processor.process([hidden_A, hidden_B])
                hidden_A, hidden_B = projected_states
            
            # 计算各项指标
            if 'cosine' in self.metrics:
                results['cosine'] = self.cosine_similarity(hidden_A, hidden_B)
            
            if 'mmd' in self.metrics:
                results['mmd'] = self.mmd_loss(hidden_A.view(-1, hidden_A.size(-1)), 
                                              hidden_B.view(-1, hidden_B.size(-1)))
            
            if 'tsne' in self.metrics and len(results) < 100:  # 限制可视化样本量
                self.visualize_tsne(hidden_A, hidden_B)
            
            # 只处理第一个样本，避免重复计算
            break
        
        return results
    
    def batch_evaluate(self, model_A, model_B, dataset, central_processor=None, batch_size=8):
        """批量评估对齐效果"""
        all_results = []
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            batch_results = []
            
            for text, labels in batch:
                try:
                    result = self.evaluate(model_A, model_B, [(text, labels)], central_processor)
                    batch_results.append(result)
                except Exception as e:
                    print(f"Error evaluating text '{text}': {e}")
                    continue
            
            all_results.extend(batch_results)
        
        # 计算平均指标
        if all_results:
            avg_results = {}
            for metric in self.metrics:
                if metric in ['cosine', 'mmd']:
                    values = [r[metric] for r in all_results if metric in r]
                    if values:
                        avg_results[f"avg_{metric}"] = np.mean(values)
                        avg_results[f"std_{metric}"] = np.std(values)
            
            return avg_results
        
        return {}
    
    def compare_alignment_methods(self, model_A, model_B, dataset, processors=None):
        """比较不同对齐方法的效果"""
        if processors is None:
            processors = {'baseline': None}
        
        comparison_results = {}
        
        for method_name, processor in processors.items():
            print(f"Evaluating {method_name}...")
            results = self.batch_evaluate(model_A, model_B, dataset, processor)
            comparison_results[method_name] = results
        
        return comparison_results
