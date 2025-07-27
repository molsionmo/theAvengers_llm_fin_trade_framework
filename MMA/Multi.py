import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

class UnifiedTokenizer:
    """统一Tokenizer，为所有模型提供一致的编码接口"""
    def __init__(self, base_tokenizer):
        self.tokenizer = base_tokenizer
    
    def encode(self, text, model_name=None):
        return self.tokenizer.encode(text)
    
    def decode(self, tokens, model_name=None):
        return self.tokenizer.decode(tokens)


class TokenMapper(nn.Module):
    """学习将不同模型的Token映射到共享空间"""
    def __init__(self, tokenizers, shared_vocab_size):
        super().__init__()
        self.mappings = nn.ModuleList([
            nn.Embedding(tok.vocab_size, shared_vocab_size) 
            for tok in tokenizers
        ])
    
    def forward(self, token_ids, model_idx):
        return self.mappings[model_idx](token_ids)


class SemanticProjector(nn.Module):
    """将不同模型的Hidden State投影到共享语义空间"""
    def __init__(self, model_dims, shared_dim):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, shared_dim),
                nn.LayerNorm(shared_dim),
                nn.GELU(),
                nn.Linear(shared_dim, shared_dim),
                nn.LayerNorm(shared_dim)
            )
            for dim in model_dims
        ])
    
    def forward(self, hidden_states):
        return [proj(hs) for proj, hs in zip(self.projections, hidden_states)]


class HiddenStateAdapter(nn.Module):
    """适配器层，使模型能理解其他模型的Hidden State"""
    def __init__(self, source_dim, target_dim):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(source_dim, target_dim),
            nn.GELU(),
            nn.Linear(target_dim, target_dim),
            nn.LayerNorm(target_dim)
        )
    
    def forward(self, hidden_state):
        return self.adapter(hidden_state)


class CentralProcessingLayer(nn.Module):
    """中心处理层，统一处理和分发不同模型的Hidden State"""
    def __init__(self, model_dims, shared_dim):
        super().__init__()
        self.token_mapper = None  # 可选的Token映射器
        self.semantic_projector = SemanticProjector(model_dims, shared_dim)
        self.adapters = nn.ModuleDict()  # 动态生成的适配器
        self.shared_dim = shared_dim  # 保存shared_dim以便在适配器中使用
    
    def register_token_mapper(self, token_mapper):
        self.token_mapper = token_mapper
    
    def get_adapter(self, source_model_idx, target_model_idx):
        """获取或创建从source到target的适配器"""
        key = f"{source_model_idx}_{target_model_idx}"
        if key not in self.adapters:
            # 适配器在共享空间中工作，所以输入和输出都是shared_dim
            self.adapters[key] = HiddenStateAdapter(self.shared_dim, self.shared_dim)
        return self.adapters[key]
    
    def process(self, hidden_states, model_indices=None):
        """处理并分发Hidden State"""
        # 投影到共享空间
        projected_states = self.semantic_projector(hidden_states)
        
        # 如果需要特定模型的适配版本
        if model_indices is not None:
            source_idx, target_idx = model_indices
            adapter = self.get_adapter(source_idx, target_idx)
            return adapter(projected_states[source_idx])
        
        return projected_states


class MultiModelCollaborator:
    """多模型协作系统"""
    def __init__(self, models, tokenizers=None, shared_dim=512):
        self.models = models
        self.model_dims = [model.config.hidden_size for model in models]
        self.shared_dim = shared_dim
        
        # 初始化中心处理层
        self.central_processor = CentralProcessingLayer(self.model_dims, shared_dim)
        
        # 初始化统一Tokenizer（如果未提供）
        if tokenizers is None:
            self.tokenizers = []
            for model in models:
                tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
                # 为没有padding token的tokenizer设置padding token
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                self.tokenizers.append(tokenizer)
            self.unified_tokenizer = UnifiedTokenizer(self.tokenizers[0])
        else:
            self.tokenizers = tokenizers
            self.unified_tokenizer = UnifiedTokenizer(tokenizers[0])
    
    def get_hidden_states(self, text, model_idx):
        """获取特定模型的Hidden State"""
        inputs = self.tokenizers[model_idx].encode_plus(
            text, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = self.models[model_idx](**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1]  # 最后一层的Hidden State
    
    def collaborate(self, text, source_model_idx, target_model_idx):
        """模型间协作：从source_model获取信息，传递给target_model"""
        # 获取源模型的Hidden State
        source_hidden = self.get_hidden_states(text, source_model_idx)
        
        # 通过中心处理层进行转换
        adapted_hidden = self.central_processor.process(
            [source_hidden], 
            model_indices=(source_model_idx, target_model_idx)
        )
        
        # 获取目标模型的正常输出进行比较
        target_inputs = self.tokenizers[target_model_idx].encode_plus(
            text, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            # 正常的目标模型输出
            normal_outputs = self.models[target_model_idx](**target_inputs, output_hidden_states=True)
            
            # 这里我们返回适配后的hidden states和正常的模型输出进行比较
            # 在实际应用中，您可能想要将adapted_hidden进一步处理或用于特定任务
            
        return {
            'adapted_hidden': adapted_hidden,
            'normal_outputs': normal_outputs,
            'normal_hidden': normal_outputs.hidden_states[-1]
        }


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


class AlignmentTrainer:
    """对齐训练器，用于训练中心处理层"""
    def __init__(self, collaborator, learning_rate=1e-4):
        self.collaborator = collaborator
        self.optimizer = torch.optim.Adam(collaborator.central_processor.parameters(), lr=learning_rate)
        
    def contrastive_loss(self, proj1, proj2, temperature=0.1):
        """对比学习损失函数"""
        # 确保两个投影有相同的序列长度
        min_len = min(proj1.size(1), proj2.size(1))
        proj1 = proj1[:, :min_len, :]
        proj2 = proj2[:, :min_len, :]
        
        # 计算余弦相似度
        cosine_sim = F.cosine_similarity(proj1, proj2, dim=-1)
        
        # 对比损失：希望相同文本的投影尽可能相似
        loss = 1 - cosine_sim.mean()
        
        return loss
    
    def train_epoch(self, train_dataset):
        """训练一个epoch"""
        self.collaborator.central_processor.train()
        total_loss = 0
        num_batches = 0
        
        for text in train_dataset:
            try:
                # 获取两个模型的Hidden State
                hidden1 = self.collaborator.get_hidden_states(text, 0)
                hidden2 = self.collaborator.get_hidden_states(text, 1)
                
                # 投影到共享空间
                projected_states = self.collaborator.central_processor.process([hidden1, hidden2])
                proj1, proj2 = projected_states
                
                # 计算对比损失
                loss = self.contrastive_loss(proj1, proj2)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Error processing text '{text}': {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def train(self, train_dataset, epochs=10, validation_dataset=None):
        """训练对齐器"""
        print(f"开始训练对齐器，共{epochs}个epoch...")
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_dataset)
            
            # 验证
            val_results = None
            if validation_dataset is not None:
                val_results = self.evaluate_alignment(validation_dataset)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  训练损失: {train_loss:.4f}")
            if val_results:
                print(f"  验证余弦相似度: {val_results['cosine']:.4f}")
                print(f"  验证MMD损失: {val_results['mmd']:.4f}")
            print("-" * 50)
    
    def evaluate_alignment(self, dataset):
        """评估对齐效果"""
        self.collaborator.central_processor.eval()
        evaluator = AlignmentEvaluator(metrics=['cosine', 'mmd'])
        
        # 准备评估数据集格式
        eval_dataset = [(text, torch.tensor([0])) for text in dataset[:5]]  # 只用前5个样本评估
        
        with torch.no_grad():
            results = evaluator.evaluate(
                self.collaborator.models[0], 
                self.collaborator.models[1], 
                eval_dataset, 
                self.collaborator.central_processor
            )
        
        return results


# 新增训练演示函数
def train_demo():
    print("正在加载模型进行训练演示...")
    # 加载两个不同的模型
    model1 = AutoModel.from_pretrained("bert-base-uncased")
    model2 = AutoModel.from_pretrained("gpt2")
    
    print("创建多模型协作系统...")
    collaborator = MultiModelCollaborator([model1, model2])
    
    print("创建训练器...")
    trainer = AlignmentTrainer(collaborator, learning_rate=1e-4)
    
    # 准备训练数据集
    train_texts = [
        "What is the capital of France?",
        "How are you today?",
        "The weather is nice.",
        "I love machine learning.",
        "Python is a great programming language.",
        "The sun is shining brightly.",
        "Coffee tastes delicious.",
        "Books are full of knowledge."
    ]
    
    # 准备验证数据集
    val_texts = [
        "What is your name?",
        "Good morning everyone.",
        "Technology is advancing rapidly."
    ]
    
    print("训练前的对齐效果:")
    pre_train_results = trainer.evaluate_alignment(val_texts)
    print(f"  余弦相似度: {pre_train_results['cosine']:.4f}")
    print(f"  MMD损失: {pre_train_results['mmd']:.4f}")
    print()
    
    # 训练对齐器
    trainer.train(train_texts, epochs=5, validation_dataset=val_texts)
    
    print("\n训练完成！")
    print("训练后的对齐效果:")
    post_train_results = trainer.evaluate_alignment(val_texts)
    print(f"  余弦相似度: {post_train_results['cosine']:.4f}")
    print(f"  MMD损失: {post_train_results['mmd']:.4f}")
    
    # 测试协作效果
    test_text = "What is the capital of France?"
    outputs = collaborator.collaborate(test_text, 0, 1)
    print(f"\n协作测试结果:")
    print(f"  适配后的hidden state shape: {outputs['adapted_hidden'].shape}")
    print(f"  正常hidden state shape: {outputs['normal_hidden'].shape}")


# 示例使用
def demo():
    print("正在加载模型...")
    # 加载两个不同的模型
    model1 = AutoModel.from_pretrained("bert-base-uncased")
    model2 = AutoModel.from_pretrained("gpt2")
    
    print("创建多模型协作系统...")
    # 创建多模型协作系统
    collaborator = MultiModelCollaborator([model1, model2])
    
    # 示例文本
    text = "What is the capital of France?"
    
    print("模型间协作测试...")
    # 模型间协作：从BERT获取信息，传递给GPT2
    outputs = collaborator.collaborate(text, 0, 1)
    
    print(f"适配后的hidden state shape: {outputs['adapted_hidden'].shape}")
    print(f"正常hidden state shape: {outputs['normal_hidden'].shape}")
    
    # 评估对齐效果（不使用t-SNE可视化）
    print("评估对齐效果...")
    evaluator = AlignmentEvaluator(metrics=['cosine', 'mmd'])
    # 假设有一个数据集
    dataset = [("What is your name?", torch.tensor([0]))] * 3
    alignment_results = evaluator.evaluate(model1, model2, dataset, collaborator.central_processor)
    
    print("Alignment Results:", alignment_results)
    print("演示完成！")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_demo()
    else:
        demo()