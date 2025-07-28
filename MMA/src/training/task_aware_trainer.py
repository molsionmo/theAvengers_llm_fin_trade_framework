"""
任务感知训练器模块

专门针对不同任务类型优化的训练器
"""

import torch
import numpy as np

from .alignment_trainer import AlignmentTrainer
from ..tasks.detector import TaskDetector, TaskType


class TaskAwareTrainer(AlignmentTrainer):
    """任务感知的对齐训练器，专门针对不同任务类型优化"""
    
    def __init__(self, collaborator, learning_rate=1e-4, task_weight_decay=0.01):
        super().__init__(collaborator, learning_rate)
        self.task_weight_decay = task_weight_decay
        self.task_detector = TaskDetector()
        
        # 为不同任务类型维护独立的损失统计
        self.task_losses = {task_type: [] for task_type in TaskType}
        
        # 任务特定的学习率调度
        self.task_lr_schedulers = {}
    
    def task_specific_loss(self, proj1, proj2, task_type, temperature=0.1):
        """针对特定任务的损失函数"""
        # 基础对比损失
        base_loss = self.contrastive_loss(proj1, proj2, temperature)
        
        # 根据任务类型调整损失权重
        task_weights = {
            TaskType.QUESTION_ANSWERING: 1.2,  # 问答任务需要更强的语义对齐
            TaskType.SENTIMENT_ANALYSIS: 1.1,  # 情感分析需要情感信息保留
            TaskType.TEXT_GENERATION: 0.9,     # 生成任务允许更多创造性
            TaskType.CONVERSATION: 1.0,        # 对话任务标准权重
            TaskType.GENERAL: 1.0              # 通用任务标准权重
        }
        
        weight = task_weights.get(task_type, 1.0)
        
        # 添加任务特定的正则化
        if task_type == TaskType.QUESTION_ANSWERING:
            # 问答任务：增加信息保持损失
            info_preserve_loss = torch.mean(torch.abs(proj1.norm(dim=-1) - proj2.norm(dim=-1)))
            base_loss += 0.1 * info_preserve_loss
        elif task_type == TaskType.TEXT_GENERATION:
            # 生成任务：增加多样性奖励
            diversity_reward = -0.05 * torch.mean(torch.var(proj2, dim=1))
            base_loss += diversity_reward
        
        return weight * base_loss
    
    def adaptive_task_loss(self, proj1, proj2, task_type, epoch=0):
        """自适应任务损失，根据训练进度调整"""
        base_loss = self.task_specific_loss(proj1, proj2, task_type)
        
        # 根据训练进度调整损失权重
        progress_factor = min(1.0, epoch / 10.0)  # 10个epoch后达到完全权重
        
        if task_type == TaskType.QUESTION_ANSWERING:
            # 问答任务早期更注重对齐，后期更注重信息保持
            alignment_weight = 1.0 - 0.3 * progress_factor
            info_weight = 0.1 + 0.2 * progress_factor
            
            alignment_loss = self.contrastive_loss(proj1, proj2)
            info_loss = torch.mean(torch.abs(proj1.norm(dim=-1) - proj2.norm(dim=-1)))
            
            return alignment_weight * alignment_loss + info_weight * info_loss
        
        return base_loss
    
    def train_epoch_with_tasks(self, train_dataset, epoch=0):
        """基于任务类型的训练epoch"""
        self.collaborator.central_processor.train()
        task_losses = {task_type: 0.0 for task_type in TaskType}
        task_counts = {task_type: 0 for task_type in TaskType}
        
        for text in train_dataset:
            try:
                # 检测任务类型
                detected_task = self.task_detector.detect_task(text)
                
                # 获取两个模型的Hidden State
                hidden1 = self.collaborator.get_hidden_states(text, 0)
                hidden2 = self.collaborator.get_hidden_states(text, 1)
                
                # 使用任务感知投影
                projected_states = self.collaborator.central_processor.process(
                    [hidden1, hidden2], 
                    text=text, 
                    task_type=detected_task
                )
                proj1, proj2 = projected_states
                
                # 计算任务特定损失
                loss = self.adaptive_task_loss(proj1, proj2, detected_task, epoch)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 记录损失
                task_losses[detected_task] += loss.item()
                task_counts[detected_task] += 1
                
            except Exception as e:
                print(f"Error processing text '{text}': {e}")
                continue
        
        # 计算平均损失
        avg_task_losses = {}
        for task_type in TaskType:
            if task_counts[task_type] > 0:
                avg_task_losses[task_type] = task_losses[task_type] / task_counts[task_type]
                self.task_losses[task_type].append(avg_task_losses[task_type])
        
        return avg_task_losses
    
    def train_with_task_awareness(self, train_dataset, epochs=10, validation_dataset=None):
        """任务感知训练"""
        print(f"开始任务感知训练，共{epochs}个epoch...")
        
        all_task_losses = []
        val_results = []
        
        for epoch in range(epochs):
            # 训练
            task_losses = self.train_epoch_with_tasks(train_dataset, epoch)
            all_task_losses.append(task_losses)
            
            # 验证
            epoch_val_results = None
            if validation_dataset is not None:
                epoch_val_results = self.evaluate_alignment(validation_dataset)
                val_results.append(epoch_val_results)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print("  任务特定训练损失:")
            for task_type, loss in task_losses.items():
                print(f"    {task_type.value}: {loss:.4f}")
            
            if epoch_val_results:
                print(f"  验证余弦相似度: {epoch_val_results['cosine']:.4f}")
                print(f"  验证MMD损失: {epoch_val_results['mmd']:.4f}")
            print("-" * 50)
        
        return {
            'task_losses': all_task_losses,
            'val_results': val_results
        }
    
    def get_task_performance_summary(self):
        """获取各任务的性能摘要"""
        summary = {}
        for task_type, losses in self.task_losses.items():
            if losses:
                summary[task_type.value] = {
                    'average_loss': np.mean(losses),
                    'final_loss': losses[-1],
                    'improvement': losses[0] - losses[-1] if len(losses) > 1 else 0,
                    'stability': np.std(losses[-5:]) if len(losses) >= 5 else np.std(losses),
                    'num_samples': len(losses)
                }
        return summary
    
    def analyze_task_convergence(self):
        """分析任务收敛情况"""
        convergence_analysis = {}
        
        for task_type, losses in self.task_losses.items():
            if len(losses) >= 3:
                # 计算损失变化趋势
                recent_losses = losses[-3:]
                loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
                
                # 判断是否收敛
                loss_variance = np.var(recent_losses)
                is_converged = loss_variance < 0.001 and abs(loss_trend) < 0.001
                
                convergence_analysis[task_type.value] = {
                    'is_converged': is_converged,
                    'loss_trend': loss_trend,
                    'loss_variance': loss_variance,
                    'recent_losses': recent_losses
                }
        
        return convergence_analysis
    
    def balance_task_training(self, train_dataset, target_samples_per_task=100):
        """平衡不同任务类型的训练样本"""
        task_samples = {task_type: [] for task_type in TaskType}
        
        # 分类训练样本
        for text in train_dataset:
            detected_task = self.task_detector.detect_task(text)
            task_samples[detected_task].append(text)
        
        # 平衡样本数量
        balanced_dataset = []
        for task_type, samples in task_samples.items():
            if len(samples) > target_samples_per_task:
                # 随机采样
                import random
                selected_samples = random.sample(samples, target_samples_per_task)
            else:
                # 重复采样
                selected_samples = samples * (target_samples_per_task // len(samples) + 1)
                selected_samples = selected_samples[:target_samples_per_task]
            
            balanced_dataset.extend(selected_samples)
        
        # 打乱数据集
        import random
        random.shuffle(balanced_dataset)
        
        return balanced_dataset
    
    def get_task_distribution(self, dataset):
        """获取数据集中的任务分布"""
        task_counts = {task_type: 0 for task_type in TaskType}
        
        for text in dataset:
            detected_task = self.task_detector.detect_task(text)
            task_counts[detected_task] += 1
        
        total_samples = len(dataset)
        task_distribution = {
            task_type.value: count / total_samples 
            for task_type, count in task_counts.items()
        }
        
        return task_distribution
