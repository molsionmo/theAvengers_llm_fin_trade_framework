# 任务感知多模型协作框架

一个支持任务感知的多模型协作系统，能够根据不同的任务类型动态调整模型间的协作策略和Hidden State适配方式。

## 📋 功能特性

- **🎯 任务自动检测**: 基于正则表达式的任务类型自动识别
- **🤝 多模型协作**: 支持不同架构模型间的Hidden State协作
- **🧠 任务感知适配**: 根据任务类型动态调整适配策略
- **🔧 对齐训练**: 支持对比学习和任务特定的训练方法
- **📊 效果评估**: 提供多种对齐效果评估指标
- **🎮 易于使用**: 简洁的API设计，支持快速集成

## 🏗️ 项目结构

```
MMA/
├── src/                          # 源代码
│   ├── core/                     # 核心组件
│   │   ├── __init__.py
│   │   ├── collaborator.py       # 多模型协作器
│   │   ├── adapters.py          # 任务感知适配器
│   │   ├── projector.py         # 语义投影器
│   │   └── processor.py         # 中心处理器
│   ├── tasks/                    # 任务模块
│   │   ├── __init__.py
│   │   └── detector.py          # 任务检测器
│   ├── training/                 # 训练模块
│   │   ├── __init__.py
│   │   ├── alignment_trainer.py # 对齐训练器
│   │   └── task_aware_trainer.py # 任务感知训练器
│   ├── utils/                    # 工具模块
│   │   ├── __init__.py
│   │   ├── evaluator.py         # 对齐评估器
│   │   └── tokenizer.py         # 统一tokenizer
│   └── __init__.py
├── tests/                        # 测试文件
├── examples/                     # 示例代码
├── config/                       # 配置文件
├── results/                      # 实验结果
└── README.md
```

## 🚀 快速开始

### 安装依赖

```bash
pip install torch transformers scikit-learn matplotlib numpy
```

### 基础使用

```python
import sys
sys.path.append('src')

from transformers import AutoModel
from src.core.collaborator import MultiModelCollaborator
from src.tasks.detector import TaskType

# 1. 加载模型
bert_model = AutoModel.from_pretrained("bert-base-uncased")
gpt2_model = AutoModel.from_pretrained("gpt2")

# 2. 创建协作系统
collaborator = MultiModelCollaborator([bert_model, gpt2_model])

# 3. 任务检测
text = "What is machine learning?"
task = collaborator.detect_task_for_text(text)
print(f"检测到任务: {task.value}")

# 4. 模型协作
result = collaborator.collaborate(
    text, 
    source_model_idx=0,  # BERT
    target_model_idx=1,  # GPT-2
    task_type=task
)

print(f"适配后 shape: {result['adapted_hidden'].shape}")
```

### 运行示例

```bash
# 基础演示
python examples/demo.py basic

# 训练演示
python examples/demo.py training

# 效果对比
python examples/demo.py comparison

# 交互式演示
python examples/demo.py interactive

# 快速开始
python examples/quick_start.py
```

## 📊 支持的任务类型

- **问答 (QA)**: 问题回答任务
- **情感分析 (Sentiment)**: 情感极性判断
- **文本生成 (Generation)**: 创意文本生成
- **对话 (Conversation)**: 对话交互
- **文本分类 (Classification)**: 通用分类任务
- **命名实体识别 (NER)**: 实体识别
- **摘要 (Summarization)**: 文本摘要
- **翻译 (Translation)**: 语言翻译
- **通用 (General)**: 其他通用任务

## 🔧 任务感知训练

```python
from src.training.task_aware_trainer import TaskAwareTrainer

# 创建训练器
trainer = TaskAwareTrainer(collaborator, learning_rate=1e-4)

# 准备训练数据
train_texts = [
    "What is AI?",
    "I love this product!",
    "Write a story...",
    # ... 更多文本
]

# 开始训练
results = trainer.train_with_task_awareness(
    train_texts, 
    epochs=10
)

# 查看性能摘要
summary = trainer.get_task_performance_summary()
print(summary)
```

## 📈 实验结果

基于我们的测试，任务感知协作系统在以下方面表现出色：

- **任务检测准确率**: 88.89%
- **适配效果改善**: 根据任务类型有不同程度的提升
- **训练收敛**: 大多数任务在3-5个epoch内收敛

详细结果请查看 `results/` 目录中的报告文件。

## 🧪 测试

运行测试套件：

```bash
# 运行任务感知测试
python tests/task_aware_test.py

# 运行协作测试
python tests/test_collaboration.py

# 运行快速测试
python tests/quick_test.py
```

## 🔬 架构说明

### 核心组件

1. **MultiModelCollaborator**: 主要的协作系统，管理多个模型的交互
2. **TaskDetector**: 基于文本特征自动检测任务类型
3. **TaskAwareAdapter**: 任务感知的适配器，根据任务调整转换策略
4. **SemanticProjector**: 将不同模型的Hidden State投影到共享语义空间
5. **CentralProcessingLayer**: 中心处理层，统一管理协作流程

### 工作流程

1. **文本输入** → **任务检测** → **Hidden State提取**
2. **语义投影** → **任务感知适配** → **协作输出**
3. **对齐训练** → **效果评估** → **性能优化**

## 📝 配置

配置文件位于 `config/config.py`，包含：

- 模型参数设置
- 训练超参数
- 任务检测规则
- 评估指标配置

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

MIT License

## 📞 联系方式

如有问题，请创建Issue或联系项目维护者。

---

## 🎯 下一步计划

- [ ] 支持更多预训练模型
- [ ] 优化任务检测算法
- [ ] 添加更多评估指标
- [ ] 支持分布式训练
- [ ] 提供预训练模型权重
