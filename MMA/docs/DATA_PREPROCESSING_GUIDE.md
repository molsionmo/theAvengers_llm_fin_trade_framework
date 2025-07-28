# 数据预处理和训练指南

本项目现在支持从互联网数据集自动下载和预处理数据，用于训练任务感知适配器。本指南将介绍如何使用这些新功能。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 查看可用数据集

```bash
python main.py data list
```

这将显示所有支持的数据集，按任务类型分组：

- **问答任务**: SQuAD, SQuAD v2
- **情感分析**: IMDB, SST-2
- **文本分类**: AG News, 20 Newsgroups
- **文本相似度**: STS Benchmark
- **自然语言推理**: SNLI, MNLI
- **文本生成**: WikiText, OpenWebText
- **对话任务**: Persona Chat
- **摘要任务**: CNN/DailyMail, XSum

### 3. 创建混合数据集

```bash
python main.py data create-mixed imdb ag_news squad --max-samples-per-dataset 1000 --output-dir ./my_data
```

这将：
- 从IMDB、AG News和SQuAD数据集各下载1000个样本
- 预处理为适合训练的格式
- 保存到`./my_data`目录

### 4. 使用数据集训练

```bash
python main.py train-with-data imdb ag_news --epochs 10 --batch-size 16 --data-dir ./my_data
```

## 📊 数据预处理功能

### 支持的数据集类型

| 任务类型 | 数据集示例 | 用途 |
|---------|-----------|------|
| question_answering | squad, squad_v2 | 问答系统训练 |
| sentiment_analysis | imdb, sst2 | 情感分析训练 |
| text_classification | ag_news, 20newsgroups | 文本分类训练 |
| text_similarity | sts_benchmark | 语义相似度训练 |
| natural_language_inference | snli, mnli | 自然语言推理训练 |
| text_generation | wikitext, openwebtext | 文本生成训练 |
| conversation | persona_chat | 对话系统训练 |
| summarization | cnn_dailymail, xsum | 文本摘要训练 |

### 数据预处理命令

#### 列出所有可用数据集
```bash
python scripts/preprocess_data.py list
```

#### 下载单个数据集
```bash
python scripts/preprocess_data.py download imdb --max-samples 500 --output-dir ./data
```

#### 创建混合数据集
```bash
python scripts/preprocess_data.py create-mixed imdb ag_news squad --max-samples-per-dataset 1000 --output-dir ./mixed_data
```

#### 预处理用于任务检测
```bash
python scripts/preprocess_data.py preprocess-task imdb ag_news --max-samples-per-dataset 500 --output-dir ./task_data
```

#### 预处理用于适配器训练
```bash
python scripts/preprocess_data.py preprocess-adapter imdb ag_news --max-samples-per-dataset 1000 --tokenizer bert-base-uncased --output-dir ./adapter_data
```

#### 验证处理后的数据
```bash
python scripts/preprocess_data.py validate ./processed_data
```

#### 查看数据统计信息
```bash
python scripts/preprocess_data.py stats ./processed_data
```

## 🎯 任务感知训练

### 训练参数配置

使用`train-with-data`命令时，可以配置以下参数：

- `--epochs`: 训练轮数 (默认: 10)
- `--batch-size`: 批次大小 (默认: 16)
- `--learning-rate`: 学习率 (默认: 1e-4)
- `--max-samples-per-dataset`: 每个数据集的最大样本数 (默认: 1000)
- `--sampling-strategy`: 任务采样策略 (balanced/proportional/random, 默认: balanced)
- `--validation-split`: 验证集比例 (默认: 0.2)
- `--tokenizer`: 分词器名称 (默认: bert-base-uncased)

### 示例训练命令

```bash
# 基础训练
python main.py train-with-data imdb ag_news

# 高级配置训练
python main.py train-with-data imdb ag_news squad sst2 \
    --epochs 15 \
    --batch-size 32 \
    --learning-rate 5e-5 \
    --max-samples-per-dataset 2000 \
    --sampling-strategy balanced \
    --validation-split 0.15
```

## 🔧 任务采样策略

### balanced (平衡采样)
- 每个任务类型获得相等的采样权重
- 适用于希望模型对所有任务类型都有良好性能的场景

### proportional (比例采样)
- 按照原始数据集中的任务分布进行采样
- 保持数据的自然分布

### random (随机采样)
- 完全随机采样
- 用于探索性实验

## 📈 数据集统计和分析

### 查看数据集信息
```python
from src.data.dataset_loader import DatasetLoader

loader = DatasetLoader('./processed_data')
stats = loader.get_data_statistics()
print(stats)
```

### 任务感知数据集功能
```python
from src.data.task_dataset import TaskAwareDataset

# 创建任务感知数据集
dataset = TaskAwareDataset(
    data,
    tokenizer,
    task_sampling_strategy='balanced',
    include_task_tokens=True
)

# 获取特定任务的批次
qa_batch = dataset.get_task_batch('question_answering', batch_size=8)

# 获取混合任务批次
mixed_batch = dataset.get_mixed_task_batch(batch_size=16)

# 查看统计信息
stats = dataset.get_task_statistics()
```

## 🎮 演示脚本

运行数据预处理演示：
```bash
python examples/data_preprocessing_demo.py
```

这个演示将：
1. 展示如何下载和预处理数据集
2. 创建任务感知数据集
3. 使用数据集进行适配器训练
4. 显示训练结果和统计信息

## ⚠️ 注意事项

1. **网络连接**: 首次运行需要从HuggingFace下载数据集，确保网络连接正常
2. **存储空间**: 某些数据集较大，确保有足够的磁盘空间
3. **内存使用**: 大数据集可能需要较多内存，可以通过`--max-samples-per-dataset`参数限制样本数量
4. **GPU支持**: 训练过程支持GPU加速，如果有CUDA可用会自动使用

## 🔍 故障排除

### 常见问题

1. **下载超时**: 可以设置HuggingFace代理或使用缓存
2. **内存不足**: 减少批次大小和样本数量
3. **CUDA错误**: 确保PyTorch CUDA版本与系统CUDA版本匹配

### 获取帮助

```bash
python main.py --help
python scripts/preprocess_data.py --help
```

## 📝 示例工作流

完整的数据预处理和训练工作流：

```bash
# 1. 查看可用数据集
python main.py data list

# 2. 创建混合数据集
python main.py data create-mixed imdb ag_news squad \
    --max-samples-per-dataset 1000 \
    --output-dir ./training_data

# 3. 验证数据
python main.py data validate ./training_data

# 4. 查看统计信息
python main.py data stats ./training_data

# 5. 开始训练
python main.py train-with-data imdb ag_news squad \
    --data-dir ./training_data \
    --epochs 10 \
    --batch-size 16

# 6. 运行评估测试
python main.py test
```

这样就完成了从数据预处理到模型训练的完整流程！
