# 多模型协作系统配置文件

# 模型配置
MODEL_CONFIG = {
    "model1_name": "bert-base-uncased",
    "model2_name": "gpt2", 
    "shared_dim": 512,  # 共享空间维度
}

# 训练配置
TRAINING_CONFIG = {
    "learning_rate": 1e-4,
    "epochs": 5,
    "batch_size": 1,  # 当前实现为单样本
    "temperature": 0.1,  # 对比学习温度参数
}

# 评估配置
EVALUATION_CONFIG = {
    "metrics": ["cosine", "mmd"],
    "test_scenarios": [
        "question_answering",
        "sentiment_analysis", 
        "text_completion",
        "semantic_similarity"
    ]
}

# 数据配置
DATA_CONFIG = {
    "train_texts": [
        "What is artificial intelligence?",
        "The weather is beautiful today",
        "I love reading books",
        "Machine learning is powerful",
        "The ocean is vast and deep",
        "Music brings joy to people",
        "Technology changes our lives",
        "Education opens new doors",
        "Science helps us understand the world",
        "Art expresses human creativity"
    ],
    "test_texts": [
        "What is the capital of France?",
        "How are you feeling today?",
        "The future looks bright"
    ]
}

# 可以通过修改这些配置来调优系统性能
