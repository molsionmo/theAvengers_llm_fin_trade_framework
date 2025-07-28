"""
任务类型检测模块

提供任务类型枚举和自动检测功能
"""

import re
from enum import Enum
from typing import Dict, List


class TaskType(Enum):
    """任务类型枚举"""
    QUESTION_ANSWERING = "qa"
    TEXT_CLASSIFICATION = "classification"
    SENTIMENT_ANALYSIS = "sentiment"
    NAMED_ENTITY_RECOGNITION = "ner"
    TEXT_GENERATION = "generation"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CONVERSATION = "conversation"
    GENERAL = "general"


class TaskDetector:
    """任务类型检测器，基于文本特征自动检测任务类型"""
    
    def __init__(self):
        self.patterns = {
            TaskType.QUESTION_ANSWERING: [
                r"\bwhat\b", r"\bhow\b", r"\bwhy\b", r"\bwhen\b", r"\bwhere\b", r"\bwho\b",
                r"\?\s*$", r"\bexplain\b", r"\bdescribe\b", r"\bdefine\b", r"\btell me\b"
            ],
            TaskType.SENTIMENT_ANALYSIS: [
                r"\blove\b", r"\bhate\b", r"\blike\b", r"\bdislike\b", r"\bfeel\b", 
                r"\bopinion\b", r"\bthink\b", r"\bemotion\b", r"\bmood\b",
                r"\bpositive\b", r"\bnegative\b", r"\bhappy\b", r"\bsad\b",
                r"\bawesome\b", r"\bterrible\b", r"\bgreat\b", r"\bbad\b",
                r"\bwonderful\b", r"\bawful\b", r"\bamazing\b", r"\bhorrible\b"
            ],
            TaskType.TEXT_GENERATION: [
                r"\bgenerate\b", r"\bcreate\b", r"\bwrite\b", r"\bcontinue\b", r"\bcomplete\b",
                r"\bstory\b", r"\bpoem\b", r"\barticle\b", r"\bcompose\b"
            ],
            TaskType.SUMMARIZATION: [
                r"\bsummarize\b", r"\bsummary\b", r"\bmain points\b", r"\bkey points\b"
            ],
            TaskType.TRANSLATION: [
                r"\btranslate\b", r"\btranslation\b", r"\bin [a-z]+ language\b"
            ],
            TaskType.CONVERSATION: [
                r"\bhello\b", r"\bhi\b", r"\bhow are you\b", r"\bgood morning\b", r"\bgood evening\b",
                r"\bnice to meet\b", r"\bbye\b", r"\bgoodbye\b", r"\bsee you\b"
            ]
        }
    
    def detect_task(self, text: str) -> TaskType:
        """检测文本对应的任务类型"""
        text_lower = text.lower()
        
        task_scores = {}
        for task_type, patterns in self.patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, text_lower))
            if score > 0:
                task_scores[task_type] = score
        
        if task_scores:
            return max(task_scores.items(), key=lambda x: x[1])[0]
        return TaskType.GENERAL
    
    def get_task_confidence(self, text: str) -> Dict[TaskType, float]:
        """获取各任务类型的置信度分数"""
        text_lower = text.lower()
        
        task_scores = {}
        total_matches = 0
        
        for task_type, patterns in self.patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, text_lower))
            task_scores[task_type] = score
            total_matches += score
        
        if total_matches == 0:
            return {TaskType.GENERAL: 1.0}
        
        # 归一化为概率分布
        task_probabilities = {}
        for task_type, score in task_scores.items():
            task_probabilities[task_type] = score / total_matches
        
        return task_probabilities
    
    def add_pattern(self, task_type: TaskType, pattern: str):
        """为特定任务类型添加新的检测模式"""
        if task_type not in self.patterns:
            self.patterns[task_type] = []
        self.patterns[task_type].append(pattern)
    
    def remove_pattern(self, task_type: TaskType, pattern: str):
        """移除特定任务类型的检测模式"""
        if task_type in self.patterns and pattern in self.patterns[task_type]:
            self.patterns[task_type].remove(pattern)
