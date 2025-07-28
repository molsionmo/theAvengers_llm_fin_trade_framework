"""
统一Tokenizer模块

为所有模型提供一致的编码接口
"""


class UnifiedTokenizer:
    """统一Tokenizer，为所有模型提供一致的编码接口"""
    def __init__(self, base_tokenizer):
        self.tokenizer = base_tokenizer
    
    def encode(self, text, model_name=None):
        """编码文本"""
        return self.tokenizer.encode(text)
    
    def decode(self, tokens, model_name=None):
        """解码tokens"""
        return self.tokenizer.decode(tokens)
    
    def encode_plus(self, text, **kwargs):
        """增强编码，支持额外参数"""
        return self.tokenizer.encode_plus(text, **kwargs)
    
    def batch_encode_plus(self, texts, **kwargs):
        """批量编码"""
        return self.tokenizer.batch_encode_plus(texts, **kwargs)
    
    def get_vocab_size(self):
        """获取词汇表大小"""
        return len(self.tokenizer.vocab)
    
    def get_special_tokens(self):
        """获取特殊tokens"""
        return {
            'pad_token': self.tokenizer.pad_token,
            'eos_token': self.tokenizer.eos_token,
            'bos_token': getattr(self.tokenizer, 'bos_token', None),
            'unk_token': getattr(self.tokenizer, 'unk_token', None),
            'cls_token': getattr(self.tokenizer, 'cls_token', None),
            'sep_token': getattr(self.tokenizer, 'sep_token', None),
            'mask_token': getattr(self.tokenizer, 'mask_token', None)
        }
