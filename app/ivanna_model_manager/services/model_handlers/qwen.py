from .base import BaseModelHandler
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenHandler(BaseModelHandler):
    """Обработчик для моделей Qwen"""
    
    def get_model_class(self):
        return AutoModelForCausalLM
    
    def get_tokenizer_class(self):
        return AutoTokenizer
    
    def get_special_tokens(self):
        return {
            'pad_token': '<|endoftext|>',
            'eos_token': '<|endoftext|>',
            'bos_token': '<|endoftext|>',
        }
    
    def load_model(self, device_map="auto"):
        # Qwen требует trust_remote_code=True
        self.model_config['trust_remote_code'] = True
        return super().load_model(device_map)