from .base import BaseModelHandler
from transformers import AutoModelForCausalLM, AutoTokenizer

class LlamaHandler(BaseModelHandler):
    """Обработчик для моделей Llama"""
    
    def get_model_class(self):
        return AutoModelForCausalLM
    
    def get_tokenizer_class(self):
        return AutoTokenizer
    
    def get_special_tokens(self):
        return {
            'pad_token': '[PAD]',
            'eos_token': '</s>',
            'bos_token': '<s>',
        }