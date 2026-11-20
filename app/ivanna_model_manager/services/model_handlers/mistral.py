from .base import BaseModelHandler
from transformers import AutoModelForCausalLM, AutoTokenizer

class MistralHandler(BaseModelHandler):
    """Handler for Mistral models"""
    
    def get_model_class(self):
        return AutoModelForCausalLM
    
    def get_tokenizer_class(self):
        return AutoTokenizer
    
    def get_special_tokens(self):
        return {
            'pad_token': '</s>',
            'eos_token': '</s>',
            'bos_token': '<s>',
        }