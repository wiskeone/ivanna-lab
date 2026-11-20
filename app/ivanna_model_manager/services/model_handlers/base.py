from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import logging

logger = logging.getLogger(__name__)

class BaseModelHandler(ABC):
    """Abstract base class for model handlers"""
    
    def __init__(self, model_config):
        self.model_config = model_config
        self.model = None
        self.tokenizer = None
        self.model_id = model_config.get('huggingface_id', '')
        self.quantization = model_config.get('quantization', True)
    
    @abstractmethod
    def get_model_class(self):
        """Returns the model class to load"""
        pass
    
    @abstractmethod
    def get_tokenizer_class(self):
        """Returns the tokenizer class"""
        pass
    
    @abstractmethod
    def get_special_tokens(self):
        """Returns special tokens for the model"""
        pass
    
    def load_model(self, device_map="auto"):
        """Loads the model from Hugging Face"""
        try:
            logger.info(f"Loading model {self.model_id} with config: {self.model_config}")

            # Quantization configuration
            bnb_config = None
            if self.quantization:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Load the model
            model_class = self.get_model_class()
            self.model = model_class.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=self.model_config.get('trust_remote_code', True),
                **self.model_config.get('model_kwargs', {})
            )
            
            # Load the tokenizer
            tokenizer_class = self.get_tokenizer_class()
            self.tokenizer = tokenizer_class.from_pretrained(
                self.model_id,
                trust_remote_code=self.model_config.get('trust_remote_code', True)
            )
            
            # Set special tokens
            special_tokens = self.get_special_tokens()
            if special_tokens:
                self.tokenizer.add_special_tokens(special_tokens)
                if hasattr(self.model, 'resize_token_embeddings'):
                    self.model.resize_token_embeddings(len(self.tokenizer))
            
            logger.info(f"Model {self.model_id} loaded successfully")
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Model loading error {self.model_id}: {str(e)}")
            raise
    
    def generate(self, prompt, **kwargs):
        """Generates text from the prompt"""
        if not self.model or not self.tokenizer:
            raise ValueError("Модель не загружена. Сначала вызовите load_model()")
        
        default_params = {
            'max_length': 512,
            'temperature': 0.7,
            'do_sample': True,
            'top_p': 0.9,
            'pad_token_id': self.tokenizer.eos_token_id,
        }
        default_params.update(kwargs)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **default_params)
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def get_model_info(self):
        """Returns model info"""
        if not self.model:
            return {"error": "Модель не загружена"}
        
        info = {
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.model.device),
            "dtype": str(next(self.model.parameters()).dtype),
            "layers": len(self.model.model.layers) if hasattr(self.model.model, 'layers') else 'N/A',
        }
        return info