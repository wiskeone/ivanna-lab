from .model_handlers.qwen import QwenHandler
from .model_handlers.llama import LlamaHandler
from .model_handlers.mistral import MistralHandler
import logging

logger = logging.getLogger(__name__)

class ModelLoaderFactory:
    """Factory for creating model handlers"""
    
    _handlers = {
        'QWEN': QwenHandler,
        'LLAMA': LlamaHandler,
        'MISTRAL': MistralHandler,
    }
    
    @classmethod
    def register_handler(cls, model_type, handler_class):
        """Registers a new handler"""
        cls._handlers[model_type.upper()] = handler_class
        logger.info(f"Handler registered for type: {model_type}")
    
    @classmethod
    def get_handler(cls, model_checkpoint):
        """Creates a handler for the model"""
        model_type = model_checkpoint.model_type

        # Prepare the configuration
        config = model_checkpoint.loader_config.copy()
        config.update({
            'huggingface_id': model_checkpoint.huggingface_id or model_checkpoint.name,
            'quantization': model_checkpoint.quantization == '4bit',
        })
        
        # Get the handler class
        handler_class = cls._handlers.get(model_type)
        if not handler_class:
            # Use QwenHandler as the fallback for unknown types
            logger.warning(f"Unknown model type {model_type}, using QwenHandler")
            handler_class = QwenHandler
            config['trust_remote_code'] = True
        
        # Create the handler instance
        return handler_class(config)
    
    @classmethod
    def get_supported_types(cls):
        """Returns the list of supported model types"""
        return list(cls._handlers.keys())