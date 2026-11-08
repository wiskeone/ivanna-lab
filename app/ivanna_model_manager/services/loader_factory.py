from .model_handlers.qwen import QwenHandler
from .model_handlers.llama import LlamaHandler
from .model_handlers.mistral import MistralHandler
import logging

logger = logging.getLogger(__name__)

class ModelLoaderFactory:
    """Фабрика для создания обработчиков моделей"""
    
    _handlers = {
        'QWEN': QwenHandler,
        'LLAMA': LlamaHandler,
        'MISTRAL': MistralHandler,
    }
    
    @classmethod
    def register_handler(cls, model_type, handler_class):
        """Регистрирует новый обработчик"""
        cls._handlers[model_type.upper()] = handler_class
        logger.info(f"Зарегистрирован обработчик для типа: {model_type}")
    
    @classmethod
    def get_handler(cls, model_checkpoint):
        """Создаёт обработчик для модели"""
        model_type = model_checkpoint.model_type
        
        # Подготавливаем конфигурацию
        config = model_checkpoint.loader_config.copy()
        config.update({
            'huggingface_id': model_checkpoint.huggingface_id or model_checkpoint.name,
            'quantization': model_checkpoint.quantization == '4bit',
        })
        
        # Получаем класс обработчика
        handler_class = cls._handlers.get(model_type)
        if not handler_class:
            # Используем QwenHandler как fallback для неизвестных типов
            logger.warning(f"Неизвестный тип модели {model_type}, используем QwenHandler")
            handler_class = QwenHandler
            config['trust_remote_code'] = True
        
        # Создаём экземпляр обработчика
        return handler_class(config)
    
    @classmethod
    def get_supported_types(cls):
        """Возвращает список поддерживаемых типов моделей"""
        return list(cls._handlers.keys())