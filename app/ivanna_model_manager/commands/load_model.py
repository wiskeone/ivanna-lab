from django.core.management.base import BaseCommand
from django.conf import settings
from ivanna_model_manager.models import ModelCheckpoint
from ivanna_model_manager.services.redis_model_service import redis_model_manager
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Load a model into memory'
    
    def add_arguments(self, parser):
        parser.add_argument('model_id', type=str, help='Model checkpoint ID or name')
        parser.add_argument('--quantization', type=str, default='float16', 
                          help='Quantization type (float16, int8, int4)')
    
    def handle(self, *args, **options):
        model_id = options['model_id']
        quantization = options['quantization']
        
        try:
            # Пытаемся найти модель в базе данных
            if model_id.isdigit():
                checkpoint = ModelCheckpoint.objects.get(id=int(model_id))
            else:
                checkpoint = ModelCheckpoint.objects.get(name=model_id)
            
            self.stdout.write(f"Loading model: {checkpoint.display_name}")
            
            # Конфигурация модели
            model_config = {
                "model_id": checkpoint.huggingface_id or checkpoint.name,
                "local_path": checkpoint.local_path,
                "quantization": quantization,
                "parameter_count": checkpoint.parameter_count,
                "context_length": checkpoint.context_length,
                "model_type": checkpoint.model_type
            }
            
            # Загружаем модель
            model_info = model_manager.load_model(model_config)
            
            if model_info:
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Model loaded successfully on {model_info.device}"
                    )
                )
                self.stdout.write(f"Model ID: {model_info.model_id}")
                self.stdout.write(f"Parameters: {model_info.parameters:,}")
                self.stdout.write(f"Context length: {model_info.context_length}")
            else:
                self.stdout.write(self.style.ERROR("Failed to load model"))
                
        except ModelCheckpoint.DoesNotExist:
            # Если модели нет в базе, загружаем напрямую из Hugging Face
            self.stdout.write(f"Model not in database, loading from Hugging Face: {model_id}")
            
            model_config = {
                "model_id": model_id,
                "quantization": quantization,
                "parameter_count": 0,  # Неизвестно
                "context_length": 2048,
                "model_type": "unknown"
            }
            
            try:
                model_info = model_manager.load_model(model_config)
                if model_info:
                    self.stdout.write(
                        self.style.SUCCESS(f"Model loaded from Hugging Face")
                    )
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error loading model: {str(e)}"))
        
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error: {str(e)}"))