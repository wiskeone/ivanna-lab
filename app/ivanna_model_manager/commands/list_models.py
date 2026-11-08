from django.core.management.base import BaseCommand
from ivanna_model_manager.services.redis_model_service import redis_model_manager
from ivanna_model_manager.models import ModelCheckpoint

class Command(BaseCommand):
    help = 'List all loaded models'
    
    def handle(self, *args, **options):
        # Загруженные в память
        loaded = model_manager.list_loaded_models()
        
        self.stdout.write("=" * 60)
        self.stdout.write("LOADED MODELS IN MEMORY:")
        self.stdout.write("=" * 60)
        
        for model_id in loaded:
            info = model_manager.get_model_info(model_id)
            self.stdout.write(f"• {model_id}")
            if info:
                self.stdout.write(f"  Device: {info['device']}")
                self.stdout.write(f"  Parameters: {info.get('parameters', 'unknown'):,}")
                self.stdout.write(f"  Context: {info.get('context_length', 'unknown')} tokens")
        
        # Модели в базе данных
        db_models = ModelCheckpoint.objects.filter(status='LOADED')
        
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write("MODELS IN DATABASE:")
        self.stdout.write("=" * 60)
        
        for model in db_models:
            memory_status = "✓ In memory" if model.is_loaded_in_memory() else "✗ On disk"
            self.stdout.write(f"• {model.display_name} ({model.name})")
            self.stdout.write(f"  Status: {model.get_status_display()} | {memory_status}")
            self.stdout.write(f"  Type: {model.get_model_type_display()}")
            self.stdout.write(f"  Parameters: {model.parameter_count:,}")
            self.stdout.write(f"  Path: {model.local_path or 'Not downloaded'}")