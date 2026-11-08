from django.apps import AppConfig

class IvannaModelManagerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ivanna_model_manager'
    verbose_name = 'Ivanna Model Manager'
    
    def ready(self):
        # Импортируем сигналы, если они есть
        try:
            import ivanna_model_manager.signals
        except ImportError:
            pass