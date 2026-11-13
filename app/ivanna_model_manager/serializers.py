from rest_framework import serializers
from .models import (
    RerankerModel,
    Experiment, ModelCheckpoint, ModelVersion,
    ModelMergeJob, InferenceSession, ModelEvaluation,
    ChatSession, ChatMessage, ModelComparison,
    FineTuningJob, ModelDownloadJob, DeletedModelLog,
    EvaluationDataset, ModelEvaluationTask, EvaluationResult,
    DocumentCollection, Document, DocumentChunk, EmbeddingModel,
    SearchQuery, SearchResult, RAGSession, RetrievedChunk,
    SearchQualityFeedback, RAGEvaluation, ProcessingPipeline,
    DocumentProcessor, DeletedDocumentLog, FineTuningMethod,
    FineTuningJob, FineTuningCheckpoint, FineTuningEvaluation,
    ModelConversionJob, 
    DistillationQuestionSet, DistillationChunkGroup, DistillationQuestion, 
    DistillationAnswerJob, DistillationDataset, ManualInstructionSet,
    ManualInstruction, HuggingFaceDataset, 
    
)

import logging
logger = logging.getLogger(__name__)

class ModelDownloadJobSerializer(serializers.ModelSerializer):
    """Сериализатор для задач загрузки моделей"""
    checkpoint_name = serializers.CharField(source='checkpoint.display_name', read_only=True)
    checkpoint_model_type = serializers.CharField(source='checkpoint.get_model_type_display', read_only=True)
    
    class Meta:
        model = ModelDownloadJob
        fields = [
            'id', 'checkpoint', 'checkpoint_name', 'checkpoint_model_type',
            'status', 'local_path', 'error_message', 'started_at', 
            'completed_at', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'status', 'local_path', 'error_message', 'started_at',
            'completed_at', 'created_at', 'updated_at'
        ]


class ExperimentSerializer(serializers.ModelSerializer):
    """Обновленный сериализатор для экспериментов"""
    models_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Experiment
        fields = '__all__'
    
    def get_models_count(self, obj):
        return obj.models.count()


class ModelVersionSerializer(serializers.ModelSerializer):
    """Обновленный сериализатор для версий моделей"""
    checkpoint_name = serializers.CharField(source='checkpoint.display_name', read_only=True)
    
    class Meta:
        model = ModelVersion
        fields = '__all__'
        read_only_fields = ['created_at']


class InferenceSessionSerializer(serializers.ModelSerializer):
    """Обновленный сериализатор для сессий инференса"""
    checkpoint_name = serializers.CharField(source='checkpoint.display_name', read_only=True)
    checkpoint_model_type = serializers.CharField(source='checkpoint.get_model_type_display', read_only=True)
    
    class Meta:
        model = InferenceSession
        fields = '__all__'
        read_only_fields = [
            'created_at', 'started_at', 'completed_at', 'validated_at'
        ]
    
    def get_base_model_name(self, obj):
        return obj.base_model.name if obj.base_model else None
    
    def get_merge_model_name(self, obj):
        return obj.merge_model.name if obj.merge_model else None
    
    def get_is_compatible(self, obj):
        return obj.is_compatible


class ModelEvaluationSerializer(serializers.ModelSerializer):
    """Сериализатор для оценок моделей"""
    checkpoint_name = serializers.SerializerMethodField()
    
    class Meta:
        model = ModelEvaluation
        fields = [
            'id', 'checkpoint', 'checkpoint_name', 'dataset_name',
            'dataset_split', 'evaluation_type', 'metrics', 'raw_results',
            'evaluated_at', 'evaluator_version', 'hardware_info',
            'is_valid', 'validation_notes'
        ]
        read_only_fields = ['evaluated_at']
    
    def get_checkpoint_name(self, obj):
        return obj.checkpoint.name if obj.checkpoint else None


# Простые сериализаторы для списков (меньше данных)
class ModelCheckpointListSerializer(serializers.ModelSerializer):
    """Упрощенный сериализатор для списка моделей"""
    size_gb = serializers.SerializerMethodField()
    
    class Meta:
        model = ModelCheckpoint
        fields = [
            'id', 'name', 'display_name', 'model_type', 'status',
            'parameter_count', 'quantization', 'size_gb', 'is_safe',
            'created_at'
        ]
    
    def get_size_gb(self, obj):
        return obj.size_gb


class ExperimentListSerializer(serializers.ModelSerializer):
    """Упрощенный сериализатор для списка экспериментов"""
    model_count = serializers.IntegerField(read_only=True)
    
    class Meta:
        model = Experiment
        fields = [
            'id', 'name', 'description', 'is_active', 'model_count',
            'created_at'
        ]


class ChatMessageSerializer(serializers.ModelSerializer):
    """Сериализатор для сообщений чата"""
    session_id = serializers.CharField(source='session.session_id', read_only=True)
    role_display = serializers.CharField(source='get_role_display', read_only=True)
    
    class Meta:
        model = ChatMessage
        fields = [
            'id', 'session', 'session_id', 'role', 'role_display',
            'content', 'generation_metadata', 'parent', 'created_at'
        ]
        read_only_fields = ['created_at']


class ChatSessionSerializer(serializers.ModelSerializer):
    """Сериализатор для чат-сессий"""
    checkpoint_display = serializers.CharField(source='checkpoint.display_name', read_only=True)
    model_type = serializers.CharField(source='checkpoint.model_type', read_only=True)
    
    # Новые поля для конфигурации
    session_config = serializers.SerializerMethodField()
    merged_config = serializers.SerializerMethodField()
    available_presets = serializers.SerializerMethodField()
    
    class Meta:
        model = ChatSession
        fields = [
            'id', 'session_id', 'user_id', 'checkpoint', 'checkpoint_display', 'model_type',
            'system_prompt', 'conversation_history', 'model_config', 
            'generation_config', 'total_messages', 'total_input_tokens',
            'total_output_tokens', 'is_active', 'created_at', 'updated_at',
            'last_activity', 'session_config', 'merged_config', 'available_presets'
        ]
        read_only_fields = [
            'id', 'session_id', 'created_at', 'updated_at', 'last_activity',
            'total_messages', 'total_input_tokens', 'total_output_tokens',
            'session_config', 'merged_config', 'available_presets'
        ]
    
    def get_session_config(self, obj):
        """Получает session_config из Redis"""
        from ..services.chat_service import chat_service
        try:
            session = chat_service.get_session(obj.session_id)
            if session and 'session_config' in session:
                return session['session_config']
        except Exception as e:
            logger.error(f"Error getting session config: {e}")
        return {
            'loader_overrides': {},
            'generation_config': {},
            'system_prompt': obj.system_prompt or '',
        }
    
    def get_merged_config(self, obj):
        """Вычисляет объединенную конфигурацию"""
        from ..services.chat_service import chat_service
        try:
            session = chat_service.get_session(obj.session_id)
            if session:
                return {
                    'generation': session.get('session_config', {}).get('generation_config', {}),
                    'model_info': {},
                }
        except Exception as e:
            logger.error(f"Error getting merged config: {e}")
        return {}
    
    def get_available_presets(self, obj):
        """Возвращает доступные пресеты для модели сессии"""
        if not obj.checkpoint:
            return []
        
        model_name = (obj.checkpoint.huggingface_id or '').lower()
        
        presets = [
            {'name': 'creative', 'label': 'Креативный', 'description': 'Для творческих задач'},
            {'name': 'balanced', 'label': 'Сбалансированный', 'description': 'Баланс креативности и точности'},
            {'name': 'precise', 'label': 'Точный', 'description': 'Для фактов и данных'},
        ]
        
        if 'medgemma' in model_name:
            presets.append({
                'name': 'medgemma_safe',
                'label': 'Безопасный (MedGemma)',
                'description': 'Оптимизировано для стабильной работы'
            })
        
        return presets
    
    def update(self, instance, validated_data):
        # Обрабатываем generation_config отдельно для обратной совместимости
        generation_config = validated_data.pop('generation_config', None)
        if generation_config:
            from ..services.chat_service import chat_service
            # Получаем текущую сессию из Redis
            session = chat_service.get_session(instance.session_id)
            if session:
                if 'session_config' not in session:
                    session['session_config'] = {}
                session['session_config']['generation_config'] = generation_config
                # Сохраняем обратно в Redis
                redis_key = f"chat:session:{instance.session_id}"
                chat_service.redis_manager.redis.setex(
                    redis_key, 86400, 
                    json.dumps(session, default=str)
                )
        
        # Обрабатываем system_prompt
        system_prompt = validated_data.pop('system_prompt', None)
        if system_prompt is not None:
            from ..services.chat_service import chat_service
            session = chat_service.get_session(instance.session_id)
            if session:
                if 'session_config' not in session:
                    session['session_config'] = {}
                session['session_config']['system_prompt'] = system_prompt
                redis_key = f"chat:session:{instance.session_id}"
                chat_service.redis_manager.redis.setex(
                    redis_key, 86400, 
                    json.dumps(session, default=str)
                )
        
        return super().update(instance, validated_data)


class ModelComparisonSerializer(serializers.ModelSerializer):
    """Сериализатор для сравнения моделей"""
    prompts_count = serializers.SerializerMethodField()
    models_count = serializers.SerializerMethodField()
    
    class Meta:
        model = ModelComparison
        fields = [
            'id', 'name', 'description', 'prompts', 'results', 'metrics',
            'is_complete', 'prompts_count', 'models_count', 'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']
    
    def get_prompts_count(self, obj):
        return len(obj.prompts) if obj.prompts else 0
    
    def get_models_count(self, obj):
        return len(obj.results) if obj.results else 0


class FineTuningJobSerializer(serializers.ModelSerializer):
    """Сериализатор для задач дообучения"""
    base_model_name = serializers.CharField(source='base_model.display_name', read_only=True)
    output_model_name = serializers.CharField(source='output_model.display_name', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    
    class Meta:
        model = FineTuningJob
        fields = [
            'id', 'name', 'description', 'base_model', 'base_model_name',
            'training_data_path', 'validation_data_path', 'data_config',
            'hyperparameters', 'output_model', 'output_model_name',
            'status', 'status_display', 'progress', 'current_step',
            'training_metrics', 'logs', 'error_message',
            'created_at', 'started_at', 'completed_at', 'updated_at'
        ]
        read_only_fields = [
            'status', 'progress', 'current_step', 'training_metrics',
            'logs', 'error_message', 'started_at', 'completed_at',
            'created_at', 'updated_at'
        ]

# Сериализатор для генерации текста
class TextGenerationSerializer(serializers.Serializer):
    """Сериализатор для генерации текста"""
    prompt = serializers.CharField(required=True, max_length=10000)
    max_tokens = serializers.IntegerField(default=512, min_value=1, max_value=8192)
    temperature = serializers.FloatField(default=0.7, min_value=0.0, max_value=2.0)
    top_p = serializers.FloatField(default=0.9, min_value=0.0, max_value=1.0)
    top_k = serializers.IntegerField(default=50, min_value=1, max_value=100)
    repetition_penalty = serializers.FloatField(default=1.0, min_value=0.0, max_value=2.0)
    system_prompt = serializers.CharField(required=False, allow_blank=True, max_length=5000)


# Сериализатор для загрузки модели
class ModelLoadSerializer(serializers.Serializer):
    """Сериализатор для загрузки модели в память"""
    quantization = serializers.ChoiceField(
        choices=['float32', 'float16', 'bfloat16', 'int8', 'int4'],
        default='float16'
    )
    max_memory = serializers.DictField(
        required=False,
        child=serializers.CharField()
    )


# Сериализатор для статуса памяти
class MemoryStatusSerializer(serializers.Serializer):
    """Сериализатор для статуса памяти"""
    device = serializers.CharField()
    memory = serializers.DictField()
    loaded_models = serializers.ListField(child=serializers.CharField())

class ModelMergeJobSerializer(serializers.ModelSerializer):
    """Обновленный сериализатор для задач слияния"""
    base_model_name = serializers.CharField(source='base_model.display_name', read_only=True)
    merge_model_name = serializers.CharField(source='merge_model.display_name', read_only=True)
    output_model_name = serializers.CharField(source='output_model.display_name', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    
    class Meta:
        model = ModelMergeJob
        fields = '__all__'


class ModelCheckpointSerializer(serializers.ModelSerializer):
    """Обновленный сериализатор для моделей с загрузками"""
    download_jobs = ModelDownloadJobSerializer(many=True, read_only=True)
    versions_count = serializers.SerializerMethodField()
    evaluations_count = serializers.SerializerMethodField()
    is_loaded_in_memory = serializers.SerializerMethodField()
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    model_type_display = serializers.CharField(source='get_model_type_display', read_only=True)
    quantization_display = serializers.CharField(source='get_quantization_display', read_only=True)
    download_progress = serializers.SerializerMethodField()
    
    class Meta:
        model = ModelCheckpoint
        fields = [
            'id', 'name', 'display_name', 'model_type', 'model_type_display',
            'huggingface_id', 'huggingface_revision', 'local_path', 'relative_path',
            'parameter_count', 'size_gb', 'context_length', 'vocab_size',
            'quantization', 'quantization_display', 'is_safe', 'safety_score',
            'is_public', 'license_type', 'tags', 'experiment', 'status',
            'status_display', 'loader_config', 'inference_speed', 'accuracy_score',
            'evaluation_metrics', 'is_active', 'created_at', 'updated_at',
            'download_jobs', 'versions_count', 'evaluations_count',
            'is_loaded_in_memory', 'download_progress',
        ]
        read_only_fields = [
            'created_at', 'updated_at', 'download_jobs', 'versions_count',
            'evaluations_count', 'is_loaded_in_memory', 'download_progress'
        ]
    
    def get_versions_count(self, obj):
        return obj.versions.count()
    
    def get_evaluations_count(self, obj):
        return obj.evaluations.count()
    
    def get_is_loaded_in_memory(self, obj):
        return obj.is_loaded_in_memory()
    
    def get_download_progress(self, obj):
        # Получаем последнюю задачу загрузки
        latest_job = obj.download_jobs.last()
        if latest_job:
            if latest_job.status == 'DOWNLOADING':
                return {
                    'status': 'downloading',
                    'job_id': latest_job.id
                }
            elif latest_job.status == 'COMPLETED':
                return {
                    'status': 'completed',
                    'job_id': latest_job.id,
                    'local_path': latest_job.local_path
                }
            elif latest_job.status == 'FAILED':
                return {
                    'status': 'failed',
                    'job_id': latest_job.id,
                    'error': latest_job.error_message
                }
        return {'status': 'not_started'}
    
    def validate(self, data):
        """Валидация с автоматическим заполнением loader_config"""
        # Если loader_config не предоставлен, создаем пустой словарь
        if 'loader_config' not in data:
            data['loader_config'] = {}
        
        # Определяем тип модели
        model_type = data.get('model_type', self.instance.model_type if self.instance else 'CUSTOM')
        
        # Получаем конфигурацию по умолчанию для типа модели
        from .models import ModelCheckpoint
        default_config = ModelCheckpoint.get_default_config(model_type)
        
        # Заполняем отсутствующие поля значениями по умолчанию
        for key, value in default_config.items():
            if key not in data['loader_config']:
                data['loader_config'][key] = value
        
        return data
    
    def create(self, validated_data):
        """Создание модели с автоматическим заполнением loader_config"""
        instance = super().create(validated_data)
        # Дополнительно вызываем fill_missing_loader_config на сохраненном экземпляре
        instance.fill_missing_loader_config()
        instance.save()
        return instance
    
    def update(self, instance, validated_data):
        # Обрабатываем loader_config отдельно
        loader_config = validated_data.pop('loader_config', None)
        if loader_config is not None:
            # Объединяем с существующим конфигом
            current_config = instance.loader_config or {}
            current_config.update(loader_config)
            instance.loader_config = current_config
        
        return super().update(instance, validated_data)

# ============================================================================
# Сериализаторы для системы оценки
# ============================================================================

class EvaluationDatasetSerializer(serializers.ModelSerializer):
    """Сериализатор для датасетов оценки"""
    questions_count = serializers.SerializerMethodField()
    created_by_username = serializers.CharField(source='created_by', read_only=True)
    
    class Meta:
        model = EvaluationDataset
        fields = [
            'id', 'name', 'description', 'file_path', 'questions',
            'questions_count', 'tags', 'source_type', 'total_questions',
            'created_at', 'updated_at', 'created_by', 'created_by_username'
        ]
        read_only_fields = ['created_at', 'updated_at', 'total_questions']
    
    def get_questions_count(self, obj):
        return len(obj.questions) if obj.questions else 0


class ModelEvaluationTaskSerializer(serializers.ModelSerializer):
    """Сериализатор для задач оценки моделей"""
    models = serializers.PrimaryKeyRelatedField(source='evaluation_models', many=True, read_only=True)
    models_info = serializers.SerializerMethodField()
    dataset_info = serializers.SerializerMethodField()
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    total_models = serializers.IntegerField(read_only=True)
    total_questions = serializers.IntegerField(read_only=True)
    estimated_time = serializers.SerializerMethodField()
    summary_metrics = serializers.SerializerMethodField()
    
    class Meta:
        model = ModelEvaluationTask
        fields = [
            'id', 'name', 'description', 'models', 'models_info',
            'dataset', 'dataset_info', 'generation_config',
            'status', 'status_display', 'progress', 'current_step',
            'results', 'ratings', 'summary_metrics',
            'total_models', 'total_questions', 'estimated_time',
            'logs', 'error_message',
            'created_at', 'started_at', 'completed_at', 'created_by'
        ]
        read_only_fields = [
            'status', 'progress', 'current_step', 'results', 'ratings',
            'summary_metrics', 'logs', 'error_message',
            'started_at', 'completed_at', 'created_at'
        ]
    
    def get_models_info(self, obj):
        models = []
        for model in obj.evaluation_models.all():
            models.append({
                'id': model.id,
                'name': model.display_name,
                'type': model.model_type,
                'parameters': model.parameter_count
            })
        return models
    
    def get_dataset_info(self, obj):
        if obj.dataset:
            return {
                'id': obj.dataset.id,
                'name': obj.dataset.name,
                'total_questions': obj.dataset.total_questions
            }
        return None
    
    def get_estimated_time(self, obj):
        return obj.estimated_time_minutes
    
    def get_summary_metrics(self, obj):
        return obj.summary_metrics
    
    def validate_generation_config(self, value):
        """Валидация конфигурации генерации"""
        required_fields = ['max_tokens', 'temperature']
        for field in required_fields:
            if field not in value:
                value[field] = 512 if field == 'max_tokens' else 0.7
        return value


class EvaluationResultSerializer(serializers.ModelSerializer):
    """Сериализатор для детализированных результатов оценки"""
    model_name = serializers.CharField(source='model.display_name', read_only=True)
    task_name = serializers.CharField(source='task.name', read_only=True)
    
    class Meta:
        model = EvaluationResult
        fields = [
            'id', 'task', 'task_name', 'model', 'model_name',
            'question_id', 'question_text', 'model_response',
            'response_time', 'tokens_used', 'automatic_metrics',
            'manual_rating', 'rating_comment', 'rated_by', 'rated_at',
            'retry_count', 'error_log',
            'created_at', 'updated_at'
        ]
        read_only_fields = [
            'created_at', 'updated_at', 'response_time', 'tokens_used',
            'automatic_metrics', 'retry_count', 'error_log'
        ]


class QuestionUploadSerializer(serializers.Serializer):
    """Сериализатор для загрузки вопросов из файла"""
    name = serializers.CharField(max_length=255, required=True)
    description = serializers.CharField(required=False, allow_blank=True)
    file = serializers.FileField(required=True)
    file_format = serializers.ChoiceField(
        choices=[('auto', 'Автоопределение'), ('json', 'JSON'), ('csv', 'CSV'), ('txt', 'TXT')],
        default='auto'
    )
    tags = serializers.ListField(
        child=serializers.CharField(),
        required=False,
        default=list
    )


class StartEvaluationSerializer(serializers.Serializer):
    """Сериализатор для запуска оценки"""
    name = serializers.CharField(max_length=255, required=True)
    description = serializers.CharField(required=False, allow_blank=True)
    model_ids = serializers.ListField(
        child=serializers.IntegerField(),
        required=True,
        min_length=1
    )
    dataset_id = serializers.IntegerField(required=True)
    generation_config = serializers.DictField(
        required=False,
        default={
            'max_tokens': 512,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.0
        }
    )


class RatingSerializer(serializers.Serializer):
    """Сериализатор для оценки ответа"""
    task_id = serializers.IntegerField(required=True)
    model_id = serializers.IntegerField(required=True)
    question_id = serializers.IntegerField(required=True)
    rating = serializers.IntegerField(
        required=True,
        min_value=1,
        max_value=5
    )
    comment = serializers.CharField(required=False, allow_blank=True)

class ChatConfigUpdateSerializer(serializers.Serializer):
    """Сериализатор для обновления конфигурации чат-сессии"""
    
    loader_overrides = serializers.DictField(
        required=False,
        default={},
        help_text="Переопределения loader_config модели (torch_dtype, device_map и т.д.)"
    )
    
    generation_config = serializers.DictField(
        required=False,
        default={},
        help_text="Переопределения параметров генерации"
    )
    
    system_prompt = serializers.CharField(
        required=False,
        allow_blank=True,
        default='',
        max_length=5000,
        help_text="Системный промпт для ассистента"
    )
    
    def validate_loader_overrides(self, value):
        """Валидация переопределений загрузки"""
        if not isinstance(value, dict):
            raise serializers.ValidationError("loader_overrides должен быть словарем")
        
        allowed_keys = {
            'torch_dtype', 'device_map', 'trust_remote_code', 
            'use_flash_attention_2', 'quantization_config',
            'dtype', 'max_memory', 'low_cpu_mem_usage'
        }
        
        # Фильтруем только разрешенные ключи
        filtered = {}
        for k, v in value.items():
            if k in allowed_keys:
                filtered[k] = v
        
        # Валидация torch_dtype/dtype
        dtype_value = filtered.get('torch_dtype') or filtered.get('dtype')
        if dtype_value:
            allowed_dtypes = {'float32', 'float16', 'bfloat16', 'auto'}
            if str(dtype_value).lower() not in allowed_dtypes:
                raise serializers.ValidationError(
                    f"torch_dtype/dtype должен быть одним из: {', '.join(allowed_dtypes)}"
                )
            filtered['torch_dtype'] = str(dtype_value).lower()
        
        return filtered
    
    def validate_generation_config(self, value):
        """Валидация параметров генерации"""
        if not isinstance(value, dict):
            raise serializers.ValidationError("generation_config должен быть словарем")
        
        validated = {}
        
        # Максимальное количество токенов
        if 'max_tokens' in value:
            max_tokens = value['max_tokens']
            if not isinstance(max_tokens, int) or max_tokens < 1 or max_tokens > 8192:
                raise serializers.ValidationError(
                    "max_tokens должен быть целым числом от 1 до 8192"
                )
            validated['max_tokens'] = max_tokens
        
        # Температура
        if 'temperature' in value:
            temp = value['temperature']
            try:
                temp_float = float(temp)
                if temp_float < 0.0 or temp_float > 2.0:
                    raise serializers.ValidationError(
                        "temperature должен быть числом от 0.0 до 2.0"
                    )
                validated['temperature'] = temp_float
            except (ValueError, TypeError):
                raise serializers.ValidationError(
                    "temperature должен быть числом от 0.0 до 2.0"
                )
        
        # Top-p
        if 'top_p' in value:
            top_p = value['top_p']
            try:
                top_p_float = float(top_p)
                if top_p_float < 0.0 or top_p_float > 1.0:
                    raise serializers.ValidationError(
                        "top_p должен быть числом от 0.0 до 1.0"
                    )
                validated['top_p'] = top_p_float
            except (ValueError, TypeError):
                raise serializers.ValidationError(
                    "top_p должен быть числом от 0.0 до 1.0"
                )
        
        # Top-k
        if 'top_k' in value:
            top_k = value['top_k']
            if not isinstance(top_k, int) or top_k < 0 or top_k > 1000:
                raise serializers.ValidationError(
                    "top_k должен быть целым числом от 0 до 1000"
                )
            validated['top_k'] = top_k
        
        # Repetition penalty
        if 'repetition_penalty' in value:
            rep_penalty = value['repetition_penalty']
            try:
                rep_float = float(rep_penalty)
                if rep_float < 1.0 or rep_float > 2.0:
                    raise serializers.ValidationError(
                        "repetition_penalty должен быть числом от 1.0 до 2.0"
                    )
                validated['repetition_penalty'] = rep_float
            except (ValueError, TypeError):
                raise serializers.ValidationError(
                    "repetition_penalty должен быть числом от 1.0 до 2.0"
                )
        
        return validated
    
    def validate(self, data):
        """Общая валидация"""
        # Если оба пустые, ничего не делаем
        if not data.get('loader_overrides') and not data.get('generation_config') and not data.get('system_prompt'):
            raise serializers.ValidationError("Нет данных для обновления конфигурации")
        
        return data


class ChatSessionConfigSerializer(serializers.Serializer):
    """Сериализатор для получения полной конфигурации сессии"""
    
    session_id = serializers.CharField(read_only=True)
    model_id = serializers.IntegerField(read_only=True)
    
    model_info = serializers.DictField(read_only=True)
    session_config = serializers.DictField(read_only=True)
    merged_generation_config = serializers.DictField(read_only=True)
    
    conversation_history_length = serializers.IntegerField(read_only=True)
    last_activity = serializers.CharField(read_only=True)


class ChatSessionCreateSerializer(serializers.Serializer):
    """Сериализатор для создания чат-сессии с конфигурацией"""
    
    model_id = serializers.IntegerField(required=True)
    system_prompt = serializers.CharField(required=False, allow_blank=True, max_length=5000)
    
    loader_overrides = serializers.DictField(
        required=False,
        default={},
        help_text="Начальные переопределения loader_config"
    )
    
    generation_config = serializers.DictField(
        required=False,
        default={},
        help_text="Начальные переопределения generation_config"
    )



####################################################################################
# RAG
####################################################################################

class DocumentCollectionSerializer(serializers.ModelSerializer):
    total_documents = serializers.IntegerField(read_only=True)
    total_chunks = serializers.IntegerField(read_only=True)
    total_vectors = serializers.IntegerField(read_only=True)
    total_size_mb = serializers.FloatField(read_only=True)
    owner_username = serializers.SerializerMethodField()

    class Meta:
        model = DocumentCollection
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at', 'total_documents', 
                           'total_chunks', 'total_size_mb', 'total_vectors')

    def get_owner_username(self, obj):
        from django.contrib.auth.models import User
        if not obj.owner_id:
            return None
        try:
            user = User.objects.get(id=int(obj.owner_id))
            return user.username
        except (ValueError, User.DoesNotExist):
            return obj.owner_id

class DocumentChunkSerializer(serializers.ModelSerializer):
    """Сериализатор для чанков документов"""
    document_title = serializers.CharField(source='document.title', read_only=True)
    preview = serializers.CharField(read_only=True)
    
    class Meta:
        model = DocumentChunk
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at', 'usage_count', 'last_used')

class DocumentSerializer(serializers.ModelSerializer):
    """Сериализатор для документов"""
    collection_name = serializers.CharField(source='collection.name', read_only=True)
    chunk_count = serializers.IntegerField(read_only=True)
    has_embeddings = serializers.BooleanField(read_only=True)
    chunks = DocumentChunkSerializer(many=True, read_only=True)
    
    class Meta:
        model = Document
        fields = '__all__'
        read_only_fields = ('uploaded_at', 'processed_at', 'indexed_at', 
                           'content_hash', 'chunk_count')

class EmbeddingModelSerializer(serializers.ModelSerializer):
    """Сериализатор для моделей эмбеддингов"""
    total_tokens_processed_formatted = serializers.SerializerMethodField()
    
    class Meta:
        model = EmbeddingModel
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at', 'total_tokens_processed', 
                           'total_requests')
    
    def get_total_tokens_processed_formatted(self, obj):
        """Форматированное отображение обработанных токенов"""
        if obj.total_tokens_processed >= 1_000_000_000:
            return f"{obj.total_tokens_processed / 1_000_000_000:.1f}B"
        elif obj.total_tokens_processed >= 1_000_000:
            return f"{obj.total_tokens_processed / 1_000_000:.1f}M"
        elif obj.total_tokens_processed >= 1_000:
            return f"{obj.total_tokens_processed / 1_000:.1f}K"
        return str(obj.total_tokens_processed)

class SearchQuerySerializer(serializers.ModelSerializer):
    """Сериализатор для поисковых запросов"""
    results_count = serializers.SerializerMethodField()
    
    class Meta:
        model = SearchQuery
        fields = '__all__'
        read_only_fields = ('created_at', 'processing_time_ms')
    
    def get_results_count(self, obj):
        """Количество результатов поиска"""
        return obj.results.count() if hasattr(obj, 'results') else 0

class SearchResultSerializer(serializers.ModelSerializer):
    """Сериализатор для результатов поиска"""
    chunk_content = serializers.CharField(source='chunk.content', read_only=True)
    chunk_preview = serializers.CharField(source='chunk.preview', read_only=True)
    document_title = serializers.CharField(source='chunk.document.title', read_only=True)
    
    class Meta:
        model = SearchResult
        fields = '__all__'
        read_only_fields = ('created_at',)

class RetrievedChunkSerializer(serializers.ModelSerializer):
    """Сериализатор для извлеченных чанков"""
    chunk_content = serializers.CharField(source='chunk.content', read_only=True)
    chunk_document = serializers.CharField(source='chunk.document.title', read_only=True)
    
    class Meta:
        model = RetrievedChunk
        fields = '__all__'
        read_only_fields = ('created_at',)

class RAGSessionSerializer(serializers.ModelSerializer):
    """Сериализатор для RAG сессий"""
    llm_model_name = serializers.CharField(source='llm_model.display_name', read_only=True)
    embedding_model_name = serializers.CharField(source='embedding_model.name', read_only=True)
    total_tokens = serializers.IntegerField(read_only=True)
    retrieved_chunks_count = serializers.SerializerMethodField()
    
    class Meta:
        model = RAGSession
        fields = '__all__'
        read_only_fields = ('created_at', 'started_at', 'completed_at', 
                           'feedback_at', 'total_time_ms', 'search_time_ms', 
                           'generation_time_ms', 'input_tokens', 'output_tokens')
    
    def get_retrieved_chunks_count(self, obj):
        """Количество извлеченных чанков"""
        return obj.retrieved_chunks.count() if hasattr(obj, 'retrieved_chunks') else 0

class RAGChatRequestSerializer(serializers.Serializer):
    """Сериализатор для запроса RAG чата"""
    query = serializers.CharField(required=True, max_length=1000)
    collection_id = serializers.IntegerField(required=False)
    document_ids = serializers.ListField(
        child=serializers.IntegerField(),
        required=False
    )
    system_prompt = serializers.CharField(required=False, max_length=2000)
    llm_model_id = serializers.IntegerField(required=False)
    embedding_model_id = serializers.IntegerField(required=False)
    search_config = serializers.DictField(required=False, default={
        'top_k': 5,
        'score_threshold': 0.7,
        'search_type': 'hybrid'
    })
    generation_config = serializers.DictField(required=False, default={
        'max_tokens': 1000,
        'temperature': 0.7,
        'top_p': 0.9
    })

class SearchQualityFeedbackSerializer(serializers.ModelSerializer):
    """Сериализатор для обратной связи по качеству поиска"""
    user_username = serializers.SerializerMethodField()
    
    class Meta:
        model = SearchQualityFeedback
        fields = '__all__'
        read_only_fields = ('created_at', 'resolved_at')
    
    def get_user_username(self, obj):
        """Получить имя пользователя"""
        from django.contrib.auth.models import User
        try:
            user = User.objects.get(id=obj.user_id) if obj.user_id else None
            return user.username if user else None
        except User.DoesNotExist:
            return None

class RAGEvaluationSerializer(serializers.ModelSerializer):
    """Сериализатор для оценки RAG системы"""
    total_queries = serializers.IntegerField(read_only=True)
    improvement_display = serializers.SerializerMethodField()
    
    class Meta:
        model = RAGEvaluation
        fields = '__all__'
        read_only_fields = ('created_at', 'started_at', 'completed_at', 
                           'actual_metrics', 'detailed_results')
    
    def get_improvement_display(self, obj):
        """Отображение улучшения"""
        if obj.improvement_percentage:
            return f"{obj.improvement_percentage:.1f}%"
        return "—"

class ProcessingPipelineSerializer(serializers.ModelSerializer):
    """Сериализатор для пайплайнов обработки"""
    success_rate_percentage = serializers.SerializerMethodField()
    
    class Meta:
        model = ProcessingPipeline
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at')
    
    def get_success_rate_percentage(self, obj):
        """Процент успешной обработки"""
        if obj.success_rate:
            return f"{obj.success_rate * 100:.1f}%"
        return "—"

class DocumentProcessorSerializer(serializers.ModelSerializer):
    """Сериализатор для логов обработки документов"""
    document_title = serializers.CharField(source='document.title', read_only=True)
    processing_time_seconds = serializers.SerializerMethodField()
    
    class Meta:
        model = DocumentProcessor
        fields = '__all__'
        read_only_fields = ('started_at', 'completed_at')
    
    def get_processing_time_seconds(self, obj):
        """Время обработки в секундах"""
        if obj.processing_time_ms:
            return f"{obj.processing_time_ms / 1000:.1f} сек"
        return "—"

class FineTuningJobSerializer(serializers.ModelSerializer):
    """Сериализатор для задач fine-tuning"""
    
    base_model_name = serializers.CharField(
        source='base_model.display_name',
        read_only=True
    )
    output_model_name = serializers.CharField(
        source='output_model.display_name',
        read_only=True
    )
    status_display = serializers.CharField(
        source='get_status_display',
        read_only=True
    )
    method_display = serializers.CharField(
        source='get_method_display',
        read_only=True
    )
    
    class Meta:
        model = FineTuningJob
        fields = '__all__'
        read_only_fields = [
            'status', 'progress', 'current_step', 'logs',
            'error_message', 'training_metrics', 'evaluation_results',
            'celery_task_id', 'created_at', 'started_at', 'completed_at'
        ]


class FineTuningCheckpointSerializer(serializers.ModelSerializer):
    class Meta:
        model = FineTuningCheckpoint
        fields = '__all__'


class FineTuningEvaluationSerializer(serializers.ModelSerializer):
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)
    
    class Meta:
        model = FineTuningEvaluation
        fields = '__all__'

class ModelConversionJobSerializer(serializers.ModelSerializer):
    source_model_name = serializers.CharField(source='source_model.display_name', read_only=True)
    output_model_name = serializers.CharField(source='output_model.display_name', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)

    class Meta:
        model = ModelConversionJob
        fields = '__all__'
        read_only_fields = ['status', 'progress', 'current_step', 'logs', 'error_message',
                            'error_details', 'celery_task_id', 'created_at', 'started_at',
                            'completed_at']


# ============================================================================
# СЕРИАЛИЗАТОРЫ ДЛЯ ДИСТИЛЛЯЦИИ
# ============================================================================

class DistillationChunkGroupSerializer(serializers.ModelSerializer):
    class Meta:
        model = DistillationChunkGroup
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at', 'logs')


class DistillationQuestionSetSerializer(serializers.ModelSerializer):
    groups = DistillationChunkGroupSerializer(many=True, read_only=True)
    group_count = serializers.IntegerField(source='groups.count', read_only=True)

    class Meta:
        model = DistillationQuestionSet
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at', 'status', 'progress', 'current_step', 'logs')


class DistillationQuestionSerializer(serializers.ModelSerializer):
    text = serializers.CharField(source='question_text', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)

    class Meta:
        model = DistillationQuestion
        fields = ['id', 'text', 'answer_text', 'status', 'status_display', 'created_at', 'updated_at']


class DistillationAnswerJobSerializer(serializers.ModelSerializer):
    question_set_name = serializers.CharField(source='question_set.name', read_only=True)

    class Meta:
        model = DistillationAnswerJob
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at', 'status', 'progress', 'current_step', 'processed_count', 'total_count', 'logs')

class DistillationDatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = DistillationDataset
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at')


# ============================================================================
# СЕРИАЛИЗАТОРЫ ДЛЯ ИНСТРУКЦИЙ
# ============================================================================
class ManualInstructionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ManualInstruction
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at')


class ManualInstructionSetSerializer(serializers.ModelSerializer):
    instructions = ManualInstructionSerializer(many=True, read_only=True)
    instruction_count = serializers.IntegerField(source='instructions.count', read_only=True)

    class Meta:
        model = ManualInstructionSet
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at')


# ============================================================================
# СЕРИАЛИЗАТОР ДЛЯ ЗАГРУЗКИ ДАТАСЕТОВ С ХФ
# ============================================================================
class HuggingFaceDatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = HuggingFaceDataset
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at')

class RerankerModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = RerankerModel
        fields = '__all__'
