from rest_framework import serializers
from ..models import (
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
        from ..models import ModelCheckpoint
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
