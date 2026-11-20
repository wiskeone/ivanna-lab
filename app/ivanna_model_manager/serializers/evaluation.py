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



class EvaluationDatasetSerializer(serializers.ModelSerializer):
    """Serializer for evaluation datasets"""
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
    """Serializer for model evaluation tasks"""
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
        """Validate the generation config"""
        required_fields = ['max_tokens', 'temperature']
        for field in required_fields:
            if field not in value:
                value[field] = 512 if field == 'max_tokens' else 0.7
        return value


class EvaluationResultSerializer(serializers.ModelSerializer):
    """Serializer for detailed evaluation results"""
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
    """Serializer for uploading questions from a file"""
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
    """Serializer for starting an evaluation"""
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
    """Serializer for rating a response"""
    task_id = serializers.IntegerField(required=True)
    model_id = serializers.IntegerField(required=True)
    question_id = serializers.IntegerField(required=True)
    rating = serializers.IntegerField(
        required=True,
        min_value=1,
        max_value=5
    )
    comment = serializers.CharField(required=False, allow_blank=True)
