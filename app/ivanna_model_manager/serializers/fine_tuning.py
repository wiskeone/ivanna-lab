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
