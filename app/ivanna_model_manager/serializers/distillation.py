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
