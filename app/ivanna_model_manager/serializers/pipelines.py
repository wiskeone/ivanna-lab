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



class ProcessingPipelineSerializer(serializers.ModelSerializer):
    """Serializer for processing pipelines"""
    success_rate_percentage = serializers.SerializerMethodField()
    
    class Meta:
        model = ProcessingPipeline
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at')
    
    def get_success_rate_percentage(self, obj):
        """Processing success rate"""
        if obj.success_rate:
            return f"{obj.success_rate * 100:.1f}%"
        return "—"


class DocumentProcessorSerializer(serializers.ModelSerializer):
    """Serializer for document processing logs"""
    document_title = serializers.CharField(source='document.title', read_only=True)
    processing_time_seconds = serializers.SerializerMethodField()
    
    class Meta:
        model = DocumentProcessor
        fields = '__all__'
        read_only_fields = ('started_at', 'completed_at')
    
    def get_processing_time_seconds(self, obj):
        """Processing time in seconds"""
        if obj.processing_time_ms:
            return f"{obj.processing_time_ms / 1000:.1f} сек"
        return "—"
