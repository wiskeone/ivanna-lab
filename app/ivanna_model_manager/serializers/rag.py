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
    """Serializer for document chunks"""
    document_title = serializers.CharField(source='document.title', read_only=True)
    preview = serializers.CharField(read_only=True)
    
    class Meta:
        model = DocumentChunk
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at', 'usage_count', 'last_used')


class DocumentSerializer(serializers.ModelSerializer):
    """Serializer for documents"""
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
    """Serializer for embedding models"""
    total_tokens_processed_formatted = serializers.SerializerMethodField()
    
    class Meta:
        model = EmbeddingModel
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at', 'total_tokens_processed', 
                           'total_requests')
    
    def get_total_tokens_processed_formatted(self, obj):
        """Formatted display of processed tokens"""
        if obj.total_tokens_processed >= 1_000_000_000:
            return f"{obj.total_tokens_processed / 1_000_000_000:.1f}B"
        elif obj.total_tokens_processed >= 1_000_000:
            return f"{obj.total_tokens_processed / 1_000_000:.1f}M"
        elif obj.total_tokens_processed >= 1_000:
            return f"{obj.total_tokens_processed / 1_000:.1f}K"
        return str(obj.total_tokens_processed)


class SearchQuerySerializer(serializers.ModelSerializer):
    """Serializer for search queries"""
    results_count = serializers.SerializerMethodField()
    
    class Meta:
        model = SearchQuery
        fields = '__all__'
        read_only_fields = ('created_at', 'processing_time_ms')
    
    def get_results_count(self, obj):
        """Number of search results"""
        return obj.results.count() if hasattr(obj, 'results') else 0


class SearchResultSerializer(serializers.ModelSerializer):
    """Serializer for search results"""
    chunk_content = serializers.CharField(source='chunk.content', read_only=True)
    chunk_preview = serializers.CharField(source='chunk.preview', read_only=True)
    document_title = serializers.CharField(source='chunk.document.title', read_only=True)
    
    class Meta:
        model = SearchResult
        fields = '__all__'
        read_only_fields = ('created_at',)


class RetrievedChunkSerializer(serializers.ModelSerializer):
    """Serializer for retrieved chunks"""
    chunk_content = serializers.CharField(source='chunk.content', read_only=True)
    chunk_document = serializers.CharField(source='chunk.document.title', read_only=True)
    
    class Meta:
        model = RetrievedChunk
        fields = '__all__'
        read_only_fields = ('created_at',)


class RAGSessionSerializer(serializers.ModelSerializer):
    """Serializer for RAG sessions"""
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
        """Number of retrieved chunks"""
        return obj.retrieved_chunks.count() if hasattr(obj, 'retrieved_chunks') else 0


class RAGChatRequestSerializer(serializers.Serializer):
    """Serializer for the RAG chat request"""
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
    """Serializer for search quality feedback"""
    user_username = serializers.SerializerMethodField()
    
    class Meta:
        model = SearchQualityFeedback
        fields = '__all__'
        read_only_fields = ('created_at', 'resolved_at')
    
    def get_user_username(self, obj):
        """Get the username"""
        from django.contrib.auth.models import User
        try:
            user = User.objects.get(id=obj.user_id) if obj.user_id else None
            return user.username if user else None
        except User.DoesNotExist:
            return None


class RAGEvaluationSerializer(serializers.ModelSerializer):
    """Serializer for RAG system evaluation"""
    total_queries = serializers.IntegerField(read_only=True)
    improvement_display = serializers.SerializerMethodField()
    
    class Meta:
        model = RAGEvaluation
        fields = '__all__'
        read_only_fields = ('created_at', 'started_at', 'completed_at', 
                           'actual_metrics', 'detailed_results')
    
    def get_improvement_display(self, obj):
        """Improvement display"""
        if obj.improvement_percentage:
            return f"{obj.improvement_percentage:.1f}%"
        return "—"


class RerankerModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = RerankerModel
        fields = '__all__'
