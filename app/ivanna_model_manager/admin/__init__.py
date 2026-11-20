"""Package re-exports (split from the monolithic module)."""

from django.contrib import admin
from ..models import (
    Benchmark, BenchmarkRun, ChatMessage, ChatSession, DeletedDocumentLog,
    Document, DocumentChunk, DocumentCollection, EmbeddingModel, Experiment,
    FineTuningJob, InferenceSession, ModelCheckpoint, ModelComparison,
    ModelEvaluation, ModelMergeJob, MoEExpertJob, MoETheme, DocumentProcessor,
    ProcessingPipeline,
    RAGEvaluation, RAGSession, RetrievedChunk, SearchQuery, SearchResult,
    SearchQualityFeedback,
)

from .core import ExperimentAdmin, InferenceSessionAdmin, IsActiveFilter, ModelCheckpointAdmin, ModelCheckpointAdminForm, ModelEvaluationAdmin, ModelEvaluationInline, ModelMergeJobAdmin, ModelTypeFilter, ModelVersionAdmin, ModelVersionInline, StatusFilter
from .chat import ChatMessageAdmin, ChatMessageInline, ChatSessionAdmin, ChatSessionAdminForm, ChatSessionStatusFilter, ModelComparisonAdmin, ModelComparisonAdminForm
from .fine_tuning import FineTuningJobAdmin, FineTuningJobAdminForm
from .rag import DocumentAdmin, DocumentChunkAdmin, DocumentChunkInline, DocumentCollectionAdmin, EmbeddingModelAdmin
from .search import RAGEvaluationAdmin, RAGSessionAdmin, SearchQualityFeedbackAdmin, SearchQueryAdmin, SearchResultAdmin, SearchResultInline
from .pipelines import DocumentProcessorAdmin, ProcessingPipelineAdmin
from .moe import MoEExpertJobAdmin, MoEThemeAdmin
from .benchmarks import BenchmarkAdmin, BenchmarkRunAdmin


# Register all models via admin.site.register()
admin.site.register(Experiment, ExperimentAdmin)

admin.site.register(ModelCheckpoint, ModelCheckpointAdmin)

admin.site.register(ModelMergeJob, ModelMergeJobAdmin)

admin.site.register(InferenceSession, InferenceSessionAdmin)

admin.site.register(ModelEvaluation, ModelEvaluationAdmin)

admin.site.register(ChatSession, ChatSessionAdmin)

admin.site.register(ChatMessage, ChatMessageAdmin)

admin.site.register(ModelComparison, ModelComparisonAdmin)

admin.site.register(FineTuningJob, FineTuningJobAdmin)

admin.site.register(DocumentCollection, DocumentCollectionAdmin)

admin.site.register(Document, DocumentAdmin)

admin.site.register(DocumentChunk, DocumentChunkAdmin)

admin.site.register(EmbeddingModel, EmbeddingModelAdmin)

admin.site.register(RAGSession, RAGSessionAdmin)

admin.site.register(SearchQuery, SearchQueryAdmin)

admin.site.register(SearchResult, SearchResultAdmin)

admin.site.register(SearchQualityFeedback, SearchQualityFeedbackAdmin)

admin.site.register(RAGEvaluation, RAGEvaluationAdmin)

admin.site.register(ProcessingPipeline, ProcessingPipelineAdmin)

admin.site.register(DocumentProcessor, DocumentProcessorAdmin)

admin.site.register(RetrievedChunk)

admin.site.register(DeletedDocumentLog)
