from django.urls import path, include
from django.views.generic import RedirectView
from rest_framework.routers import DefaultRouter
from django.conf import settings
from django.contrib.auth import views as auth_views

from . import views
from . import api_views
from . import serializers

# API Router
router = DefaultRouter()
router.register(r'experiments', api_views.ExperimentViewSet, basename='experiment')
router.register(r'models', api_views.ModelCheckpointViewSet, basename='model')
router.register(r'merge-jobs', api_views.ModelMergeJobViewSet, basename='merge-job')
router.register(r'inference-sessions', api_views.InferenceSessionViewSet, basename='inference-session')
router.register(r'evaluations', api_views.ModelEvaluationViewSet, basename='evaluation')
router.register(r'download-jobs', api_views.ModelDownloadJobViewSet, basename='download-job')
router.register(r'chat-sessions', api_views.ChatSessionViewSet, basename='chat-session')
router.register(r'rag/collections', api_views.DocumentCollectionViewSet, basename='rag-collection')
router.register(r'rag/documents', api_views.DocumentViewSet, basename='rag-document')
router.register(r'rag/sessions', api_views.RAGSessionViewSet, basename='rag-session')
router.register(r'fine-tuning-jobs', api_views.FineTuningJobViewSet, basename='fine-tuning-job')
router.register(r'conversion-jobs', api_views.ModelConversionJobViewSet, basename='conversion-job')
router.register(r'embedding-models', api_views.EmbeddingModelViewSet, basename='embedding-model')
router.register(r'distillation/questionsets', api_views.DistillationQuestionSetViewSet, basename='distillation-questionset')
router.register(r'distillation/groups', api_views.DistillationChunkGroupViewSet, basename='distillation-group')
router.register(r'distillation/questions', api_views.DistillationQuestionViewSet, basename='distillation-question')
router.register(r'distillation/answerjobs', api_views.DistillationAnswerJobViewSet, basename='distillation-answerjob')
router.register(r'distillation/datasets', api_views.DistillationDatasetViewSet, basename='distillation-dataset')
router.register(r'manual-instruction-sets', api_views.ManualInstructionSetViewSet, basename='manual-instruction-set')
router.register(r'manual-instructions', api_views.ManualInstructionViewSet, basename='manual-instruction')

urlpatterns = [
    # Аутентификация
    path('accounts/login/', auth_views.LoginView.as_view(template_name='ivanna_model_manager/login.html'), name='login'),
    path('accounts/logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),
    path('accounts/', include('django.contrib.auth.urls')),

    # Профиль
    path('profile/', views.profile_view, name='profile'),

    # Главная
    path('', views.DashboardView.as_view(), name='dashboard'),

    # API через router
    path('api/', include(router.urls)),

    # Отдельные API-эндпоинты (health, gpu, system, evaluation, huggingface, rag search, statistics)
    path('api/health/', api_views.api_health_check, name='api_health'),
    path('api/gpu-status/', api_views.gpu_status, name='gpu_status'),
    path('api/system-metrics/', api_views.system_metrics, name='system_metrics'),
    path('api/evaluation/datasets/', api_views.evaluation_datasets_api, name='api_evaluation_datasets'),
    path('api/evaluation/datasets/upload/', api_views.upload_dataset_api, name='api_upload_dataset'),
    path('api/evaluation/tasks/start/', api_views.start_evaluation_api, name='api_start_evaluation'),
    path('api/evaluation/tasks/<int:task_id>/status/', api_views.evaluation_task_status_api, name='api_evaluation_task_status'),
    path('api/evaluation/tasks/<int:task_id>/results/', api_views.evaluation_results_api, name='api_evaluation_results'),
    path('api/evaluation/tasks/<int:task_id>/compare/', api_views.compare_models_api, name='api_compare_models'),
    path('api/evaluation/rate/', api_views.rate_response_api, name='api_rate_response'),
    path('api/huggingface/', api_views.HuggingFaceAPI.as_view(), name='huggingface_api'),
    # v5.0: пресеты параметров генерации
    # v5.1: пауза/резюме fine-tuning
    path('api/fine-tuning/<int:job_id>/pause/', api_views.api_fine_tuning_pause, name='api_ft_pause'),
    path('api/fine-tuning/<int:job_id>/resume/', api_views.api_fine_tuning_resume, name='api_ft_resume'),
    path('api/generation-presets/', api_views.api_generation_presets_list, name='api_generation_presets'),
    path('api/generation-presets/create/', api_views.api_generation_preset_create, name='api_generation_preset_create'),
    path('api/generation-presets/<int:preset_id>/delete/', api_views.api_generation_preset_delete, name='api_generation_preset_delete'),
    path('api/chat/sessions/<str:session_id>/apply_preset/', api_views.api_chat_apply_preset, name='api_chat_apply_preset'),
    path('api/evaluation-stats/', api_views.api_evaluation_stats, name='api_evaluation_stats'),
    path('api/statistics/', api_views.api_statistics_view, name='api_statistics'),
    path('api/rag-statistics/', api_views.api_rag_statistics, name='api_rag_statistics'),
    path('api/rag/search/', api_views.rag_search, name='rag_search'),
    path('api/tasks/<str:task_id>/result/', api_views.task_result, name='task_result'),

    # HTML-вьюхи для моделей
    path('models/', views.ModelListView.as_view(), name='model_list'),
    path('models/create/', views.UploadModelView.as_view(), name='model_create'),
    path('models/local-create/', views.LocalModelCreateView.as_view(), name='model_local_create'),
    path('models/<int:pk>/', views.ModelDetailView.as_view(), name='model_detail'),
    path('models/<int:pk>/edit/', views.ModelUpdateView.as_view(), name='model_edit'),
    path('models/<int:pk>/config/', views.model_config_view, name='model_config'),
    path('models/<int:pk>/download/', views.download_model_view, name='model_download'),
    path('models/<int:pk>/test/', views.InferenceTestView.as_view(), name='model_test'),
    path('models/<int:pk>/delete/', views.model_delete_view, name='model_delete'),
    path('models/<int:model_pk>/versions/<int:version_pk>/restore/', views.restore_model_version_view, name='model_version_restore'),
    path('models/<int:pk>/create-version/', views.create_model_version_view, name='model_create_version'),
    path('models/<int:pk>/start-merge/', views.start_merge_view, name='model_start_merge'),

    # Эксперименты
    path('experiments/', views.ExperimentListView.as_view(), name='experiment_list'),
    path('experiments/create/', views.ExperimentCreateView.as_view(), name='experiment_create'),
    path('experiments/<int:pk>/', views.ExperimentDetailView.as_view(), name='experiment_detail'),
    path('experiments/<int:pk>/edit/', views.ExperimentUpdateView.as_view(), name='experiment_edit'),
    path('experiments/<int:pk>/deactivate/', views.experiment_deactivate_view, name='experiment_deactivate'),
    path('experiments/<int:pk>/activate/', views.experiment_activate_view, name='experiment_activate'),
    path('experiments/<int:pk>/create-checkpoint/', views.create_checkpoint_from_experiment, name='experiment_create_checkpoint'),

    # Чат
    path('chat/sessions/', views.chat_sessions_management, name='chat_sessions_management'),
    path('chat/create/', views.create_chat_session, name='chat_create'),
    path('chat/<str:session_id>/', views.chat_view, name='chat_session'),
    path('chat/', views.chat_view, name='chat_view'),
    path('chat/sessions/<str:session_id>/send/', views.send_message, name='chat_send_message'),
    path('chat/sessions/<str:session_id>/clear/', views.clear_chat_session, name='clear_chat_session'),
    path('chat/sessions/<str:session_id>/delete/', views.delete_chat_session, name='delete_chat_session'),
    path('chat/sessions/<str:session_id>/rename/', views.rename_chat_session, name='rename_chat_session'),
    path('chat/sessions/<str:session_id>/change-model/', views.change_chat_session_model, name='change_chat_session_model'),
    path('chat/sessions/<str:session_id>/export/', views.export_chat_session, name='export_chat_session'),
    path('api/models/<int:model_id>/force-local-load/', api_views.force_local_load_model, name='force_local_load_model'),
    path('api/chat/sessions/<str:session_id>/preview/', api_views.api_chat_session_preview, name='api_chat_session_preview'),
    
    # Слияние моделей
    path('merge/', views.ModelMergeView.as_view(), name='merge_create'),

    # Сравнение моделей
    path('comparisons/', views.ComparisonView.as_view(), name='comparison_list'),
    path('comparisons/create/', views.ComparisonCreateView.as_view(), name='comparison_create'),
    path('comparisons/<int:pk>/', views.ComparisonDetailView.as_view(), name='comparison_detail'),

    # Оценка (Evaluation)
    path('evaluation/', views.evaluation_dashboard, name='evaluation_dashboard'),
    path('evaluation/datasets/', views.EvaluationDatasetListView.as_view(), name='evaluation_dataset_list'),
    path('evaluation/datasets/create/', views.create_dataset_manual_view, name='evaluation_dataset_create'),
    path('evaluation/datasets/upload/', views.upload_dataset_view, name='evaluation_dataset_upload'),
    path('evaluation/datasets/<int:pk>/', views.EvaluationDatasetDetailView.as_view(), name='evaluation_dataset_detail'),
    path('evaluation/datasets/<int:pk>/edit/', views.edit_dataset_questions_view, name='evaluation_dataset_edit'),
    path('evaluation/tasks/', views.evaluation_task_list_view, name='evaluation_task_list'),
    path('evaluation/tasks/create/', views.create_evaluation_task_view, name='evaluation_task_create'),
    path('evaluation/tasks/<int:pk>/', views.evaluation_task_detail_view, name='evaluation_task_detail'),
    path('evaluation/tasks/<int:task_id>/progress/', views.evaluation_task_progress_view, name='evaluation_task_progress'),
    path('evaluation/tasks/<int:task_id>/compare/', views.compare_models_view, name='evaluation_task_compare'),
    path('evaluation/rate/', views.rate_response_view, name='evaluation_rate'),
    path('api/evaluation/tasks/<int:task_id>/model/<int:model_id>/results/', views.get_model_results, name='api_evaluation_model_results'),
    path('evaluation/tasks/<int:task_id>/restart/', views.restart_evaluation_task, name='evaluation_task_restart'),
    path('evaluation/tasks/<int:task_id>/delete/', views.delete_evaluation_task, name='evaluation_task_delete'),
    path('api/evaluation/datasets/<int:dataset_id>/compare-tasks/', api_views.compare_tasks_api, name='api_compare_tasks'),
    path('evaluation/tasks/<int:task_id>/force-reset/', views.force_reset_stuck_task, name='evaluation_task_force_reset'),
    path('api/evaluation/datasets/<int:dataset_id>/questions/', api_views.evaluation_dataset_questions, name='api_evaluation_dataset_questions'),
    path('api/evaluation/datasets/<int:dataset_id>/questions/<int:question_id>/', api_views.evaluation_dataset_question_update, name='api_evaluation_dataset_question_update'),
    path('evaluation/datasets/<int:pk>/export/', views.export_evaluation_dataset, name='evaluation_dataset_export'),
    path('api/evaluation/datasets/<int:pk>/import/', api_views.evaluation_dataset_import, name='api_evaluation_dataset_import'),

    # RAG (HTML)
    path('rag/collections/', views.rag_collection_list, name='rag_collection_list'),
    path('rag/collections/create/', views.rag_collection_create, name='rag_collection_create'),
    path('rag/collections/<int:pk>/', views.rag_collection_detail, name='rag_collection_detail'),
    path('rag/collections/<int:pk>/delete/', views.rag_collection_delete, name='rag_collection_delete'),
    path('rag/collections/<int:pk>/create-embeddings/', views.create_collection_embeddings, name='rag_create_embeddings'),
    path('rag/chat/', views.rag_chat_view, name='rag_chat'),
    path('rag/documents/upload/', views.rag_upload_documents, name='rag_upload_documents'),
    path('rag/documents/', views.document_list, name='document_list'),
    path('rag/documents/collection/<int:collection_id>/', views.document_list, name='document_list_collection'),
    path('documents/<int:pk>/', views.document_detail, name='document_detail'),
    path('documents/<int:pk>/edit/', views.document_edit, name='document_edit'),
    path('documents/<int:pk>/delete/', views.document_delete, name='document_delete'),
    path('documents/<int:pk>/process/', views.document_process, name='document_process'),
    path('documents/<int:pk>/reindex/', views.document_reindex, name='document_reindex'),
    path('documents/<int:pk>/download/', views.document_download, name='document_download'),
    path('rag/documents/bulk-action/', views.document_bulk_action, name='document_bulk_action'),
    path('document-chunks/<int:pk>/edit/', views.document_chunk_edit, name='document_chunk_edit'),
    path('documents/<int:document_id>/status/', views.ajax_document_status, name='ajax_document_status'),
    path('collections/<int:pk>/build-index/', views.build_collection_index, name='build_collection_index'),
    path('collections/<int:pk>/update-index/', views.update_collection_index, name='update_collection_index'),
    path('collections/<int:pk>/index-stats/', views.get_index_stats, name='get_index_stats'),
    path('collections/<int:collection_id>/check-files/', views.check_index_files, name='check_index_files'),
    path('documents/<int:pk>/reload/', views.document_reload, name='document_reload'),

    # API для чанков
    path('api/document-chunks/<int:chunk_id>/', api_views.api_document_chunk_detail, name='api_document_chunk_detail'),

    # FINETUNING
    path('fine-tuning/', views.fine_tuning_dashboard, name='fine_tuning_dashboard'),
    path('fine-tuning/create/', views.fine_tuning_create, name='fine_tuning_create'),
    path('fine-tuning/jobs/', views.fine_tuning_jobs, name='fine_tuning_jobs'),
    path('fine-tuning/<int:pk>/', views.fine_tuning_detail, name='fine_tuning_detail'),

    # Fine-tuning templates
    path('fine-tuning/templates/', views.fine_tuning_templates, name='fine_tuning_templates'),
    path('fine-tuning/templates/create/', views.fine_tuning_template_create, name='fine_tuning_template_create'),
    path('fine-tuning/templates/<int:pk>/', views.fine_tuning_template_detail, name='fine_tuning_template_detail'),
    path('fine-tuning/templates/<int:pk>/apply/', views.fine_tuning_template_apply, name='fine_tuning_template_apply'),
    path('fine-tuning/<int:pk>/logs/', views.fine_tuning_logs, name='fine_tuning_logs'),
    path('fine-tuning/<int:pk>/progress/', views.fine_tuning_progress, name='fine_tuning_progress'),
    path('fine-tuning/<int:pk>/metrics/', views.fine_tuning_metrics, name='fine_tuning_metrics'),

    # Конвертация моделей
    path('conversion/', views.conversion_dashboard, name='conversion_dashboard'),
    path('conversion/create/', views.conversion_create, name='conversion_create'),
    path('conversion/<int:pk>/', views.conversion_detail, name='conversion_detail'),
    path('conversion/<int:pk>/logs/', views.conversion_logs, name='conversion_logs'),
    path('conversion/<int:pk>/progress/', views.conversion_progress, name='conversion_progress'),

    # AJAX для RAG-чата
    path('rag/ajax/chat/', views.ajax_rag_chat, name='ajax_rag_chat'),

    # Страница расширенного поиска RAG
    path('rag/search/', views.rag_search_page, name='rag_search_page'),
    path('rag/search/<int:collection_id>/', views.rag_search_page, name='rag_search_page_collection'),
    path('api/rag/search-suggestions/', api_views.api_rag_search_suggestions, name='api_rag_search_suggestions'),
    path('api/rag/collection-stats/<int:collection_id>/', api_views.api_rag_collection_stats, name='api_rag_collection_stats'),
    path('api/rag/search-history/', api_views.api_rag_search_history, name='api_rag_search_history'),
    path('api/rag/export-search-results/', api_views.api_rag_export_search_results, name='api_rag_export_search_results'),
    path('api/documents/<int:document_id>/preview/', api_views.api_document_preview, name='api_document_preview'),

    # AJAX эндпоинт для поиска
    path('rag/ajax/advanced-search/', views.ajax_rag_advanced_search, name='ajax_rag_advanced_search'),

    # Системные метрики и статусы
    path('metrics/', views.SystemMetricsView.as_view(), name='system_metrics_view'),
    path('gpu-status/', views.gpu_status_view, name='gpu_status_view'),
    path('health/', views.health_check_view, name='health_check'),
    path('api/tasks/<str:task_id>/status/', views.task_status, name='task_status'),

    # Статические страницы
    path('about/', views.AboutView.as_view(), name='about'),
    path('docs/', views.DocumentationView.as_view(), name='docs'),

    # API для статистики (HTML/JSON) – оставлено для совместимости
    path('api/supported-types/', views.SupportedModelTypesView.as_view(), name='supported_types'),

    # Перенаправления для старых URL
    path('upload/', RedirectView.as_view(pattern_name='model_create', permanent=True)),
    path('start-merge/', RedirectView.as_view(pattern_name='merge_create', permanent=True)),

    # Новые API для чата
    path('api/chat/sessions/<str:session_id>/send_message/', api_views.api_chat_send_message, name='api_chat_send_message'),
    path('api/chat/sessions/<str:session_id>/update_rag_config/', api_views.api_chat_update_rag_config, name='api_chat_update_rag_config'),
    path('api/user/rag-collections/', api_views.api_user_rag_collections, name='api_user_rag_collections'),

    # Эмбендинги
    path('embedding-models/', views.EmbeddingModelListView.as_view(), name='embedding_model_list'),
    path('embedding-models/create/', views.EmbeddingModelCreateView.as_view(), name='embedding_model_create'),
    path('embedding-models/<int:pk>/', views.EmbeddingModelDetailView.as_view(), name='embedding_model_detail'),
    path('embedding-models/<int:pk>/delete/', views.embedding_model_delete, name='embedding_model_delete'),
    path('collections/<int:pk>/rebuild-embeddings-index/', views.rebuild_embeddings_index, name='rebuild_embeddings_index'),
    path('collections/<int:pk>/rebuild-embeddings/', views.rebuild_collection_embeddings, name='rebuild_collection_embeddings'),
    path('collections/<int:pk>/change-embedding-model/', views.change_collection_embedding_model, name='change_collection_embedding_model'),
    path('embedding-models/<int:pk>/edit/', views.EmbeddingModelUpdateView.as_view(), name='embedding_model_edit'),
    path('embedding-models/<int:pk>/download/', views.embedding_model_download, name='embedding_model_download'),
    path('embedding-models/<int:pk>/download-status/', views.embedding_model_download_status, name='embedding_model_download_status'),
    path('embedding-models/<int:pk>/reset-download/', views.embedding_model_reset_download, name='embedding_model_reset_download'),

    # Дистилляция
    path('questionsets/', views.DistillationQuestionSetListView.as_view(), name='distillation_question_set_list'),
    path('questionsets/create/', views.DistillationQuestionSetCreateView.as_view(), name='distillation_question_set_create'),
    path('questionsets/<int:pk>/', views.DistillationQuestionSetDetailView.as_view(), name='distillation_question_set_detail'),
    path('questionsets/<int:pk>/edit-groups/', views.distillation_question_set_edit_groups, name='distillation_question_set_edit_groups'),
    path('questionsets/<int:pk>/start-questions/', views.distillation_start_question_generation, name='distillation_start_question_generation'),
    path('questions/', views.DistillationQuestionListView.as_view(), name='distillation_question_list'),
    path('questions/<int:pk>/edit/', views.distillation_question_edit, name='distillation_question_edit'),
    path('answerjobs/', views.DistillationAnswerJobListView.as_view(), name='distillation_answer_job_list'),
    path('answerjobs/create/', views.DistillationAnswerJobCreateView.as_view(), name='distillation_answer_job_create'),
    path('answerjobs/<int:pk>/', views.DistillationAnswerJobDetailView.as_view(), name='distillation_answer_job_detail'),
    path('answerjobs/<int:pk>/start/', views.distillation_start_answer_generation, name='distillation_start_answer_generation'),
    path('distillation/answerjobs/create/', views.DistillationAnswerJobCreateView.as_view(), name='distillation_answer_job_create'),
    path('answerjobs/<int:pk>/delete/', views.distillation_delete_answer_job, name='distillation_answer_job_delete'),
    path('answerjobs/<int:pk>/restart/', views.distillation_restart_answer_job, name='distillation_restart_answer_job'),
    path('distillation/questionsets/<int:pk>/resume/', views.distillation_resume_question_generation, name='distillation_resume_question_generation'),
    path('distillation/answerjobs/<int:pk>/resume/', views.distillation_resume_answer_generation, name='distillation_resume_answer_generation'),
    path('distillation/dataset-builder/', views.DistillationDatasetBuilderView.as_view(), name='distillation_dataset_builder'),
    path('distillation/datasets/', views.DistillationDatasetListView.as_view(), name='distillation_dataset_list'),
    path('distillation/datasets/<int:pk>/', views.DistillationDatasetDetailView.as_view(), name='distillation_dataset_detail'),
    path('distillation/datasets/<int:pk>/export/', views.distillation_dataset_export, name='distillation_dataset_export'),
    path('questionsets/<int:pk>/edit/', views.DistillationQuestionSetUpdateView.as_view(), name='distillation_question_set_edit'),
    
    # Ручные инструкции
    path('manual-instructions/', views.ManualInstructionBuilderView.as_view(), name='manual_instruction_builder'),
    path('manual-instruction-sets/<int:pk>/', views.manual_instruction_set_detail, name='manual_instruction_set_detail'),
    path('manual-instruction-sets/<int:pk>/export/', views.manual_instruction_set_export, name='manual_instruction_set_export'),

    # Hugging Face Datasets
    path('hf-datasets/', views.HuggingFaceDatasetListView.as_view(), name='hf_dataset_list'),
    path('hf-datasets/create/', views.HuggingFaceDatasetCreateView.as_view(), name='hf_dataset_create'),
    path('hf-datasets/<int:pk>/', views.HuggingFaceDatasetDetailView.as_view(), name='hf_dataset_detail'),
    path('hf-datasets/<int:pk>/import/', views.hf_dataset_start_import, name='hf_dataset_import'),
    path('hf-datasets/<int:pk>/delete/', views.hf_dataset_delete, name='hf_dataset_delete'),

    # API дистилляции
    path('api/distillation/task-status/<int:question_set_id>/', api_views.distillation_task_status, name='api_distillation_task_status'),
    path('api/distillation/answer-job-status/<int:job_id>/', api_views.distillation_answer_job_status, name='api_distillation_answer_job_status'),
    
    path('api/distillation/dataset-builder-data/', api_views.distillation_dataset_builder_data, name='api_distillation_dataset_builder_data'),
    # Слияние моделей
    path('merge/create/', views.ModelMergeView.as_view(), name='merge_create')  # FIX: был name='merge_detail', а {% url 'merge_create' %} в 4 шаблонах падал с NoReverseMatch

]

if settings.DEBUG:
    from django.conf.urls.static import static
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)