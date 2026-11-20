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



class ChatMessageSerializer(serializers.ModelSerializer):
    """Serializer for chat messages"""
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
    """Serializer for chat sessions"""
    checkpoint_display = serializers.CharField(source='checkpoint.display_name', read_only=True)
    model_type = serializers.CharField(source='checkpoint.model_type', read_only=True)
    
    # New config fields
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
        """Get session_config from Redis"""
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
        """Compute the merged config"""
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
        """Return available presets for the session model"""
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
        # Handle generation_config separately for backward compatibility
        generation_config = validated_data.pop('generation_config', None)
        if generation_config:
            from ..services.chat_service import chat_service
            # Get the current session from Redis
            session = chat_service.get_session(instance.session_id)
            if session:
                if 'session_config' not in session:
                    session['session_config'] = {}
                session['session_config']['generation_config'] = generation_config
                # Save back to Redis
                redis_key = f"chat:session:{instance.session_id}"
                chat_service.redis_manager.redis.setex(
                    redis_key, 86400, 
                    json.dumps(session, default=str)
                )
        
        # Handle system_prompt
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
    """Serializer for model comparisons"""
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


class ChatConfigUpdateSerializer(serializers.Serializer):
    """Serializer for updating chat session config"""
    
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
        """Validate loader overrides"""
        if not isinstance(value, dict):
            raise serializers.ValidationError("loader_overrides должен быть словарем")
        
        allowed_keys = {
            'torch_dtype', 'device_map', 'trust_remote_code', 
            'use_flash_attention_2', 'quantization_config',
            'dtype', 'max_memory', 'low_cpu_mem_usage'
        }
        
        # Keep only allowed keys
        filtered = {}
        for k, v in value.items():
            if k in allowed_keys:
                filtered[k] = v
        
        # Validate torch_dtype/dtype
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
        """Validate generation parameters"""
        if not isinstance(value, dict):
            raise serializers.ValidationError("generation_config должен быть словарем")
        
        validated = {}
        
        # Max token count
        if 'max_tokens' in value:
            max_tokens = value['max_tokens']
            if not isinstance(max_tokens, int) or max_tokens < 1 or max_tokens > 8192:
                raise serializers.ValidationError(
                    "max_tokens должен быть целым числом от 1 до 8192"
                )
            validated['max_tokens'] = max_tokens
        
        # Temperature
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
        """Cross-field validation"""
        # If everything is empty, nothing to do
        if not data.get('loader_overrides') and not data.get('generation_config') and not data.get('system_prompt'):
            raise serializers.ValidationError("Нет данных для обновления конфигурации")
        
        return data


class ChatSessionConfigSerializer(serializers.Serializer):
    """Serializer for the full session config"""
    
    session_id = serializers.CharField(read_only=True)
    model_id = serializers.IntegerField(read_only=True)
    
    model_info = serializers.DictField(read_only=True)
    session_config = serializers.DictField(read_only=True)
    merged_generation_config = serializers.DictField(read_only=True)
    
    conversation_history_length = serializers.IntegerField(read_only=True)
    last_activity = serializers.CharField(read_only=True)


class ChatSessionCreateSerializer(serializers.Serializer):
    """Serializer for creating a chat session with config"""
    
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
