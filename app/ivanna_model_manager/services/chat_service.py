import json
import time
import logging
from typing import Dict, Any, List, Optional
from django.utils import timezone
from django.db import IntegrityError

from .redis_model_service import redis_model_manager
from .rag_service import EnhancedSearchService, RAGService  # RAG остаётся крутым
from ..tasks import generate_text_task
logger = logging.getLogger(__name__)


class ChatService:
    """
    Сервис чата с интеграцией Redis и RAG.
    - Управление сессиями (создание, получение, добавление сообщений).
    - Генерация ответов через redis_model_manager.
    - Интеграция с RAG (поиск документов, добавление контекста).
    """

    def __init__(self):
        self.redis_manager = redis_model_manager
        self.rag_service = RAGService()
        self.search_service = EnhancedSearchService()

    # ----------------------------------------------------------------------
    # Управление сессиями (Redis + БД)
    # ----------------------------------------------------------------------
    def create_session(self, user_id: str, model_id: str = None) -> Dict[str, Any]:
        """
        Создаёт новую сессию чата.
        - Генерирует уникальный session_id.
        - Сохраняет базовую информацию в Redis (TTL 1 день).
        - Создаёт запись в БД ChatSession.
        """
        from ..models import ChatSession, ModelCheckpoint

        timestamp = int(time.time())
        random_suffix = int(time.time() * 1000) % 10000
        session_id = f"chat_{user_id}_{timestamp}_{random_suffix}"

        # Конфигурация по умолчанию
        generation_config = {}
        model = None
        if model_id:
            try:
                model = ModelCheckpoint.objects.get(id=model_id, is_active=True)
                generation_config = model.loader_config.get('generation_config', {})
            except ModelCheckpoint.DoesNotExist:
                logger.warning(f"Model {model_id} not found, session created without model")

        # Данные для Redis
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'model_id': model_id,
            'is_active': True,
            'created_at': timezone.now().isoformat(),
            'last_activity': timezone.now().isoformat(),
            'conversation_history': [],
            # Объединённая конфигурация для сессии
            'session_config': {
                'loader_overrides': {},
                'generation_config': generation_config,
                'system_prompt': 'Ты полезный ассистент. Отвечай на русском языке.',
            },
            # RAG настройки (по умолчанию выключены)
            'rag_config': {
                'enabled': False,
                'selected_collections': [],
                'top_k': 5,
                'semantic_weight': 0.7,
                'keyword_weight': 0.2,
                'freshness_weight': 0.1,
                'use_reranking': True,
            }
        }

        # Сохраняем в Redis
        redis_key = f"chat:session:{session_id}"
        self.redis_manager.redis.setex(redis_key, 86400, json.dumps(session_data, default=str))

        # Создаём в БД
        try:
            chat_session = ChatSession.objects.create(
                session_id=session_id,
                user_id=user_id,
                checkpoint=model,
                is_active=True,
                created_at=timezone.now(),
                last_activity=timezone.now(),
                generation_config=generation_config,
                conversation_history=[]
            )
            if model:
                chat_session.model_config = model.loader_config.copy()
                chat_session.save()
            logger.info(f"✅ Chat session created: {session_id}")
        except IntegrityError:
            # Если вдруг такой session_id уже есть (крайне маловероятно)
            session_id = f"chat_{user_id}_{timestamp}_{int(time.time() * 1000000) % 1000000}"
            session_data['session_id'] = session_id
            self.redis_manager.redis.setex(f"chat:session:{session_id}", 86400, json.dumps(session_data, default=str))
            chat_session = ChatSession.objects.create(
                session_id=session_id,
                user_id=user_id,
                checkpoint=model,
                is_active=True,
                created_at=timezone.now(),
                last_activity=timezone.now(),
                generation_config=generation_config,
                conversation_history=[]
            )
            logger.info(f"✅ Chat session created (fallback): {session_id}")

        return {
            'success': True,
            'session_id': session_id,
            'user_id': user_id,
            'model_id': model_id,
            'created_at': session_data['created_at'],
            'session_config': session_data['session_config'],
            'rag_config': session_data['rag_config'],
            'db_session_id': chat_session.id,
        }

    def get_session(self, session_id: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        Получить сессию из Redis.
        Если в Redis нет, пытается восстановить из БД.
        
        Args:
            session_id: ID сессии
            force_refresh: если True, игнорирует возможный локальный кэш и всегда берет из Redis
        """
        redis_key = f"chat:session:{session_id}"
        
        # Логируем вызов
        logger.info(f"🔍 get_session: session={session_id}, force_refresh={force_refresh}")
        
        # Если force_refresh, удаляем ключ из Redis, чтобы получить свежие данные
        if force_refresh:
            logger.info(f"🔄 Force refresh for session {session_id}")
            # Можно не удалять, а просто читать заново, но для гарантии удалим
            # self.redis_manager.redis.delete(redis_key)
            # Но лучше просто читаем заново без удаления
        
        # Получаем данные из Redis
        session_data = self.redis_manager.redis.get(redis_key)

        if session_data:
            try:
                session = json.loads(session_data)
                # Обновляем время последней активности
                session['last_activity'] = timezone.now().isoformat()
                # Продлеваем TTL
                self.redis_manager.redis.setex(redis_key, 86400, json.dumps(session, default=str))
                logger.info(f"✅ Session {session_id} loaded from Redis, history length: {len(session.get('conversation_history', []))}")
                return session
            except json.JSONDecodeError as e:
                logger.error(f"❌ Invalid session data for {session_id}: {e}")
                return None

        # Если в Redis нет, восстанавливаем из БД
        logger.info(f"🔄 Session {session_id} not in Redis, restoring from DB")
        try:
            from ..models import ChatSession
            chat_session = ChatSession.objects.get(session_id=session_id, is_active=True)

            generation_config = chat_session.generation_config or {}
            session_data = {
                'session_id': chat_session.session_id,
                'user_id': chat_session.user_id,
                'model_id': chat_session.checkpoint.id if chat_session.checkpoint else None,
                'is_active': chat_session.is_active,
                'created_at': chat_session.created_at.isoformat(),
                'last_activity': chat_session.last_activity.isoformat(),
                'conversation_history': chat_session.conversation_history or [],
                'session_config': {
                    'loader_overrides': {},
                    'generation_config': generation_config,
                    'system_prompt': chat_session.system_prompt or 'Ты полезный ассистент. Отвечай на русском языке.',
                },
                'rag_config': {},  # можно будет расширить
            }
            # Сохраняем в Redis
            self.redis_manager.redis.setex(redis_key, 86400, json.dumps(session_data, default=str))
            logger.info(f"✅ Session {session_id} restored from DB, history length: {len(session_data.get('conversation_history', []))}")
            return session_data
        except ChatSession.DoesNotExist:
            logger.warning(f"❌ Session {session_id} not found in DB")
            return None
        except Exception as e:
            logger.error(f"❌ Error restoring session {session_id} from DB: {e}", exc_info=True)
            return None

    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None) -> bool:
        """
        Добавить сообщение в историю диалога.
        Обновляет Redis и БД.
        """
        import traceback
        
        logger.info(f"🔵 [add_message] START: session={session_id}, role={role}, content_len={len(content)}")
        
        # 1. Получаем сессию
        session = self.get_session(session_id)
        if not session:
            logger.error(f"❌ [add_message] Session {session_id} not found")
            return False
        
        logger.info(f"✅ [add_message] Session found, current history length: {len(session.get('conversation_history', []))}")
        
        # 2. Создаём сообщение
        message = {
            'role': role,
            'content': content,
            'timestamp': timezone.now().isoformat()
        }
        if metadata:
            message['metadata'] = self._serialize_metadata(metadata)
            logger.info(f"📝 [add_message] Metadata included: {list(metadata.keys())}")
        
        # 3. Добавляем в Redis
        if 'conversation_history' not in session:
            session['conversation_history'] = []
            logger.info("📝 [add_message] Created new conversation_history array")
        
        before_count = len(session['conversation_history'])
        session['conversation_history'].append(message)
        after_count = len(session['conversation_history'])
        logger.info(f"📝 [add_message] History size: {before_count} → {after_count}")
        
        # Ограничим историю 50 сообщениями
        if after_count > 50:
            session['conversation_history'] = session['conversation_history'][-50:]
            logger.info(f"✂️ [add_message] Trimmed history to 50 messages")
        
        session['last_activity'] = timezone.now().isoformat()
        redis_key = f"chat:session:{session_id}"
        
        # 4. Сохраняем в Redis
        try:
            serialized = json.dumps(session, default=str)
            self.redis_manager.redis.setex(redis_key, 86400, serialized)
            logger.info(f"💾 [add_message] Saved to Redis, key={redis_key}, size={len(serialized)}")
        except Exception as e:
            logger.error(f"❌ [add_message] Redis save error: {e}")
            logger.error(traceback.format_exc())
            return False
        
        # 5. Обновляем БД
        try:
            from ..models import ChatSession
            chat_session = ChatSession.objects.get(session_id=session_id)
            
            if role == 'user':
                chat_session.add_message('user', content)
                logger.info(f"💾 [add_message] DB updated for user message")
            else:
                chat_session.add_message('assistant', content, metadata or {})
                logger.info(f"💾 [add_message] DB updated for assistant message")
                
        except ChatSession.DoesNotExist:
            logger.error(f"❌ [add_message] ChatSession {session_id} not found in DB")
        except Exception as e:
            logger.error(f"❌ [add_message] DB error: {e}")
            logger.error(traceback.format_exc())
        
        logger.info(f"🟢 [add_message] END: success=True")
        return True

    def get_conversation_history(self, session_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """Получить историю диалога из Redis."""
        session = self.get_session(session_id)
        if not session:
            return []
        history = session.get('conversation_history', [])
        if limit:
            return history[-limit:]
        return history

    def clear_history(self, session_id: str) -> bool:
        """Очистить историю чата."""
        session = self.get_session(session_id)
        if not session:
            return False
        session['conversation_history'] = []
        session['last_activity'] = timezone.now().isoformat()
        redis_key = f"chat:session:{session_id}"
        self.redis_manager.redis.setex(redis_key, 86400, json.dumps(session, default=str))

        try:
            from ..models import ChatSession
            chat_session = ChatSession.objects.get(session_id=session_id)
            chat_session.conversation_history = []
            chat_session.total_messages = 0
            chat_session.total_input_tokens = 0
            chat_session.total_output_tokens = 0
            chat_session.save()
        except Exception as e:
            logger.error(f"Failed to clear DB history: {e}")
        return True

    def close_session(self, session_id: str) -> bool:
        """Закрыть сессию (деактивировать и удалить из Redis)."""
        try:
            from ..models import ChatSession
            chat_session = ChatSession.objects.get(session_id=session_id)
            chat_session.is_active = False
            chat_session.save()
        except Exception:
            pass
        redis_key = f"chat:session:{session_id}"
        self.redis_manager.redis.delete(redis_key)
        logger.info(f"Closed session {session_id}")
        return True

    def list_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        from ..models import ChatSession
        sessions = ChatSession.objects.filter(
            user_id=user_id,
            is_active=True
        ).order_by('-last_activity')

        result = []
        for s in sessions:
            redis_data = self.get_session(s.session_id)
            total_tokens = s.total_input_tokens + s.total_output_tokens
            history = redis_data.get('conversation_history', s.conversation_history or []) if redis_data else s.conversation_history or []
            last_msg = history[-1].get('content', '')[:100] if history else ''

            # Проверяем, загружена ли модель
            model_loaded = False
            if s.checkpoint:
                model_id = s.checkpoint.huggingface_id or s.checkpoint.name
                info = self.redis_manager.get_model_info(model_id)
                if info and info.get('loaded'):
                    model_loaded = True

            session_info = {
                'id': s.id,
                'session_id': s.session_id,
                'model_id': s.checkpoint.id if s.checkpoint else None,
                'model_name': s.checkpoint.display_name if s.checkpoint else None,
                'model_loaded': model_loaded,   # новое поле
                'notes': s.notes or f"Сессия от {s.created_at.strftime('%d.%m.%Y')}",
                'created_at': s.created_at.isoformat(),
                'last_activity': s.last_activity.isoformat(),
                'total_messages': s.total_messages,
                'total_tokens': total_tokens,
                'message_count': len(history),
                'last_message': last_msg,
                'redis_data': redis_data,
            }
            result.append(session_info)
        return result

    # ----------------------------------------------------------------------
    # Генерация ответов (обычный чат)
    # ----------------------------------------------------------------------

    from celery.result import AsyncResult
    from ..tasks import generate_text_task

    # ----------------------------------------------------------------------
    # Генерация ответов с RAG
    # ----------------------------------------------------------------------
    def generate_response(self, session_id: str, message: str,
                          generation_config_overrides: Dict = None,
                          use_rag: bool = None,
                          rag_config: Dict = None) -> Dict[str, Any]:
        """
        Генерация ответа ассистента.
        Если переданы use_rag и rag_config, они имеют приоритет над сохранёнными в сессии.
        """
        logger.info(f"🟡 [generate_response] START: session={session_id}, message_len={len(message)}")

        session = self.get_session(session_id)
        if not session:
            logger.error(f"❌ [generate_response] Session {session_id} not found")
            return {'success': False, 'error': 'Session not found'}

        model_id = session.get('model_id')
        if not model_id:
            logger.error(f"❌ [generate_response] No model selected for session {session_id}")
            return {'success': False, 'error': 'No model selected for chat'}

        logger.info(f"✅ [generate_response] Session found, model_id={model_id}")

        # Получаем конфигурацию из сессии
        session_config = session.get('session_config', {})
        system_prompt = session_config.get('system_prompt', 'Ты полезный ассистент. Отвечай на русском языке.')
        gen_config = session_config.get('generation_config', {}).copy()
        if generation_config_overrides:
            gen_config.update(generation_config_overrides)
            logger.info(f"⚙️ [generate_response] Generation config overrides applied")

        # Определяем финальные RAG-параметры
        session_rag = session.get('rag_config', {})
        final_use_rag = use_rag if use_rag is not None else session_rag.get('enabled', False)
        final_rag_config = rag_config if rag_config is not None else session_rag

        # Проверяем, включён ли RAG и есть ли выбранные коллекции
        use_rag_enabled = final_use_rag and final_rag_config.get('selected_collections', [])
        logger.info(f"🔍 [generate_response] RAG enabled: {use_rag_enabled}, collections: {final_rag_config.get('selected_collections', [])}")

        # Добавляем сообщение пользователя в историю
        logger.info(f"📝 [generate_response] Calling add_message for user message...")
        add_result = self.add_message(session_id, 'user', message)
        if not add_result:
            logger.error(f"❌ [generate_response] Failed to add user message to session {session_id}")
            return {'success': False, 'error': 'Failed to save message'}

        # Получаем обновлённую сессию для формирования промпта
        session = self.get_session(session_id, force_refresh=True)
        if not session:
            logger.error(f"❌ [generate_response] Session {session_id} not found after adding message")
            return {'success': False, 'error': 'Session not found'}

        # Собираем промпт с историей
        history = session.get('conversation_history', [])
        prompt_parts = [f"System: {system_prompt}"]
        for msg in history:
            role = 'User' if msg['role'] == 'user' else 'Assistant'
            prompt_parts.append(f"{role}: {msg['content']}")
        prompt_parts.append("Assistant:")
        full_prompt = "\n".join(prompt_parts)

        # Запускаем задачу Celery с RAG-параметрами
        try:
            task = generate_text_task.delay(
                model_id=int(model_id),
                prompt=full_prompt,
                session_id=session_id,
                use_rag=use_rag_enabled,
                rag_config=final_rag_config if use_rag_enabled else None,
                **gen_config
            )
            logger.info(f"🚀 [generate_response] Task created: {task.id}")
        except Exception as e:
            logger.error(f"❌ [generate_response] Failed to create task: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

        # Сохраняем ID задачи в Redis
        session['last_task_id'] = task.id
        session['pending_messages'] = session.get('pending_messages', 0) + 1
        redis_key = f"chat:session:{session_id}"
        self.redis_manager.redis.setex(redis_key, 86400, json.dumps(session, default=str))

        return {
            'success': True,
            'task_id': task.id,
            'session_id': session_id,
            'async': True
        }

    # ----------------------------------------------------------------------
    # RAG: поиск документов, управление коллекциями
    # ----------------------------------------------------------------------
    def search_documents(self, session_id: str, query: str,
                         collection_ids: List[int] = None) -> Dict[str, Any]:
        """
        Поиск по документам RAG. Использует EnhancedSearchService.
        """
        session = self.get_session(session_id)
        if not session:
            return {'success': False, 'error': 'Session not found'}

        if not collection_ids:
            collection_ids = session.get('rag_config', {}).get('selected_collections', [])
        if not collection_ids:
            return {'success': False, 'error': 'No collections selected for search'}

        try:
            # Используем улучшенный гибридный поиск
            rag_config = session.get('rag_config', {})
            results = self.search_service.enhanced_hybrid_search(
                query=query,
                collection_id=collection_ids[0] if len(collection_ids) == 1 else None,
                k=rag_config.get('top_k', 5),
                semantic_weight=rag_config.get('semantic_weight', 0.7),
                keyword_weight=rag_config.get('keyword_weight', 0.2),
                freshness_weight=rag_config.get('freshness_weight', 0.1),
                use_reranking=rag_config.get('use_reranking', True)
            )

            formatted = []
            for r in results[:rag_config.get('top_k', 5)]:
                formatted.append({
                    'chunk_id': r.get('chunk_id'),
                    'document_id': r.get('document_id'),
                    'document_title': r.get('document_title', 'Без названия'),
                    'content': r.get('content', ''),
                    'preview': (r.get('content', '')[:300] + '...') if len(r.get('content', '')) > 300 else r.get('content', ''),
                    'relevance_score': r.get('final_score', r.get('similarity', 0)),
                    'scores': {
                        'semantic': r.get('semantic_score', 0),
                        'keyword': r.get('keyword_score', 0),
                        'freshness': r.get('freshness_score', 0),
                        'reranker': r.get('reranker_score', 0),
                    },
                    'source': r.get('source', 'hybrid'),
                })

            # Сохраняем последний поиск в сессию
            session['rag_config'] = session.get('rag_config', {})
            session['rag_config']['last_search'] = {
                'query': query,
                'timestamp': timezone.now().isoformat(),
                'results_count': len(formatted),
                'collection_ids': collection_ids,
            }
            redis_key = f"chat:session:{session_id}"
            self.redis_manager.redis.setex(redis_key, 86400, json.dumps(session, default=str))

            return {'success': True, 'query': query, 'total_results': len(formatted), 'results': formatted}

        except Exception as e:
            logger.error(f"Error searching documents: {e}", exc_info=True)
            return {'success': False, 'error': f'Search error: {str(e)}'}

    def update_rag_config(self, session_id: str, rag_config: Dict) -> bool:
        """Обновить RAG-конфигурацию сессии."""
        session = self.get_session(session_id)
        if not session:
            return False
        session['rag_config'] = session.get('rag_config', {})
        session['rag_config'].update(rag_config)
        session['rag_config']['last_updated'] = timezone.now().isoformat()
        redis_key = f"chat:session:{session_id}"
        self.redis_manager.redis.setex(redis_key, 86400, json.dumps(session, default=str))
        logger.info(f"✅ RAG config updated for session {session_id}")
        return True

    def get_user_collections(self, user_id: str) -> List[Dict[str, Any]]:
        """Получить список коллекций, доступных пользователю."""
        try:
            from ..models import DocumentCollection
            from django.db.models import Q, Count

            collections = DocumentCollection.objects.filter(
                Q(owner_id=str(user_id)) | Q(is_public=True),
                is_active=True
            ).annotate(
                doc_count=Count('documents', filter=Q(documents__is_active=True)),
                chunk_count=Count('documents__chunks', filter=Q(documents__chunks__is_active=True)),
                has_embedding_count=Count('documents__chunks', filter=Q(documents__chunks__has_embedding=True))
            ).order_by('-created_at')

            result = []
            for c in collections:
                result.append({
                    'id': c.id,
                    'name': c.name,
                    'description': c.description,
                    'doc_count': c.doc_count,
                    'chunk_count': c.chunk_count,
                    'has_embeddings': c.has_embedding_count > 0,
                    'vector_index_created': c.vector_index_created,
                    'is_public': c.is_public,
                    'is_owner': c.owner_id == str(user_id),
                    'created_at': c.created_at.isoformat() if c.created_at else None,
                    'embedding_progress': round(
                        (c.has_embedding_count / c.chunk_count * 100) if c.chunk_count > 0 else 0, 1
                    ),
                })
            return result
        except Exception as e:
            logger.error(f"Error getting user collections: {e}")
            return []

    def _get_rag_context(self, session_id: str, query: str, rag_config: Dict) -> str:
        """Получает контекст из RAG для запроса."""
        collection_ids = rag_config.get('selected_collections', [])
        if not collection_ids:
            return ""
        
        try:
            from .rag_service import EnhancedSearchService
            search_service = EnhancedSearchService()
            
            # Используем первую коллекцию (или можно модифицировать для нескольких)
            collection_id = collection_ids[0]
            
            results = search_service.enhanced_hybrid_search(
                query=query,
                collection_id=collection_id,
                k=rag_config.get('top_k', 3),
                semantic_weight=rag_config.get('semantic_weight', 0.7),
                keyword_weight=rag_config.get('keyword_weight', 0.2),
                freshness_weight=rag_config.get('freshness_weight', 0.1),
                use_reranking=rag_config.get('use_reranking', True)
            )
            
            context_parts = []
            for i, r in enumerate(results[:3]):
                context_parts.append(
                    f"[Документ {i+1}: {r.get('document_title', 'Без названия')}]\n{r.get('content', '')[:500]}"
                )
            
            return "\n\n---\n\n".join(context_parts)
        except Exception as e:
            logger.error(f"Error getting RAG context: {e}")
            return ""

    # ----------------------------------------------------------------------
    # Вспомогательные методы
    # ----------------------------------------------------------------------
    @staticmethod
    def _serialize_metadata(metadata: Dict) -> Dict:
        """Сериализация метаданных для JSON (преобразование datetime, etc)."""
        if not metadata:
            return {}
        serialized = {}
        for k, v in metadata.items():
            if hasattr(v, 'isoformat'):
                serialized[k] = v.isoformat()
            elif isinstance(v, dict):
                serialized[k] = ChatService._serialize_metadata(v)
            elif isinstance(v, list):
                serialized[k] = [
                    ChatService._serialize_metadata(item) if isinstance(item, dict) else
                    item.isoformat() if hasattr(item, 'isoformat') else item
                    for item in v
                ]
            else:
                serialized[k] = v
        return serialized


# Глобальный экземпляр
chat_service = ChatService()