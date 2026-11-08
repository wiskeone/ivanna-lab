"""
v5.0: клиент vLLM (OpenAI-совместимый API).

Позволяет направлять инференс сессий на внешний vLLM-сервер вместо
in-process transformers. Контракт результата совпадает с
redis_model_manager.generate_text — вызывающий код (generate_text_task)
не меняет форму ответа.

Конфигурация (settings.py / .env):
    VLLM_BASE_URL   — базовый URL, например http://vllm:8001/v1
    VLLM_API_KEY    — токен, если сервер за прокси (по умолчанию не нужен)
    VLLM_TIMEOUT    — таймаут запроса, сек (дефолт 300 — длинные генерации)
"""
import logging
import time

import requests
from django.conf import settings

logger = logging.getLogger(__name__)


class VLLMError(Exception):
    """Ошибка обращения к vLLM-серверу."""


class VLLMClient:
    def __init__(self, base_url: str = None, api_key: str = None, timeout: int = None):
        self.base_url = (base_url or getattr(settings, 'VLLM_BASE_URL', 'http://vllm:8001/v1')).rstrip('/')
        self.api_key = api_key or getattr(settings, 'VLLM_API_KEY', '') or None
        self.timeout = timeout or getattr(settings, 'VLLM_TIMEOUT', 300)

    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        h = {'Content-Type': 'application/json'}
        if self.api_key:
            h['Authorization'] = f'Bearer {self.api_key}'
        return h

    def health(self) -> bool:
        """GET /health — жив ли сервер (без /v1 префикса)."""
        try:
            base = self.base_url.rsplit('/v1', 1)[0]
            r = requests.get(f'{base}/health', timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list:
        """GET /v1/models — какие модели засервированы."""
        try:
            r = requests.get(f'{self.base_url}/models', headers=self._headers(), timeout=30)
            r.raise_for_status()
            return [m.get('id') for m in r.json().get('data', [])]
        except Exception as e:
            logger.warning(f"vLLM list_models failed: {e}")
            return []

    # ------------------------------------------------------------------

    @staticmethod
    def build_payload(model: str, prompt: str, max_tokens: int = 512,
                      temperature: float = 0.7, top_p: float = 1.0,
                      top_k: int = -1, repetition_penalty: float = 0.0,
                      stop: list = None, **dropped) -> dict:
        """Собирает payload /v1/completions. Неизвестные kwargs отбрасываются
        (совместимость с gen_config сессии), vLLM-расширения (top_k,
        repetition_penalty) передаются верхним уровнем — vLLM их понимает."""
        payload = {
            'model': model,
            'prompt': prompt,
            'max_tokens': int(max_tokens),
            'temperature': float(temperature),
            'top_p': float(top_p),
            'stream': False,
        }
        if top_k and int(top_k) > 0:
            payload['top_k'] = int(top_k)
        if repetition_penalty and float(repetition_penalty) >= 1.0:
            payload['repetition_penalty'] = float(repetition_penalty)
        if stop:
            payload['stop'] = stop
        return payload

    def generate(self, model: str, prompt: str, **kwargs) -> dict:
        """
        POST /v1/completions — flat-промпт (паритет с transformers-путём,
        где тоже подаётся собранный текст с System/User/Assistant).

        Возвращает словарь той же формы, что redis_model_manager.generate_text:
        {'success', 'response', 'input_tokens', 'output_tokens', 'total_tokens',
         'generation_time', 'tokens_per_second', 'device'}
        """
        payload = self.build_payload(model, prompt, **kwargs)
        start = time.time()
        try:
            r = requests.post(
                f'{self.base_url}/completions',
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            if r.status_code != 200:
                detail = ''
                try:
                    detail = r.json().get('error', {}).get('message', '') or r.text[:300]
                except Exception:
                    detail = r.text[:300]
                logger.error(f"vLLM completions failed: HTTP {r.status_code}: {detail}")
                return {'success': False, 'error': f'vLLM HTTP {r.status_code}: {detail}'}

            data = r.json()
            choices = data.get('choices') or []
            if not choices:
                return {'success': False, 'error': 'vLLM returned no choices'}
            response = choices[0].get('text', '')

            usage = data.get('usage', {}) or {}
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
            generation_time = time.time() - start

            return {
                'success': True,
                'response': response,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
                'generation_time': generation_time,
                'tokens_per_second': output_tokens / generation_time if generation_time > 0 else 0,
                'device': 'vllm',
            }
        except requests.exceptions.Timeout:
            logger.error(f"vLLM timeout after {self.timeout}s (model={model})")
            return {'success': False, 'error': f'vLLM request timed out after {self.timeout}s'}
        except requests.exceptions.ConnectionError as e:
            logger.error(f"vLLM connection error: {e}")
            return {'success': False, 'error': f'vLLM is unreachable: {e}'}
        except Exception as e:
            logger.error(f"vLLM generate error: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    # ------------------------------------------------------------------

    def chat(self, model: str, messages: list, **kwargs) -> dict:
        """
        POST /v1/chat/completions — для сценариев с шаблоном диалога.
        messages — стандартный OpenAI-формат [{'role', 'content'}, ...].
        """
        payload = {
            'model': model,
            'messages': messages,
            'stream': False,
            'max_tokens': int(kwargs.get('max_tokens', 512)),
            'temperature': float(kwargs.get('temperature', 0.7)),
        }
        try:
            r = requests.post(
                f'{self.base_url}/chat/completions',
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            content = (data.get('choices') or [{}])[0].get('message', {}).get('content', '')
            usage = data.get('usage', {}) or {}
            return {
                'success': True,
                'response': content,
                'input_tokens': usage.get('prompt_tokens', 0),
                'output_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0),
                'device': 'vllm',
            }
        except Exception as e:
            logger.error(f"vLLM chat error: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
