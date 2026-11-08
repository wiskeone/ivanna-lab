"""v5.1: middleware для корреляционного ID — каждый HTTP-запрос получает
UUID, который проходит через все логи (веб + Celery через брокер)."""
import uuid

from .log_context import set_request_id, clear_request_id


class RequestIdMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        rid = uuid.uuid4().hex[:12]
        set_request_id(rid)
        try:
            return self.get_response(request)
        finally:
            clear_request_id()
