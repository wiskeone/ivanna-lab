"""v5.1: middleware для корреляционного ID — каждый HTTP-запрос получает
UUID, который проходит через все логи (веб + Celery через брокер).
v5.2: дефолт языка интерфейса — русский, независимо от Accept-Language браузера.
"""
import uuid

from django.conf import settings

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


class DefaultLanguageMiddleware:
    """Язык по умолчанию = settings.LANGUAGE_CODE (ru), если пользователь не
    выбрал язык явно (cookie django_language от переключателя 🌐).

    LocaleMiddleware подхватывает Accept-Language браузера, из-за чего у
    англоязычных браузеров интерфейс по умолчанию был английским.
    Должен стоять ПОСЛЕ LocaleMiddleware.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not request.COOKIES.get(settings.LANGUAGE_COOKIE_NAME):
            request.LANGUAGE_CODE = settings.LANGUAGE_CODE
        return self.get_response(request)
