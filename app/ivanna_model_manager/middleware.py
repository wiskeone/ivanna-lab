"""v5.1: middleware for correlation ID — every HTTP request gets a UUID
that flows through all logs (web + Celery via broker).
v5.2: default interface language is Russian, regardless of browser Accept-Language.
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
    """Default language = settings.LANGUAGE_CODE (ru) unless the user has
    explicitly chosen a language (django_language cookie from the 🌐 switch).

    LocaleMiddleware picks up browser Accept-Language, which made the
    interface default to English for English-language browsers.
    Must be placed AFTER LocaleMiddleware.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not request.COOKIES.get(settings.LANGUAGE_COOKIE_NAME):
            request.LANGUAGE_CODE = settings.LANGUAGE_CODE
        return self.get_response(request)
