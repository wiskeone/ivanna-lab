"""v4: CI-настройки — sqlite, минимум внешних сервисов."""
from .settings import *  # noqa: F401,F403

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': '/tmp/ivanna_test.sqlite3',
    }
}
CACHES = {
    'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}
}
CELERY_TASK_ALWAYS_EAGER = True
