import os
from pathlib import Path
from dotenv import load_dotenv
from celery.schedules import crontab

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# -------------------------------------------------------------------------
# Безопасность
# -------------------------------------------------------------------------
SECRET_KEY = os.getenv('SECRET_KEY', 'django-insecure-default-key-change-in-production')
DEBUG = os.getenv('DEBUG', 'False') == 'True'

ALLOWED_HOSTS = [
    'localhost',
    '127.0.0.1',
    '0.0.0.0',
    'web',
    'web:8000',
]
# v61.5: i18n — русский исходник интерфейса, английский каркас (locale/)
from django.utils.translation import gettext_lazy as _
LANGUAGE_CODE = 'ru'
USE_I18N = True
LANGUAGES = [('ru', _('Русский')), ('en', _('English'))]
LOCALE_PATHS = [BASE_DIR / 'locale']

# v5.0: vLLM — внешний OpenAI-совместимый инференс-бэкенд
VLLM_BASE_URL = os.getenv('VLLM_BASE_URL', 'http://vllm:8001/v1')
VLLM_API_KEY = os.getenv('VLLM_API_KEY', '')
VLLM_TIMEOUT = int(os.getenv('VLLM_TIMEOUT', '300'))
VLLM_SERVED_MODEL = os.getenv('VLLM_SERVED_MODEL', '')

env_hosts = os.getenv('ALLOWED_HOSTS', '')
if env_hosts:
    ALLOWED_HOSTS.extend([host.strip() for host in env_hosts.split(',') if host.strip()])

# -------------------------------------------------------------------------
# Приложения и middleware
# -------------------------------------------------------------------------
INSTALLED_APPS = [
    'django_prometheus',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    'rest_framework',
    'corsheaders',
    'django_filters',
    'drf_yasg',

    'debug_toolbar',

    'ivanna_model_manager',
]

MIDDLEWARE = [
    'django_prometheus.middleware.PrometheusBeforeMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.locale.LocaleMiddleware',
    'ivanna_model_manager.middleware.RequestIdMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'django_prometheus.middleware.PrometheusAfterMiddleware',
]

# Добавляем debug_toolbar middleware только если DEBUG=True
if DEBUG:
    MIDDLEWARE.insert(0, 'debug_toolbar.middleware.DebugToolbarMiddleware')
    # Для корректной работы debug_toolbar необходимо указать INTERNAL_IPS
    INTERNAL_IPS = [
        '127.0.0.1',
        'localhost',
    ]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
            BASE_DIR / 'templates',
            BASE_DIR / 'ivanna_model_manager' / 'templates',
        ],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'ivanna_model_manager.context_processors.rag_stats',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

# -------------------------------------------------------------------------
# База данных
# -------------------------------------------------------------------------
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('DB_NAME', 'ivanna_lab'),
        'USER': os.getenv('DB_USER', 'ml_user'),
        'PASSWORD': os.getenv('DB_PASSWORD', 'secure_password'),
        'HOST': os.getenv('DB_HOST', 'localhost'),
        'PORT': os.getenv('DB_PORT', '5432'),
    }
}

# -------------------------------------------------------------------------
# Кэш (Redis)
# -------------------------------------------------------------------------
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': f"redis://{os.getenv('REDIS_HOST', 'localhost')}:6379/1",
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}

# -------------------------------------------------------------------------
# Celery
# -------------------------------------------------------------------------
CELERY_BROKER_URL = f"redis://{os.getenv('REDIS_HOST', 'localhost')}:6379/0"
CELERY_RESULT_BACKEND = f"redis://{os.getenv('REDIS_HOST', 'localhost')}:6379/0"
CELERY_ACCEPT_CONTENT = ['json', 'pathaware_json']
CELERY_TASK_SERIALIZER = 'pathaware_json'
CELERY_RESULT_SERIALIZER = 'pathaware_json'
CELERY_TIMEZONE = 'UTC'
CELERY_BROKER_CONNECTION_RETRY_ON_STARTUP = True

# Маршрутизация задач по очередям
CELERY_TASK_ROUTES = {
    # GPU задачи
    'ivanna_model_manager.tasks.load_model_to_memory_task': {'queue': 'gpu_tasks'},
    'ivanna_model_manager.tasks.unload_model_from_memory_task': {'queue': 'gpu_tasks'},
    'ivanna_model_manager.tasks.generate_text_task': {'queue': 'gpu_tasks'},
    'ivanna_model_manager.tasks.merge_models_task': {'queue': 'gpu_tasks'},
    'ivanna_model_manager.tasks.run_evaluation_task': {'queue': 'gpu_tasks'},
    'ivanna_model_manager.tasks.process_document_task': {'queue': 'gpu_tasks'},
    'ivanna_model_manager.tasks.create_embeddings_for_collection_task': {'queue': 'gpu_tasks'},
    'ivanna_model_manager.tasks.run_fine_tuning_task': {'queue': 'gpu_tasks'},

    # CPU задачи
    'ivanna_model_manager.tasks.download_model_task': {'queue': 'cpu_tasks'},
    'ivanna_model_manager.tasks.cleanup_download_jobs': {'queue': 'cpu_tasks'},
    'ivanna_model_manager.tasks.sync_model_statuses_task': {'queue': 'cpu_tasks'},
    'ivanna_model_manager.tasks.refresh_model_ttl': {'queue': 'cpu_tasks'},
    'ivanna_model_manager.tasks.health_check_task': {'queue': 'cpu_tasks'},
    'ivanna_model_manager.tasks.process_uploaded_dataset': {'queue': 'cpu_tasks'},
    'ivanna_model_manager.tasks.export_evaluation_results': {'queue': 'cpu_tasks'},
    'ivanna_model_manager.tasks.build_faiss_index_task': {'queue': 'cpu_tasks'},
    'ivanna_model_manager.tasks.update_faiss_index_task': {'queue': 'cpu_tasks'},
    'ivanna_model_manager.tasks.regenerate_document_chunks_task': {'queue': 'cpu_tasks'},
    'ivanna_model_manager.tasks.rebuild_collection_index_task': {'queue': 'cpu_tasks'},
}

CELERY_TASK_DEFAULT_RETRY_DELAY = 30
CELERY_TASK_MAX_RETRIES = 3

# Периодические задачи (Celery Beat)
CELERY_BEAT_SCHEDULE = {
    'sync-model-statuses-every-5-minutes': {
        'task': 'ivanna_model_manager.tasks.sync_model_statuses_task',
        'schedule': crontab(minute='*/5'),
    },
    'refresh-model-ttl-every-5-minutes': {
        'task': 'ivanna_model_manager.tasks.refresh_model_ttl',
        'schedule': 300.0,
    },
    'health-check-every-15-minutes': {
        'task': 'ivanna_model_manager.tasks.health_check_task',
        'schedule': 900.0,
    },
    'cleanup-download-jobs-daily': {
        'task': 'ivanna_model_manager.tasks.cleanup_download_jobs',
        'schedule': crontab(minute=0, hour=3),
    },
    'update-collection-statistics-hourly': {
        'task': 'ivanna_model_manager.tasks.update_collection_statistics_task',
        'schedule': crontab(minute=0),
    },
    'validate-documents-daily': {
        'task': 'ivanna_model_manager.tasks.validate_documents_task',
        'schedule': crontab(minute=0, hour=2),
    },
    'cleanup-vector-index-daily': {
        'task': 'ivanna_model_manager.tasks.cleanup_vector_index_task',
        'schedule': crontab(minute=0, hour=2),
    },
    'check-system-health-hourly': {
        'task': 'ivanna_model_manager.tasks.check_system_health_task',
        'schedule': crontab(minute=30, hour='*/1'),
    },
    'generate-system-report-weekly': {
        'task': 'ivanna_model_manager.tasks.generate_system_report_task',
        'schedule': crontab(minute=0, hour=9, day_of_week=0),
    },
}

# -------------------------------------------------------------------------
# Django REST Framework
# -------------------------------------------------------------------------
REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
    'DEFAULT_FILTER_BACKENDS': ['django_filters.rest_framework.DjangoFilterBackend'],
}

# -------------------------------------------------------------------------
# Пути для моделей и кэша
# -------------------------------------------------------------------------
MODEL_CACHE_DIR = BASE_DIR / 'media' / 'model_cache'
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

HF_HOME = BASE_DIR / '.cache' / 'huggingface'
HF_HOME.mkdir(parents=True, exist_ok=True)

os.environ['HF_HOME'] = str(HF_HOME)
os.environ['HUGGINGFACE_HUB_CACHE'] = str(HF_HOME / 'hub')
os.environ['HF_DATASETS_CACHE'] = str(HF_HOME / 'datasets')

HUGGINGFACE_HUB_TOKEN = os.environ.get('HF_TOKEN', '')

PYTORCH_CACHE_DIR = BASE_DIR / '.cache' / 'pytorch'
PYTORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('TORCH_HOME', str(PYTORCH_CACHE_DIR))

# -------------------------------------------------------------------------
# RAG / FAISS
# -------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
EMBEDDING_DIMENSION = 384

FAISS_INDEX_PATH = BASE_DIR / 'data' / 'faiss_index'
FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)

VECTOR_STORE_CONFIG = {
    'type': 'faiss',
    'dimension': EMBEDDING_DIMENSION,
    'metric': 'cosine',
    'use_gpu': False,
    'index_path': str(FAISS_INDEX_PATH),
}

# -------------------------------------------------------------------------
# RAG / ПОИСК
# -------------------------------------------------------------------------
KEYWORD_SEARCH_ENGINE = os.getenv('KEYWORD_SEARCH_ENGINE', 'bm25')  # 'bm25' или 'elasticsearch'
BM25_CACHE_SIZE = int(os.getenv('BM25_CACHE_SIZE', '10'))  # максимальное количество коллекций в кэше BM25

# -------------------------------------------------------------------------
# Статические и медиа файлы
# -------------------------------------------------------------------------
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'static'
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

DOCUMENT_UPLOAD_DIR = MEDIA_ROOT / 'documents'
DOCUMENT_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB

ALLOWED_DOCUMENT_EXTENSIONS = {
    '.pdf': 'PDF',
    '.docx': 'DOCX',
    '.doc': 'DOCX',
    '.txt': 'TXT',
    '.html': 'HTML',
    '.htm': 'HTML',
    '.md': 'MARKDOWN',
    '.csv': 'CSV',
    '.xlsx': 'EXCEL',
    '.xls': 'EXCEL',
    '.json': 'JSON',
    '.xml': 'XML',
}

# -------------------------------------------------------------------------
# Ограничения загрузки (Django)
# -------------------------------------------------------------------------
DATA_UPLOAD_MAX_MEMORY_SIZE = MAX_UPLOAD_SIZE

# -------------------------------------------------------------------------
# Админка
# -------------------------------------------------------------------------
ADMIN_SITE_HEADER = "IvAna Lab - Управление LLM моделями"
ADMIN_SITE_TITLE = "IvAna Lab Admin"
ADMIN_INDEX_TITLE = "Панель управления моделями"

# -------------------------------------------------------------------------
# CORS и аутентификация
# -------------------------------------------------------------------------
CORS_ALLOWED_ORIGINS = os.getenv('CORS_ALLOWED_ORIGINS', 'http://localhost:8000,http://127.0.0.1:8000').split(',')
CORS_ALLOW_CREDENTIALS = True

LOGIN_URL = '/accounts/login/'
LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/'

SESSION_COOKIE_AGE = 86400
SESSION_SAVE_EVERY_REQUEST = True

# -------------------------------------------------------------------------
# Логирование
# -------------------------------------------------------------------------
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        # v5.1: request/task id в каждом сообщении — сопоставление логов
        # веб-запросов и Celery-задач через корреляционный ID
        'verbose': {
            'format': '{levelname} {asctime} [{name}] {module} {process:d} {task_id} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} [{name}] [{task_id}] {message}',
            'style': '{',
        },
    },
    'filters': {
        'request_id': {
            '()': 'ivanna_model_manager.log_context.RequestIdFilter',
        },
    },
    'handlers': {
        'console': {
            'filters': ['request_id'],
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
        'file': {
            'filters': ['request_id'],
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'maxBytes': 10 * 1024 * 1024,
            'backupCount': 5,
            'filename': BASE_DIR / 'logs' / 'django.log',
            'formatter': 'verbose',
        },
        'celery_file': {
            'filters': ['request_id'],
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'maxBytes': 10 * 1024 * 1024,
            'backupCount': 5,
            'filename': BASE_DIR / 'logs' / 'celery.log',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True,
        },
        'ivanna_model_manager': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': True,
        },
        'ivanna_model_manager.tasks': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'celery': {
            'handlers': ['console', 'celery_file'],
            'level': 'INFO',
            'propagate': True,
        },
        'transformers': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': False,
        },
        'torch': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': False,
        },
    },
}

LOGS_DIR = BASE_DIR / 'logs'
LOGS_DIR.mkdir(exist_ok=True)