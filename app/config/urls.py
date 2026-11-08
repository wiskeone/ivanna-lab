from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from rest_framework import permissions
from django_prometheus import exports

# ============================================================================
# Swagger/OpenAPI документация
# ============================================================================
schema_view = get_schema_view(
    openapi.Info(
        title="ML Model Lab API",
        default_version='v1',
        description="API для управления языковыми моделями",
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

# ============================================================================
# Основные URL-адреса (работают всегда)
# ============================================================================
urlpatterns = [
    # Административная панель Django
    path('admin/', admin.site.urls),
    
    # Основное приложение
    path('', include('ivanna_model_manager.urls')),
    
    # Документация API
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), 
         name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), 
         name='schema-redoc'),
    
    # Health check endpoint (используется Docker healthcheck)
    path('', include('config.health')),  # health.py содержит path('health/', ...)

    path('metrics/', exports.ExportToDjangoView, name='prometheus-django-metrics'),
]

# ============================================================================
# URL-адреса ТОЛЬКО для режима отладки (DEBUG=True)
# ============================================================================
if settings.DEBUG:
    # Django Debug Toolbar (панель отладки)
    # ВАЖНО: Убедитесь, что 'debug_toolbar' добавлен в INSTALLED_APPS 
    # и MIDDLEWARE в config/settings.py при DEBUG=True
    try:
        import debug_toolbar
        urlpatterns += [path('__debug__/', include(debug_toolbar.urls))]
    except ImportError:
        pass  # Если пакет не установлен, просто пропускаем
    
    # Обслуживание медиафайлов разработчиком Django (только для DEBUG!)
    urlpatterns += static(settings.MEDIA_URL, 
                          document_root=settings.MEDIA_ROOT)
    # Обслуживание статических файлов (обычно через Nginx в production)
    urlpatterns += static(settings.STATIC_URL, 
                          document_root=settings.STATIC_ROOT)