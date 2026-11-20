from django.contrib import admin
from django.urls import path, include
from django.views.i18n import set_language
from django.conf import settings
from django.conf.urls.static import static
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from rest_framework import permissions
from django_prometheus import exports

# ============================================================================
# Swagger/OpenAPI documentation
# ============================================================================
schema_view = get_schema_view(
    openapi.Info(
        title="ML Model Lab API",
        default_version='v1',
        description="API for managing language models",
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

# ============================================================================
# Core URLs (always active)
# ============================================================================
urlpatterns = [
    path('i18n/setlang/', set_language, name='set_language'),
    # Django admin panel
    path('admin/', admin.site.urls),

    # Main application
    path('', include('ivanna_model_manager.urls')),

    # API documentation
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0),
         name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0),
         name='schema-redoc'),

    # Health check endpoint (used by Docker healthcheck)
    path('', include('config.health')),  # health.py contains path('health/', ...)

    path('metrics/', exports.ExportToDjangoView, name='prometheus-django-metrics'),
]

# ============================================================================
# URLs for DEBUG mode only (DEBUG=True)
# ============================================================================
if settings.DEBUG:
    # Django Debug Toolbar
    # IMPORTANT: Make sure 'debug_toolbar' is added to INSTALLED_APPS
    # and MIDDLEWARE in config/settings.py when DEBUG=True
    try:
        import debug_toolbar
        urlpatterns += [path('__debug__/', include(debug_toolbar.urls))]
    except ImportError:
        pass  # Skip if the package is not installed

    # Media files served by Django (DEBUG only!)
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
    # Static files (usually served by Nginx in production)
    urlpatterns += static(settings.STATIC_URL,
                          document_root=settings.STATIC_ROOT)