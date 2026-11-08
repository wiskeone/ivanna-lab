from django.http import HttpResponse
from django.urls import path

def health_check(request):
    """Health check для Docker"""
    return HttpResponse('OK', status=200, content_type='text/plain')

urlpatterns = [
    path('health/', health_check, name='health_check'),
]
