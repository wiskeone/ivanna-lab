from django.contrib import admin
from django.contrib.admin.views.decorators import staff_member_required
from django.shortcuts import render
from django.db.models import Count, Avg, Q
from django.utils import timezone
from datetime import timedelta
from ..models import ModelCheckpoint, ModelMergeJob, InferenceSession


@staff_member_required
def admin_dashboard(request):
    """Custom dashboard page for the admin site."""

    # Model statistics
    model_stats = {
        'total': ModelCheckpoint.objects.count(),
        'active': ModelCheckpoint.objects.filter(is_active=True).count(),
        'loaded': ModelCheckpoint.objects.filter(status='LOADED').count(),
        'today': ModelCheckpoint.objects.filter(
            created_at__date=timezone.now().date()
        ).count(),
    }
    
    # Merge statistics
    merge_stats = {
        'total': ModelMergeJob.objects.count(),
        'running': ModelMergeJob.objects.filter(status='RUNNING').count(),
        'failed': ModelMergeJob.objects.filter(status='FAILED').count(),
        'completed': ModelMergeJob.objects.filter(status='COMPLETED').count(),
    }
    
    # Active sessions
    active_sessions = InferenceSession.objects.filter(
        is_active=True,
        last_activity__gte=timezone.now() - timedelta(minutes=5)
    ).select_related('checkpoint')[:10]
    
    # Recent jobs
    recent_jobs = ModelMergeJob.objects.select_related(
        'base_model', 'merge_model'
    ).order_by('-created_at')[:10]
    
    # Recent models
    recent_models = ModelCheckpoint.objects.select_related(
        'experiment'
    ).order_by('-created_at')[:10]
    
    context = {
        'title': 'Дашборд управления моделями',
        'model_stats': model_stats,
        'merge_stats': merge_stats,
        'active_sessions': active_sessions,
        'recent_jobs': recent_jobs,
        'recent_models': recent_models,
    }
    
    return render(request, 'admin/dashboard.html', context)


# Point the admin index to the custom template
admin.site.index_template = 'admin/custom_index.html'