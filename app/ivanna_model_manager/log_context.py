"""
v5.1: correlation ID for logs — end-to-end from web to Celery.

Mechanism:
- Web: middleware generates a UUID per request and stores it in the local context
- Celery: before_task_publish stores task_id in headers,
  before_task calls TaskIdFilter, after_task restores it
- Formatter: {task_id} from record — request UUID or Celery task ID

Result: a single ID allows finding all logs produced by one user
action — from HTTP request to background worker.
"""
import logging
import threading
import uuid

logger = logging.getLogger(__name__)

_local = threading.local()


def get_request_id() -> str:
    """Current correlation ID (or '-' if none)."""
    return getattr(_local, 'request_id', '-') or '-'


def set_request_id(rid: str = None):
    """Set correlation ID (middleware / Celery signal)."""
    _local.request_id = rid or '-'


def clear_request_id():
    _local.request_id = None


class RequestIdFilter(logging.Filter):
    """Adds record.task_id — the value for {task_id} in the formatter."""

    def filter(self, record):
        record.task_id = get_request_id()
        # record.name is already available for {name}
        return True


# ═══ Celery signals: pass task_id through the broker ═══

def setup_celery_log_signals(celery_app):
    """
    Connect Celery signals for correlation. Called from config/celery.py.
    """
    from celery.signals import before_task_publish, task_prerun, task_postrun

    @before_task_publish.connect
    def _inject(sender=None, headers=None, **kwargs):
        """Before publishing to broker: embed request_id into headers."""
        if headers is not None:
            rid = get_request_id()
            if rid == '-':
                rid = uuid.uuid4().hex[:12]
            headers['correlation_id'] = rid

    @task_prerun.connect
    def _prerun(task=None, **kwargs):
        """Task received: set correlation_id as the thread request_id."""
        if task and task.request:
            rid = task.request.correlation_id or task.request.id or '-'
            set_request_id(rid)

    @task_postrun.connect
    def _postrun(task=None, **kwargs):
        """Task finished — clean up the thread."""
        clear_request_id()

    logger.info("v5.1: Celery log correlation signals connected")
