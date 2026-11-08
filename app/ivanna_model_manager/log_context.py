"""
v5.1: корреляционный ID для логов — сквозной через веб → Celery.

Механизм:
- Веб: middleware генерирует UUID на каждый запрос, кладёт в local-контекст
- Celery: before_task_publish сохраняет task_id в заголовок,
  before_task вызывает TaskIdFilter, after_task_url восстанавливает
- Formatter: {task_id} из record — UUID запроса или ID Celery-задачи

Результат: по одному ID можно найти все логи, порождённые одним
пользовательским действием — от HTTP-запроса до фонового воркера.
"""
import logging
import threading
import uuid

_local = threading.local()


def get_request_id() -> str:
    """Текущий корреляционный ID (или '-', если нет)."""
    return getattr(_local, 'request_id', '-') or '-'


def set_request_id(rid: str = None):
    """Установить корреляционный ID (middleware / Celery signal)."""
    _local.request_id = rid or '-'


def clear_request_id():
    _local.request_id = None


class RequestIdFilter(logging.Filter):
    """Добавляет record.task_id — значение для {task_id} в форматтере."""

    def filter(self, record):
        record.task_id = get_request_id()
        # record.name и так доступен для {name}
        return True


# ═══ Celery-сигналы: проброс task_id через брокер ═══

def setup_celery_log_signals(celery_app):
    """
    Подключить сигналы Celery для корреляции. Вызывается из config/celery.py.
    """
    from celery.signals import before_task_publish, task_prerun, task_postrun

    @before_task_publish.connect
    def _inject(sender=None, headers=None, **kwargs):
        """До отправки в брокер: вшить request_id в заголовки."""
        if headers is not None:
            rid = get_request_id()
            if rid == '-':
                rid = uuid.uuid4().hex[:12]
            headers['correlation_id'] = rid

    @task_prerun.connect
    def _prerun(task=None, **kwargs):
        """Получил задачу: ставим correlation_id как request_id потока."""
        if task and task.request:
            rid = task.request.correlation_id or task.request.id or '-'
            set_request_id(rid)

    @task_postrun.connect
    def _postrun(task=None, **kwargs):
        """Задача закончилась — чистим поток."""
        clear_request_id()

    logger.info("v5.1: Celery log correlation signals connected")
