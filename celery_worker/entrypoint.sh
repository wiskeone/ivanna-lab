#!/bin/bash

set -e

echo "=========================================="
echo "Инициализация Celery Worker"
echo "=========================================="

# Проверка PostgreSQL
if [ -n "$DB_HOST" ] && [ -n "$DB_PORT" ]; then
    echo "Ожидание PostgreSQL ($DB_HOST:$DB_PORT)..."
    while ! nc -z $DB_HOST $DB_PORT; do
        echo "PostgreSQL недоступен, повторная попытка через 1 сек..."
        sleep 1
    done
    echo "✓ PostgreSQL доступен"
else
    echo "⚠ Предупреждение: переменные DB_HOST или DB_PORT не установлены"
fi

# Проверка Redis
if [ -n "$REDIS_HOST" ]; then
    echo "Ожидание Redis ($REDIS_HOST:6379)..."
    while ! nc -z $REDIS_HOST 6379; do
        echo "Redis недоступен, повторная попытка через 1 сек..."
        sleep 1
    done
    echo "✓ Redis доступен"
else
    echo "⚠ Предупреждение: переменная REDIS_HOST не установлена"
fi

# Проверка PyTorch и CUDA (критически важно для GPU задач)
echo "Проверка окружения PyTorch..."
if python3 -c "import torch; print(f'PyTorch версия: {torch.__version__}')"; then
    echo "✓ PyTorch успешно импортирован"
    
    # Проверка доступности CUDA
    if python3 -c "import torch; print(f'CUDA доступна: {torch.cuda.is_available()}')"; then
        if python3 -c "import torch; print(f'Количество GPU: {torch.cuda.device_count()}')"; then
            python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    print(f'    Память: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB')
"
            echo "✓ CUDA GPU настроены корректно"
        fi
    else
        echo "⚠ ВНИМАНИЕ: CUDA недоступна, но Celery worker будет запущен"
        echo "  Проверьте:"
        echo "  1. Наличие драйверов NVIDIA в контейнере"
        echo "  2. Настройки runtime: nvidia в docker-compose"
        echo "  3. Переменные окружения NVIDIA_VISIBLE_DEVICES"
    fi
else
    echo "✗ ОШИБКА: Не удалось импортировать PyTorch"
    echo "  Проверьте установку PyTorch в Dockerfile"
    exit 1
fi

# Проверка переменных окружения
echo "Проверка критических переменных окружения..."
for var in CELERY_BROKER_URL DB_NAME DB_USER; do
    if [ -z "${!var}" ]; then
        echo "⚠ Предупреждение: переменная $var не установлена"
    else
        if [ "$var" = "DB_PASSWORD" ]; then
            echo "  ✓ $var: установлена (скрыто)"
        else
            echo "  ✓ $var: ${!var}"
        fi
    fi
done

# Применение миграций (если нужно)
echo "Проверка миграций Django..."
if [ -f "manage.py" ]; then
    if [ "$APPLY_MIGRATIONS" = "true" ]; then
        echo "Применение миграций..."
        python3 manage.py migrate --noinput
        echo "✓ Миграции применены"
    else
        echo "⚠ Миграции не применяются (APPLY_MIGRATIONS != true)"
    fi
fi

# Проверка конфигурации Celery
echo "Проверка конфигурации Celery..."
if python3 -c "
import os
from importlib.util import find_spec

if find_spec('config'):
    # Попытка импорта celery приложения
    try:
        from config.celery import app
        print('✓ Celery приложение найдено в config.celery')
        
        # Проверка брокера
        broker_url = app.conf.get('broker_url', 'не установлен')
        print(f'  Broker URL: {broker_url}')
        
        # Проверка результатов
        result_backend = app.conf.get('result_backend', 'не установлен')
        print(f'  Result Backend: {result_backend}')
        
    except Exception as e:
        print(f'✗ Ошибка импорта Celery: {e}')
else:
    print('✗ Модуль config не найден')
"; then
    echo "✓ Конфигурация Celery проверена"
fi

echo "=========================================="
echo "Запуск Celery Worker с параметрами:"
echo "  Логирование: ${CELERY_LOG_LEVEL:-info}"
echo "  Конкурентность: ${CELERY_CONCURRENCY:-2}"
echo "  Очереди: ${CELERY_QUEUES:-default}"
echo "=========================================="

# Расширенная команда запуска с проверкой аргументов
if [ $# -eq 0 ]; then
    # Если аргументы не переданы, используем значения из переменных окружения
    echo "Запуск с параметрами из переменных окружения..."
    exec celery -A config.celery worker \
        --loglevel=${CELERY_LOG_LEVEL:-info} \
        --concurrency=${CELERY_CONCURRENCY:-2} \
        --hostname=worker@%h \
        --queues=${CELERY_QUEUES:-default} \
        --prefetch-multiplier=${CELERY_PREFETCH:-2} \
        --max-tasks-per-child=${CELERY_MAX_TASKS:-100} \
        --without-gossip \
        --without-mingle \
        --pidfile=/tmp/celery.pid
else
    # Если аргументы переданы, используем их
    echo "Запуск с переданными аргументами: $@"
    exec "$@"
fi