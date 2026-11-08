#!/bin/bash

set -e

echo "Ожидание PostgreSQL..."
while ! nc -z $DB_HOST $DB_PORT; do
  sleep 0.1
done
echo "PostgreSQL доступен"

echo "Ожидание Redis..."
while ! nc -z $REDIS_HOST 6379; do
  sleep 0.1
done
echo "Redis доступен"

echo "Выполнение миграций..."
# makemigrations в прод-старте НЕТ: миграции — часть репозитория.
# Расхождение моделей и миграций ловит CI (makemigrations --check).
# Для разработки: docker exec -it ivanna_web python manage.py makemigrations
python manage.py migrate

echo "Сбор статики..."
python manage.py collectstatic --noinput

# Создание суперпользователя — только через DJANGO_SUPERUSER_* env
# (FIX: раньше молча создавался admin/admin — готовый бэкдор).
# Пример: DJANGO_SUPERUSER_USERNAME=admin DJANGO_SUPERUSER_EMAIL=... DJANGO_SUPERUSER_PASSWORD=...
if [ -n "$DJANGO_SUPERUSER_USERNAME" ] && [ -n "$DJANGO_SUPERUSER_PASSWORD" ]; then
  python manage.py createsuperuser --noinput || true
fi

echo "Запуск сервера..."
exec "$@"