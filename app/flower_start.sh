#!/bin/bash
cd /app

# Ждем готовности Redis
until redis-cli -h redis -p 6379 ping | grep -q "PONG"; do
  echo "Waiting for Redis..."
  sleep 5
done

# Ждем готовности воркеров
echo "Waiting for Celery workers..."
sleep 30

# Запускаем Flower с увеличенными таймаутами
celery -A config flower \
  --port=5555 \
  --broker=redis://redis:6379/0 \
  --inspect_timeout=30.0 \
  --persistent=True