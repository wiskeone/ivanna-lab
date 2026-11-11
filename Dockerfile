# Базовый образ с CUDA 12.8.1, cuDNN и Ubuntu 24.04
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# 1. Установка Python 3.13 из PPA deadsnakes (Launchpad бывает флакует — ретраи)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    software-properties-common \
    && ok=0; for i in 1 2 3 4 5; do \
         if add-apt-repository -y ppa:deadsnakes/ppa; then ok=1; break; fi; \
         echo "deadsnakes PPA недоступен (попытка $i/5), повтор через 15с"; sleep 15; \
       done; [ "$ok" = "1" ] \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.13 \
        python3.13-dev \
        python3.13-venv \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем pip для Python 3.13
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13

# 2. Системные зависимости (включая компиляторы и библиотеки)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    g++ \
    curl \
    git \
    netcat-openbsd \
    ca-certificates \
    poppler-utils \
    libmagic1 \
    libfreetype6 \
    libjpeg-turbo8 \
    libopenjp2-7 \
    libpng16-16 \
    libtiff6 \
    libxcb1 \
    libxext6 \
    libxrender1 \
    cmake \
    ninja-build \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Таймауты для pip
ENV PIP_DEFAULT_TIMEOUT=6000 \
    PIP_RETRIES=10

# 3. Виртуальное окружение с Python 3.13
RUN python3.13 -m venv /opt/venv
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:$PATH"

# 4. Установка uv
RUN pip install uv

# Увеличиваем таймаут для uv (сетевые задержки)
ENV UV_HTTP_TIMEOUT=6000

# 5. Копируем файл с основными зависимостями (без torch, vllm, mergekit, bitsandbytes)
COPY requirements.in .

# 6. Компилируем и устанавливаем основные зависимости
RUN uv pip compile requirements.in --output-file requirements.txt && \
    uv pip install -r requirements.txt

# 7. Устанавливаем PyTorch и семейство из зеркала Alibaba (CUDA 12.8)
RUN uv pip install \
    --find-links https://mirrors.aliyun.com/pytorch-wheels/cu128/ \
    torch==2.10.0+cu128 \
    torchvision==0.25.0+cu128 \
    torchaudio==2.10.0+cu128 \
    torchcodec==0.10.0+cu128

# 8. bitsandbytes (проверенная версия, работает с CUDA)
RUN uv pip install bitsandbytes==0.49.2

# 9. mergekit (последняя стабильная версия с PyPI)
RUN uv pip install mergekit==0.1.4

# ========== llama.cpp: только Python-скрипт конвертации ==========
# (C++-бинарники проект не вызывает; сборка master сломалась на линковке
# bin/llama — дрейф CMake-таргетов upstream. convert.py в upstream удалён
# (ныне convert_hf_to_gguf.py в корне) — восстанавливаем симлинком.)
RUN git config --global http.postBuffer 524288000 && \
    git config --global http.lowSpeedLimit 0 && \
    git config --global http.lowSpeedTime 999999 && \
    git clone --depth 1 --single-branch https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp && \
    cd /opt/llama.cpp && \
    ln -sf convert_hf_to_gguf.py convert.py

# 10. Копируем код приложения — Django-проект (app/) прямо в /app:
# manage.py и config/ ожидаются там (compose: "cd /app && python manage.py migrate")
COPY app/ .

# 11. Точка входа
RUN echo '#!/bin/bash\nset -e\necho "=== Запуск Celery ==="\nexec "$@"\n' > /entrypoint.sh && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]