#!/usr/bin/env python3
"""
Проверка окружения Docker-контейнера
"""

import os
import sys
import torch
import subprocess
from pathlib import Path

def check_dirs():
    """Проверяем существование директорий"""
    print("📁 Проверка директорий:")
    
    dirs = [
        ('/app', 'Рабочая директория'),
        ('/app/media/model_cache', 'Кэш моделей'),
        ('/root/.cache/huggingface', 'HF кэш'),
        ('/root/.cache/transformers', 'Transformers кэш'),
        ('/root/.cache/torch', 'Torch кэш'),
        ('/app/static', 'Статика'),
        ('/app/media', 'Медиа'),
    ]
    
    for path, desc in dirs:
        p = Path(path)
        exists = p.exists()
        is_dir = p.is_dir() if exists else False
        status = "✅" if exists and is_dir else "❌"
        print(f"  {status} {desc}: {path} {'существует' if exists else 'не существует'}")
        
        if exists and is_dir:
            try:
                # Пробуем создать тестовый файл
                test_file = p / '.write_test'
                test_file.write_text('test')
                test_file.unlink()
                print(f"    ↳ ✅ Доступ на запись")
            except Exception as e:
                print(f"    ↳ ❌ Нет доступа на запись: {e}")

def check_gpu():
    """Проверяем GPU"""
    print("\n🎮 Проверка GPU:")
    
    if torch.cuda.is_available():
        print(f"  ✅ CUDA доступна")
        print(f"  Версия CUDA: {torch.version.cuda}")
        print(f"  Устройство: {torch.cuda.get_device_name(0)}")
        print(f"  Память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Проверяем nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            print(f"  nvidia-smi: доступен")
        except FileNotFoundError:
            print(f"  nvidia-smi: не найден")
    else:
        print("  ❌ CUDA недоступна")

def check_env_vars():
    """Проверяем переменные окружения"""
    print("\n🔧 Проверка переменных окружения:")
    
    env_vars = [
        'MODEL_CACHE_DIR',
        'HUGGINGFACE_HUB_CACHE', 
        'HF_HOME',
        'TORCH_HOME',
        'HF_TOKEN',
        'PYTHONPATH',
        'DJANGO_SETTINGS_MODULE',
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'не установлена')
        status = "✅" if value != 'не установлена' else "⚠️"
        # Маскируем токен
        if var == 'HF_TOKEN' and value != 'не установлена':
            value = f"{value[:10]}..." if len(value) > 10 else "***"
        print(f"  {status} {var}: {value}")

def check_python_packages():
    """Проверяем Python пакеты"""
    print("\n🐍 Проверка Python пакетов:")
    
    packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('accelerate', 'Accelerate'),
        ('bitsandbytes', 'BitsAndBytes'),
        ('huggingface_hub', 'Hugging Face Hub'),
        ('django', 'Django'),
        ('celery', 'Celery'),
    ]
    
    for pkg_name, display_name in packages:
        try:
            __import__(pkg_name)
            version = sys.modules[pkg_name].__version__
            print(f"  ✅ {display_name}: {version}")
        except ImportError:
            print(f"  ❌ {display_name}: не установлен")
        except AttributeError:
            print(f"  ✅ {display_name}: установлен (версия неизвестна)")

def main():
    print("=" * 60)
    print("Ivanna Lab - Проверка окружения Docker")
    print("=" * 60)
    
    check_dirs()
    check_gpu()
    check_env_vars()
    check_python_packages()
    
    print("\n" + "=" * 60)
    print("✅ Проверка завершена")
    
    # Рекомендации
    print("\n💡 Рекомендации:")
    
    if not torch.cuda.is_available():
        print("  • Убедитесь что Docker запущен с флагом --gpus all")
        print("  • Проверьте установку драйверов NVIDIA")
    
    cache_dirs = [
        Path('/root/.cache/huggingface'),
        Path('/root/.cache/transformers'),
        Path('/root/.cache/torch'),
    ]
    
    for dir_path in cache_dirs:
        if dir_path.exists():
            try:
                # Примерный расчет размера
                total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                if total_size > 1e9:  # > 1GB
                    print(f"  • Кэш {dir_path.name}: {total_size/1e9:.1f}GB - очистите при необходимости")
            except:
                pass

if __name__ == "__main__":
    main()