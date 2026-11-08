import os
import logging
import subprocess
import tempfile
import json
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelMerger:
    """Сервис слияния моделей с использованием mergekit"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def merge_models(
        self,
        base_model_path: str,
        merge_model_path: str,
        output_path: str,
        method: str = 'linear',
        merge_ratio: float = 0.5,
        density: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Выполняет слияние двух моделей с помощью mergekit.

        Args:
            base_model_path: Путь к базовой модели (локальный)
            merge_model_path: Путь к модели для слияния (локальный)
            output_path: Путь для сохранения результата
            method: Метод слияния ('linear', 'slerp', 'ties', 'dare')
            merge_ratio: Коэффициент слияния (0.0-1.0). Для TIES/DARE игнорируется.
            density: Плотность для TIES/DARE (0.0-1.0)
            **kwargs: Дополнительные параметры (передаются в mergekit)

        Returns:
            Словарь с результатом: {'success': bool, 'output_path': str, 'message': str}
        """
        logger.info(f"Starting model merge: {base_model_path} + {merge_model_path} using {method}")

        # Создаём временный файл конфигурации для mergekit
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            config_yaml = self._build_merge_config(
                base_model_path, merge_model_path, method, merge_ratio, density, **kwargs
            )
            tmp.write(config_yaml)
            tmp_path = tmp.name

        try:
            # Запускаем mergekit
            cmd = [
                'mergekit-yaml',
                tmp_path,
                output_path,
                '--copy-tokenizer',
                '--allow-crimes',  # разрешает некоторые операции, например, слияние разных архитектур
            ]
            # Если есть GPU, можно добавить --cuda
            if torch.cuda.is_available():
                cmd.append('--cuda')

            logger.info(f"Running command: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            output_lines = []
            for line in process.stdout:
                line = line.strip()
                if line:
                    logger.info(f"[mergekit] {line}")
                    output_lines.append(line)

            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"Mergekit exited with code {return_code}. Output: {' '.join(output_lines)}")

            logger.info(f"Merge completed successfully: {output_path}")
            return {
                'success': True,
                'output_path': output_path,
                'message': 'Merge completed successfully',
                'logs': '\n'.join(output_lines)
            }

        except Exception as e:
            logger.error(f"Merge failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'output_path': None,
                'message': f"Merge failed: {e}"
            }
        finally:
            # Удаляем временный файл
            os.unlink(tmp_path)

    def _build_merge_config(
        self,
        base_path: str,
        merge_path: str,
        method: str,
        ratio: float,
        density: float,
        **kwargs
    ) -> str:
        """
        Строит YAML конфигурацию для mergekit.
        """
        # Базовый конфиг
        config = {
            'merge_method': method,
            'base_model': base_path,
            'models': [
                {'model': base_path, 'parameters': {'weight': 1.0 - ratio}},
                {'model': merge_path, 'parameters': {'weight': ratio}}
            ],
            'dtype': 'bfloat16',
        }

        # Для TIES и DARE нужна дополнительная настройка
        if method in ('ties', 'dare'):
            config['parameters'] = {
                'density': density,
                'normalize': True
            }
            # Для TIES можно указать другие параметры
            if method == 'ties':
                config['parameters']['merge_mechanism'] = 'ties'

        # Добавляем дополнительные параметры из kwargs
        for key, value in kwargs.items():
            if key not in config:
                config[key] = value

        # Преобразуем в YAML
        import yaml
        return yaml.dump(config, default_flow_style=False)

    def validate_merge_config(self, config: Dict) -> bool:
        """Проверяет, что в конфиге есть обязательные поля."""
        required = ['base_model', 'merge_model', 'output_path']
        for field in required:
            if field not in config:
                logger.error(f"Missing required field: {field}")
                return False
        return True

    def estimate_merge_resources(self, model_size_gb: float) -> Dict[str, float]:
        """Оценка ресурсов."""
        return {
            'estimated_memory_gb': round(model_size_gb * 2.5, 2),
            'estimated_disk_gb': round(model_size_gb * 1.2, 2),
            'estimated_time_minutes': model_size_gb * 5
        }


# Глобальный экземпляр
merger = ModelMerger()
ModelMergerService = ModelMerger