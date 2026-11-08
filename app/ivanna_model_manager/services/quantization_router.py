import logging
import torch
from typing import Optional, Dict, Any, Tuple, List
from transformers import BitsAndBytesConfig
import quanto

logger = logging.getLogger(__name__)


class QuantizationRouter:
    """
    Маршрутизатор квантизации с каскадной системой fallback'ов.

    Стратегия понижения:
    int4 GPU → int4 CPU → int8 GPU → int8 CPU → bfloat16 GPU → float16 GPU → float32 GPU → float32 CPU
    """

    # Иерархия методов квантизации (от самого экономного к самому точному)
    QUANTIZATION_HIERARCHY = [
        ('int4', 'gpu', 1),
        ('int4', 'cpu', 2),
        ('int8', 'gpu', 3),
        ('int8', 'cpu', 4),
        ('bfloat16', 'gpu', 5),
        ('float16', 'gpu', 6),
        ('float32', 'gpu', 7),   # Добавлен float32 на GPU как наиболее точный (без квантизации)
        ('float32', 'cpu', 8),
    ]

    @classmethod
    def get_quantization_config(
        cls,
        requested_quant: Optional[str],
        model_size_gb: float,
        preferred_device: str = 'auto'
    ) -> Tuple[Optional[Any], str, str]:
        """
        Возвращает конфигурацию квантизации, проходя по иерархии fallback'ов.

        Args:
            requested_quant: Запрошенный тип квантизации (int4, int8, float16, bfloat16, None)
            model_size_gb: Размер модели в GB
            preferred_device: Предпочтительное устройство ('auto', 'cuda', 'cpu')

        Returns:
            (config, actual_quantization, target_device)
        """
        available_options = cls._get_available_options(requested_quant)

        logger.info(f"🔍 Starting quantization fallback search for requested={requested_quant}")

        for quant_method, device in available_options:
            logger.info(f"Trying {quant_method} on {device}...")

            success, config, actual_quant, target_device = cls._try_config(
                quant_method, device, model_size_gb
            )

            if success:
                logger.info(f"✅ Success with {actual_quant} on {target_device}")
                return config, actual_quant, target_device

            logger.info(f"❌ {quant_method} on {device} failed, trying next option...")

        logger.warning("All quantization options failed, falling back to CPU float32")
        return None, 'float32', 'cpu'

    @classmethod
    def _get_available_options(cls, requested_quant: Optional[str]) -> List[Tuple[str, str]]:
        """
        Возвращает список опций для попытки на основе запрошенной квантизации.
        """
        option_map = {
            'int4': [
                ('int4', 'gpu'),
                ('int4', 'cpu'),
                ('int8', 'gpu'),
                ('int8', 'cpu'),
                ('bfloat16', 'gpu'),
                ('float16', 'gpu'),
                ('float32', 'gpu'),
                ('float32', 'cpu'),
            ],
            'int8': [
                ('int8', 'gpu'),
                ('int8', 'cpu'),
                ('bfloat16', 'gpu'),
                ('float16', 'gpu'),
                ('float32', 'gpu'),
                ('float32', 'cpu'),
            ],
            'bfloat16': [
                ('bfloat16', 'gpu'),
                ('float16', 'gpu'),
                ('float32', 'gpu'),
                ('float32', 'cpu'),
            ],
            'float16': [
                ('float16', 'gpu'),
                ('float32', 'gpu'),
                ('float32', 'cpu'),
            ],
            None: [  # Без квантизации – сначала пробуем float32 на GPU, затем float16, затем CPU float32
                ('float32', 'gpu'),
                ('float16', 'gpu'),
                ('float32', 'cpu'),
            ],
        }

        if requested_quant not in option_map:
            logger.warning(f"Unknown quantization {requested_quant}, using default options")
            return option_map[None]

        return option_map[requested_quant]

    @classmethod
    def _try_config(
        cls,
        quant_method: str,
        device: str,
        model_size_gb: float
    ) -> Tuple[bool, Optional[Any], str, str]:
        """
        Пытается создать конфигурацию для конкретного метода и устройства.
        Возвращает (успех, конфиг, фактический_метод, устройство).
        """
        try:
            # Проверяем доступность GPU если нужно
            if device == 'gpu':
                if not torch.cuda.is_available():
                    logger.debug(f"GPU not available for {quant_method}")
                    return False, None, quant_method, 'cpu'

                free_memory, total_memory = torch.cuda.mem_get_info()
                free_gb = free_memory / 1024**3

                # Оценка потребности в памяти
                if quant_method in ('int4', 'int8'):
                    required_gb = model_size_gb * (0.5 if quant_method == 'int4' else 1.0)
                else:
                    required_gb = model_size_gb

                # Добавляем запас 20%
                required_gb_with_buffer = required_gb * 1.2

                if free_gb < required_gb_with_buffer:
                    logger.debug(f"Not enough GPU memory for {quant_method}: need {required_gb_with_buffer:.2f}GB, have {free_gb:.2f}GB")
                    return False, None, quant_method, 'cpu'

            # Создаём конфигурацию в зависимости от метода
            if quant_method in ('int4', 'int8') and device == 'gpu':
                config = cls._get_gpu_config(quant_method)
                return True, config, quant_method, 'cuda'

            elif quant_method in ('int4', 'int8') and device == 'cpu':
                config = {'method': 'quanto', 'quantization': quant_method}
                return True, config, f"quanto_{quant_method}", 'cpu'

            elif quant_method == 'float32' and device == 'gpu':
                # Обычная загрузка на GPU в float32 (без квантизации)
                return True, None, 'float32', 'cuda'

            elif quant_method == 'float32' and device == 'cpu':
                return True, None, 'float32', 'cpu'

            elif quant_method in ('float16', 'bfloat16') and device == 'gpu':
                # Обычная загрузка на GPU с указанным типом
                return True, None, quant_method, 'cuda'

            return False, None, quant_method, device

        except Exception as e:
            logger.debug(f"Error trying {quant_method} on {device}: {e}")
            return False, None, quant_method, device

    @staticmethod
    def _get_gpu_config(quantization: str) -> Optional[BitsAndBytesConfig]:
        """Конфигурация для bitsandbytes на GPU."""
        if quantization == 'int4':
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        elif quantization == 'int8':
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True  # <-- добавляем эту строку
            )
        return None