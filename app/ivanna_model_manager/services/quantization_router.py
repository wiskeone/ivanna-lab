import logging
import torch
from typing import Optional, Dict, Any, Tuple, List
from transformers import BitsAndBytesConfig
import quanto

logger = logging.getLogger(__name__)


class QuantizationRouter:
    """
    Quantization router with a cascading fallback system.

    Downgrade strategy:
    int4 GPU → int4 CPU → int8 GPU → int8 CPU → bfloat16 GPU → float16 GPU → float32 GPU → float32 CPU
    """

    # Quantization method hierarchy (from most memory-efficient to most accurate)
    QUANTIZATION_HIERARCHY = [
        ('int4', 'gpu', 1),
        ('int4', 'cpu', 2),
        ('int8', 'gpu', 3),
        ('int8', 'cpu', 4),
        ('bfloat16', 'gpu', 5),
        ('float16', 'gpu', 6),
        ('float32', 'gpu', 7),   # float32 on GPU added as the most accurate option (no quantization)
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
        Return a quantization configuration by walking the fallback hierarchy.

        Args:
            requested_quant: Requested quantization type (int4, int8, float16, bfloat16, None)
            model_size_gb: Model size in GB
            preferred_device: Preferred device ('auto', 'cuda', 'cpu')

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
        Return the list of options to try based on the requested quantization.
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
            None: [  # No quantization — try float32 on GPU first, then float16, then CPU float32
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
        Try to build a configuration for a specific method and device.
        Returns (success, config, actual_method, device).
        """
        try:
            # Check GPU availability if needed
            if device == 'gpu':
                if not torch.cuda.is_available():
                    logger.debug(f"GPU not available for {quant_method}")
                    return False, None, quant_method, 'cpu'

                free_memory, total_memory = torch.cuda.mem_get_info()
                free_gb = free_memory / 1024**3

                # Estimate the memory requirement
                if quant_method in ('int4', 'int8'):
                    required_gb = model_size_gb * (0.5 if quant_method == 'int4' else 1.0)
                else:
                    required_gb = model_size_gb

                # Add a 20% buffer
                required_gb_with_buffer = required_gb * 1.2

                if free_gb < required_gb_with_buffer:
                    logger.debug(f"Not enough GPU memory for {quant_method}: need {required_gb_with_buffer:.2f}GB, have {free_gb:.2f}GB")
                    return False, None, quant_method, 'cpu'

            # Build the configuration depending on the method
            if quant_method in ('int4', 'int8') and device == 'gpu':
                config = cls._get_gpu_config(quant_method)
                return True, config, quant_method, 'cuda'

            elif quant_method in ('int4', 'int8') and device == 'cpu':
                config = {'method': 'quanto', 'quantization': quant_method}
                return True, config, f"quanto_{quant_method}", 'cpu'

            elif quant_method == 'float32' and device == 'gpu':
                # Plain load on GPU in float32 (no quantization)
                return True, None, 'float32', 'cuda'

            elif quant_method == 'float32' and device == 'cpu':
                return True, None, 'float32', 'cpu'

            elif quant_method in ('float16', 'bfloat16') and device == 'gpu':
                # Plain load on GPU with the given dtype
                return True, None, quant_method, 'cuda'

            return False, None, quant_method, device

        except Exception as e:
            logger.debug(f"Error trying {quant_method} on {device}: {e}")
            return False, None, quant_method, device

    @staticmethod
    def _get_gpu_config(quantization: str) -> Optional[BitsAndBytesConfig]:
        """bitsandbytes configuration for GPU."""
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
                llm_int8_enable_fp32_cpu_offload=True  # <-- added this line
            )
        return None