import logging
import time
from threading import Lock
from typing import Dict, Optional, Tuple, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..models import ModelCheckpoint  # fixed: moved up one level
from .quantization_router import QuantizationRouter  # quantization_router in the same folder
from .redis_model_service import redis_model_manager  # redis_model_service in the same folder

logger = logging.getLogger(__name__)

class ModelCache:
    """Single cache of loaded models shared by all Celery tasks."""
    
    def __init__(self):
        self._cache: Dict[int, Tuple[AutoModelForCausalLM, AutoTokenizer, Dict[str, Any]]] = {}
        self._lock = Lock()
    
    def get_model(self, model_id: int, quantization: Optional[str] = None) -> Optional[Tuple[AutoModelForCausalLM, AutoTokenizer]]:
        """
        Get a model from the cache or load it.
        Returns (model, tokenizer) or (None, None) on error.
        """
        with self._lock:
            if model_id in self._cache:
                model, tokenizer, config = self._cache[model_id]
                if config.get('quantization') == quantization:
                    logger.info(f"Using cached model {model_id} with quant={quantization}")
                    return model, tokenizer
                else:
                    logger.warning(f"Model {model_id} loaded with different quantization, reloading")
                    self._unload_model(model_id)
            
            try:
                checkpoint = ModelCheckpoint.objects.get(id=model_id, is_active=True)
            except ModelCheckpoint.DoesNotExist:
                logger.error(f"Model {model_id} not found")
                return None, None

            model_identifier = checkpoint.huggingface_id or checkpoint.name

            if not redis_model_manager.register_model_loading(model_identifier, checkpoint.loader_config):
                logger.error(f"Model {model_identifier} already loading elsewhere")
                return None, None

            try:
                model_size_gb = checkpoint.size_gb
                quant_config, actual_quant, target_device = QuantizationRouter.get_quantization_config(
                    quantization, model_size_gb
                )

                tokenizer = AutoTokenizer.from_pretrained(
                    checkpoint.local_path,
                    trust_remote_code=True,
                    use_fast=True
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                if isinstance(quant_config, dict) and quant_config.get('method') == 'quanto':
                    # Use the function from tasks.py; to avoid a circular import it should
                    # be moved to a separate module. For now keep it simple: a direct
                    # import (which may still create a cycle).
                    from ..tasks import _load_model_with_quanto  # import from tasks
                    model, tokenizer = _load_model_with_quanto(
                        checkpoint.local_path,
                        quantization=quant_config['quantization']
                    )
                else:
                    model_kwargs = {
                        'trust_remote_code': True,
                        'low_cpu_mem_usage': True,
                    }
                    if target_device == 'cuda':
                        model_kwargs['device_map'] = 'auto'
                        if quant_config:
                            model_kwargs['quantization_config'] = quant_config
                        else:
                            if actual_quant == 'float16':
                                model_kwargs['torch_dtype'] = torch.float16
                            elif actual_quant == 'bfloat16':
                                model_kwargs['torch_dtype'] = torch.bfloat16
                    else:
                        model_kwargs['torch_dtype'] = torch.float32

                    model = AutoModelForCausalLM.from_pretrained(
                        checkpoint.local_path,
                        **model_kwargs
                    )

                config_info = {
                    'quantization': actual_quant,
                    'device': target_device,
                    'loaded_at': time.time()
                }
                self._cache[model_id] = (model, tokenizer, config_info)

                redis_model_manager.complete_model_loading(model_identifier, {
                    'model_id': model_identifier,
                    'loaded': True,
                    'device': target_device,
                    'quantization': actual_quant,
                })

                logger.info(f"✅ Model {model_id} loaded with quant={actual_quant} on {target_device}")
                return model, tokenizer

            except Exception as e:
                logger.error(f"Error loading model {model_id}: {e}", exc_info=True)
                redis_model_manager.register_model_error(model_identifier, str(e))
                return None, None

    def _unload_model(self, model_id: int):
        with self._lock:
            if model_id in self._cache:
                del self._cache[model_id]
                logger.info(f"Unloaded model {model_id} from cache")

    def clear(self):
        with self._lock:
            self._cache.clear()
            logger.info("Model cache cleared")

# Global instance
model_cache = ModelCache()