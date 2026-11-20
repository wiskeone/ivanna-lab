import json
import pickle
import time
import logging
from typing import Optional, Dict, Any, List, Union
import redis
import os
import socket
import torch
from threading import Lock

logger = logging.getLogger(__name__)


class RedisModelManager:
    """
    Distributed manager for loaded models.
    - Stores info about loaded models in Redis (keys model:<id>:info).
    - Maintains a local cache of models loaded in the current process (local_models).
    - Provides methods for registering load start, progress, completion and unloading.
    - Single entry point for text generation (checks local cache, otherwise reports the model is loaded in another process).
    """

    def __init__(self, redis_host='redis', redis_port=6379, redis_db=0):
        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=False,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True
        )
        # Local cache of loaded models
        self.local_models: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()  # for thread safety
        self.process_info = {
            'pid': os.getpid(),
            'host': socket.gethostname(),
            'service': 'model_manager'
        }
        # Check Redis connection (non-blocking: imports must not fail without redis
        # — CI tests and local Django commands; actual work will fail with a clear
        # error on the first operation, redis-py reconnects on its own)
        try:
            self.redis.ping()
            logger.info(f"✅ RedisModelManager ready. PID: {self.process_info['pid']}, Host: {self.process_info['host']}")
        except Exception as e:
            logger.error(f"⚠️ Redis unavailable at startup (will retry on operations): {e}")

    # ----------------------------------------------------------------------
    # Loading status management (Redis)
    # ----------------------------------------------------------------------
    def register_model_loading(self, model_id: str, config: Dict[str, Any],
                               process_type: str = 'web') -> bool:
        """
        Register the start of model loading. Acquires a lock.
        Returns True if the lock was acquired.
        """
        lock_key = f"model:{model_id}:lock"
        status_key = f"model:{model_id}:status"

        # Try to acquire the lock (SETNX with TTL)
        lock_data = pickle.dumps({
            'process': self.process_info,
            'started_at': time.time(),
            'process_type': process_type
        })
        lock_acquired = self.redis.set(lock_key, lock_data, nx=True, ex=300)  # 5 minutes

        if lock_acquired:
            # Save the loading status
            self.redis.setex(status_key, 300, json.dumps({
                'status': 'LOADING',
                'model_id': model_id,
                'process': self.process_info,
                'started_at': time.time(),
                'config': config,
                'progress': 0,
                'current_step': 'starting'
            }))
            logger.info(f"🔒 Lock acquired for model {model_id}")
            return True
        else:
            # Check if the lock is stale
            stale_lock = self.redis.get(lock_key)
            if stale_lock:
                try:
                    lock_info = pickle.loads(stale_lock)
                    if time.time() - lock_info.get('started_at', 0) > 300:
                        self.redis.delete(lock_key)
                        logger.warning(f"🧹 Removed stale lock for model {model_id}")
                        return self.register_model_loading(model_id, config, process_type)
                except Exception:
                    pass
            logger.info(f"⚠️ Model {model_id} already being loaded by another process")
            return False

    def update_loading_progress(self, model_id: str, progress: int, step: str = "") -> None:
        """Update model loading progress in Redis."""
        status_key = f"model:{model_id}:status"
        status_data = self.redis.get(status_key)
        if status_data:
            try:
                status = json.loads(status_data)
                status['progress'] = progress
                if step:
                    status['current_step'] = step
                status['updated_at'] = time.time()
                self.redis.setex(status_key, 300, json.dumps(status))
                logger.debug(f"📊 {model_id} progress: {progress}% - {step}")
            except Exception as e:
                logger.error(f"Error updating progress: {e}")

    def register_model_error(self, model_id: str, error_message: str) -> None:
        """Register a model loading error."""
        status_key = f"model:{model_id}:status"
        self.redis.setex(status_key, 300, json.dumps({
            'status': 'ERROR',
            'model_id': model_id,
            'error': error_message,
            'error_time': time.time()
        }))
        lock_key = f"model:{model_id}:lock"
        self.redis.delete(lock_key)
        logger.error(f"❌ Registered error for {model_id}: {error_message}")

    def complete_model_loading(self, model_id: str, model_info: Dict[str, Any]) -> bool:
        """
        Complete model loading: store full info in Redis,
        remove the lock and the LOADING status.
        """
        try:
            info_key = f"model:{model_id}:info"
            status_key = f"model:{model_id}:status"
            lock_key = f"model:{model_id}:lock"

            full_info = {
                **model_info,
                'loaded': True,
                'loaded_at': time.time(),
                'loaded_by': self.process_info,
                'last_accessed': time.time()
            }
            # TTL 1 hour, automatically extended on access
            self.redis.setex(info_key, 3600, pickle.dumps(full_info))
            # Remove the LOADING status
            self.redis.delete(status_key)
            # Release the lock
            self.redis.delete(lock_key)

            logger.info(f"✅ Model {model_id} loading completed, info stored in Redis")
            return True
        except Exception as e:
            logger.error(f"❌ Error completing model loading: {e}")
            return False

    def unload_model(self, model_id: str, force: bool = False) -> bool:
        """
        Fully unload the model and clean up all associated memory
        """
        import gc
        import torch
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"🗑️ Starting unload of model {model_id}, force={force}")

        # 1. Remove from local cache with full cleanup
        with self._lock:
            if model_id in self.local_models:
                model_data = self.local_models.pop(model_id)

                # Remove all references to the model and related objects
                for key in ['model', 'tokenizer', 'pipeline']:
                    if key in model_data:
                        obj = model_data[key]
                        # If a close method (or similar) exists, call it
                        if hasattr(obj, 'close'):
                            try:
                                obj.close()
                            except:
                                pass
                        # Drop the reference
                        del model_data[key]

                logger.info(f"✅ Model {model_id} removed from local cache")

        # 2. Force garbage collection
        gc.collect()
        logger.info(f"🧹 Garbage collection done")

        # 3. Clear CUDA cache (if GPU is used)
        if torch.cuda.is_available():
            # Clear the cache
            torch.cuda.empty_cache()
            # Synchronize to be safe
            torch.cuda.synchronize()

            # Additionally: reset peak memory stats
            torch.cuda.reset_peak_memory_stats()

            # Log current memory state
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"🎯 CUDA memory after cleanup: allocated={allocated:.2f}GB, cached={cached:.2f}GB")

        # 4. Remove from Redis
        try:
            # Remove all keys associated with the model
            keys_to_delete = [
                f"model:{model_id}:info",
                f"model:{model_id}:status",
                f"model:{model_id}:lock",
            ]
            
            for key in keys_to_delete:
                self.redis.delete(key)
            
            # Remove by pattern (in case of other keys)
            pattern_keys = self.redis.keys(f"model:{model_id}:*")
            if pattern_keys:
                for key in pattern_keys:
                    self.redis.delete(key)

            logger.info(f"✅ All Redis keys for {model_id} removed")

        except Exception as e:
            logger.error(f"❌ Error removing from Redis: {e}")

        # 5. One more GC pass to be sure
        gc.collect()

        logger.info(f"✅ Model {model_id} fully unloaded")
        return True

    # ----------------------------------------------------------------------
    # Getting model information
    # ----------------------------------------------------------------------
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get info about a loaded model.
        Checks the local cache first, then Redis.
        If the model is loaded in another process, returns info from Redis.
        """
        # 1. Check the local cache
        if model_id in self.local_models:
            local_data = self.local_models[model_id]
            info = {
                'model_id': model_id,
                'loaded': True,
                'device': local_data.get('device_info', 'unknown'),
                'parameters': local_data.get('parameters', 0),
                'loaded_at': local_data.get('loaded_at'),
                'loaded_by': self.process_info,
                'last_accessed': time.time(),
                'is_local': True,
                'gpu_memory': local_data.get('gpu_memory', {})
            }
            # Update last_accessed in Redis
            self._update_last_accessed(model_id)
            return info

        # 2. Check Redis
        info_key = f"model:{model_id}:info"
        info_data = self.redis.get(info_key)
        if info_data:
            try:
                info = pickle.loads(info_data)
                info['last_accessed'] = time.time()
                # Extend the TTL
                self.redis.expire(info_key, 3600)
                return info
            except Exception as e:
                logger.error(f"❌ Error unpickling model info: {e}")
                return None
        return None

    def get_loading_status(self, model_id: str) -> Dict[str, Any]:
        """Get model loading status (LOADING, ERROR or NOT_FOUND)."""
        status_key = f"model:{model_id}:status"
        status_data = self.redis.get(status_key)
        if status_data:
            try:
                return json.loads(status_data)
            except Exception:
                return {'status': 'ERROR', 'error': 'Invalid status data'}
        # Check if the model is already loaded
        if self.get_model_info(model_id):
            return {'status': 'LOADED', 'model_id': model_id}
        return {'status': 'NOT_FOUND', 'model_id': model_id}

    def list_loaded_models(self) -> List[str]:
        """Get the list of model_id for all models registered in Redis."""
        keys = self.redis.keys("model:*:info")
        models = []
        for key in keys:
            try:
                # key: b'model:some_id:info'
                model_id = key.decode('utf-8').split(':')[1]
                models.append(model_id)
            except Exception:
                continue
        return models

    def get_all_model_infos(self) -> List[Dict[str, Any]]:
        """Get info about all loaded models (from Redis)."""
        models = self.list_loaded_models()
        infos = []
        for model_id in models:
            info = self.get_model_info(model_id)
            if info:
                infos.append(info)
        return infos

    def cleanup_stale_models(self, timeout_seconds: int = 3600) -> int:
        """
        Remove from Redis the info about models not accessed for more than timeout_seconds.
        Returns the number of cleaned up entries.
        """
        count = 0
        models = self.list_loaded_models()
        for model_id in models:
            info = self.get_model_info(model_id)
            if info:
                last_accessed = info.get('last_accessed', 0)
                if time.time() - last_accessed > timeout_seconds:
                    self.unload_model(model_id, 'cleanup')
                    count += 1
        logger.info(f"🧹 Cleaned up {count} stale model entries")
        return count

    # ----------------------------------------------------------------------
    # Local model registration (after successful load in the current process)
    # ----------------------------------------------------------------------
    def register_local_model(self, model_id: str, model: Any, tokenizer: Any,
                             config: Dict[str, Any], device_info: str,
                             pipeline: Any = None, quantization: Optional[str] = None) -> Dict[str, Any]:
        """
        Store the loaded model in the local cache and register it in Redis.

        Args:
            model_id: Model identifier
            model: PyTorch model object
            tokenizer: Tokenizer
            config: Model configuration
            device_info: Device info ('cuda', 'cpu', etc.)
            pipeline: Generation pipeline (optional)
            quantization: Actual quantization type (e.g. 'quanto_int4', 'int4', etc.)

        Returns:
            Dict with model info
        """
        with self._lock:
            # Collect model info
            parameters = 0
            try:
                parameters = sum(p.numel() for p in model.parameters())
            except Exception as e:
                logger.warning(f"Could not count parameters: {e}")

            gpu_memory = {}
            if torch.cuda.is_available() and device_info == 'cuda':
                try:
                    gpu_memory = {
                        'allocated': torch.cuda.memory_allocated() / 1024**3,
                        'reserved': torch.cuda.memory_reserved() / 1024**3,
                        'free': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3
                    }
                except Exception as e:
                    logger.warning(f"Could not get GPU memory info: {e}")

            # Determine the exact model device
            device_str = str(next(model.parameters()).device) if parameters > 0 else device_info

            local_data = {
                'model': model,
                'tokenizer': tokenizer,
                'pipeline': pipeline,
                'config': config,
                'device_info': device_info,
                'device': device_str,
                'parameters': parameters,
                'gpu_memory': gpu_memory,
                'loaded_at': time.time(),
                'quantization': quantization,
            }
            self.local_models[model_id] = local_data

        model_info = {
            'model_id': model_id,
            'loaded': True,
            'device': device_info,
            'device_detail': device_str,
            'quantization': quantization,
            'parameters': parameters,
            'parameters_human': f"{parameters:,}",
            'config': config,
            'loaded_at': local_data['loaded_at'],
            'loaded_by': self.process_info,
            'gpu_memory': gpu_memory,
            'is_local': True,
            'has_pipeline': pipeline is not None,
        }
        
        self.complete_model_loading(model_id, model_info)
        logger.info(f"✅ Registered local model: {model_id} on {device_info} (quant={quantization})")
        return model_info

    # ----------------------------------------------------------------------
    # Text generation
    # ----------------------------------------------------------------------
    def generate_text(self, model_id: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Text generation with improved diagnostics.
        If the model is not loaded locally, returns an error with the need_local_load flag.
        """
        import os
        current_pid = os.getpid()
        logger.info(f"🔧 generate_text called for {model_id} in process {current_pid}")

        # 1. Check the local cache
        if model_id in self.local_models:
            return self._generate_local(model_id, prompt, **kwargs)

        # 2. Check Redis — the model may be loaded in another process
        redis_info = self.get_model_info(model_id)
        if redis_info and redis_info.get('loaded'):
            loaded_by = redis_info.get('loaded_by', {})
            return {
                'success': False,
                'error': f'Модель загружена в другом процессе (PID: {loaded_by.get("pid")}). Нажмите "Загрузить локально" для использования в этом окне.',
                'model_id': model_id,
                'current_pid': current_pid,
                'loaded_pid': loaded_by.get('pid'),
                'loaded_host': loaded_by.get('host'),
                'need_local_load': True,   # single flag for the frontend
            }

        # 3. Model is not loaded anywhere
        return {
            'success': False,
            'error': f'Модель {model_id} не загружена в память. Загрузите модель через интерфейс.',
            'model_id': model_id,
            'need_local_load': True,       # also indicate that loading is needed
        }

    def _generate_local(self, model_id: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Text generation for a locally loaded model."""
        local_data = self.local_models.get(model_id)
        if not local_data:
            return {'success': False, 'error': 'Model not in local cache'}

        model = local_data['model']
        tokenizer = local_data['tokenizer']
        pipeline = local_data.get('pipeline')
        device_info = local_data.get('device_info', 'cpu')
        device = next(model.parameters()).device

        # Set the number of threads for CPU
        if device.type == 'cpu':
            num_threads = 7
            torch.set_num_threads(num_threads)
            logger.info(f"Set PyTorch threads to {num_threads} for CPU generation (model {model_id})")

        max_tokens = kwargs.get('max_tokens', 512)
        temperature = kwargs.get('temperature', 0.7)
        top_p = kwargs.get('top_p', 0.9)
        top_k = kwargs.get('top_k', 50)
        repetition_penalty = kwargs.get('repetition_penalty', 1.0)

        start_time = time.time()

        try:
            if pipeline:
                logger.info(f"Starting generation via pipeline on {device_info} (max_new_tokens={max_tokens})")
                result = pipeline(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=max(temperature, 0.1),
                    top_p=top_p,
                    top_k=top_k if top_k > 0 else None,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_full_text=False,
                )
                gen_time = time.time() - start_time
                logger.info(f"Pipeline generation completed in {gen_time:.2f}s")

                response = result[0]['generated_text'].strip()
                generation_time = gen_time
                input_tokens = len(tokenizer.encode(prompt))
                output_tokens = len(tokenizer.encode(response))
            else:
                # Direct generation via the model
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                # Move tensors to the model device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                generation_kwargs = {
                    'max_new_tokens': max_tokens,
                    'do_sample': True,
                    'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
                    'eos_token_id': tokenizer.eos_token_id,
                }
                if temperature > 0:
                    generation_kwargs['temperature'] = temperature
                if top_p > 0:
                    generation_kwargs['top_p'] = top_p
                if top_k > 0:
                    generation_kwargs['top_k'] = top_k
                if repetition_penalty >= 1.0:
                    generation_kwargs['repetition_penalty'] = repetition_penalty

                logger.info(f"Starting direct generation on {device} (max_new_tokens={max_tokens})")
                gen_start = time.time()
                with torch.no_grad():
                    outputs = model.generate(**inputs, **generation_kwargs)
                gen_time = time.time() - gen_start
                logger.info(f"Direct generation completed in {gen_time:.2f}s, output tokens: {outputs.shape[1] - inputs['input_ids'].shape[1]}")

                generation_time = time.time() - start_time
                response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                input_tokens = inputs['input_ids'].shape[1]
                output_tokens = outputs.shape[1] - input_tokens

            return {
                'success': True,
                'response': response,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
                'generation_time': generation_time,
                'tokens_per_second': output_tokens / generation_time if generation_time > 0 else 0,
                'device': device_info,
            }

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM in generation for {model_id}: {e}. Moving model to CPU and retrying.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self._move_to_cpu_and_retry(model_id, model, tokenizer, pipeline, local_data, prompt, **kwargs)

        except RuntimeError as e:
            err_msg = str(e).lower()
            if "cublas" in err_msg or "cuda" in err_msg or "cudnn" in err_msg:
                logger.error(f"CUDA-related error in generation for {model_id}: {e}. Moving model to CPU and retrying.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return self._move_to_cpu_and_retry(model_id, model, tokenizer, pipeline, local_data, prompt, **kwargs)
            else:
                raise

        except Exception as e:
            logger.error(f"Error generating text: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _move_to_cpu_and_retry(self, model_id, model, tokenizer, pipeline, local_data, prompt, **kwargs):
        logger.info(f"🔄 Moving model {model_id} to CPU for retry...")

        # 1. Move the model to CPU
        model.to('cpu')

        # 2. Determine the quantization type
        quant = local_data.get('quantization', '')

        # 3. If the model is not quantized (float16/float32), convert to float32 for stability
        if 'quanto' not in quant and 'int' not in quant:
            model = model.float()
            logger.info(f"Converted non-quantized model to float32 on CPU (was {quant})")
        else:
            logger.info(f"Keeping quantization {quant} on CPU, no conversion")

        # 4. Remove a possible device_map
        if hasattr(model, 'hf_device_map'):
            model.hf_device_map = None

        # 5. Drop the pipeline
        if pipeline:
            logger.info("Removing old pipeline, will use direct generation on CPU")
        local_data['pipeline'] = None
        local_data['device_info'] = 'cpu'
        self.local_models[model_id] = local_data

        # 6. Clear the CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("Cleared CUDA cache after moving model to CPU")

        logger.info(f"✅ Model {model_id} moved to CPU with {quant}, retrying generation")

        return self._generate_local(model_id, prompt, **kwargs)

    # ----------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------
    def _update_last_accessed(self, model_id: str) -> None:
        """Update the last access time in Redis."""
        info_key = f"model:{model_id}:info"
        info_data = self.redis.get(info_key)
        if info_data:
            try:
                info = pickle.loads(info_data)
                info['last_accessed'] = time.time()
                self.redis.setex(info_key, 3600, pickle.dumps(info))
            except Exception as e:
                logger.error(f"❌ Error updating last accessed: {e}")

    def get_available_memory(self) -> Dict[str, Any]:
        """Get info about available memory (GPU/CPU)."""
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                return {
                    'device': 'cuda',
                    'total_gb': props.total_memory / 1024**3,
                    'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                    'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                    'free_gb': (props.total_memory - torch.cuda.memory_reserved()) / 1024**3,
                }
            except Exception:
                pass
        # CPU fallback
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                'device': 'cpu',
                'total_gb': mem.total / 1024**3,
                'available_gb': mem.available / 1024**3,
                'percent': mem.percent,
            }
        except ImportError:
            return {'device': 'unknown', 'error': 'psutil not installed'}

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        loaded_models = self.list_loaded_models()
        return {
            'loaded_models_count': len(loaded_models),
            'loaded_models': loaded_models,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'memory_info': self.get_available_memory(),
            'redis_connected': True,
            'timestamp': time.time(),
        }

    def clean_stale_locks(self, max_age_seconds=300):
        """Remove locks older than max_age_seconds."""
        import pickle
        keys = self.redis.keys("model:*:lock")
        cleaned = 0
        for key in keys:
            lock_data = self.redis.get(key)
            if lock_data:
                try:
                    lock_info = pickle.loads(lock_data)
                    if time.time() - lock_info.get('started_at', 0) > max_age_seconds:
                        self.redis.delete(key)
                        logger.warning(f"Removed stale lock: {key}")
                        cleaned += 1
                except:
                    self.redis.delete(key)
                    cleaned += 1
        if cleaned:
            logger.info(f"Cleaned {cleaned} stale locks")
        return cleaned


# Global instance (singleton)
redis_model_manager = RedisModelManager()