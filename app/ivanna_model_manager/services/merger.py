import os
import logging
import subprocess
import tempfile
import json
import torch
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelMerger:
    """Model merging service using mergekit"""

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
        Merge two models using mergekit.

        Args:
            base_model_path: Path to the base model (local)
            merge_model_path: Path to the model to merge (local)
            output_path: Path to save the result
            method: Merge method ('linear', 'slerp', 'ties', 'dare')
            merge_ratio: Merge ratio (0.0-1.0). Ignored for TIES/DARE.
            density: Density for TIES/DARE (0.0-1.0)
            **kwargs: Additional parameters (passed to mergekit)

        Returns:
            Dict with the result: {'success': bool, 'output_path': str, 'message': str}
        """
        logger.info(f"Starting model merge: {base_model_path} + {merge_model_path} using {method}")

        # Create a temporary mergekit config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            config_yaml = self._build_merge_config(
                base_model_path, merge_model_path, method, merge_ratio, density, **kwargs
            )
            tmp.write(config_yaml)
            tmp_path = tmp.name

        try:
            # Run mergekit
            cmd = [
                'mergekit-yaml',
                tmp_path,
                output_path,
                '--copy-tokenizer',
                '--allow-crimes',  # allows certain operations, e.g. merging different architectures
            ]
            # If a GPU is available, --cuda can be added
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
            # Remove the temporary file
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
        Build the mergekit YAML configuration.
        """
        # Base config
        config = {
            'merge_method': method,
            'base_model': base_path,
            'models': [
                {'model': base_path, 'parameters': {'weight': 1.0 - ratio}},
                {'model': merge_path, 'parameters': {'weight': ratio}}
            ],
            'dtype': 'bfloat16',
        }

        # TIES and DARE require additional settings
        if method in ('ties', 'dare'):
            config['parameters'] = {
                'density': density,
                'normalize': True
            }
            # Additional parameters can be set for TIES
            if method == 'ties':
                config['parameters']['merge_mechanism'] = 'ties'

        # Add extra parameters from kwargs: scalar keys
        # (dtype etc.) are overridden by the caller, structural ones are protected
        protected = ('merge_method', 'base_model', 'models', 'parameters')
        for key, value in kwargs.items():
            if key not in protected:
                config[key] = value

        # Convert to YAML
        import yaml
        return yaml.dump(config, default_flow_style=False)

    def validate_merge_config(self, config: Dict) -> bool:
        """Check that the config contains the required fields."""
        required = ['base_model', 'merge_model', 'output_path']
        for field in required:
            if field not in config:
                logger.error(f"Missing required field: {field}")
                return False
        return True

    def estimate_merge_resources(self, model_size_gb: float) -> Dict[str, float]:
        """Estimate resources."""
        return {
            'estimated_memory_gb': round(model_size_gb * 2.5, 2),
            'estimated_disk_gb': round(model_size_gb * 1.2, 2),
            'estimated_time_minutes': model_size_gb * 5
        }


# Global instance
merger = ModelMerger()
ModelMergerService = ModelMerger