"""
Weights & Biases logger implementation.
"""

import logging
from typing import Dict, Any, Optional, Union

from .base_logger import BaseLogger

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class WandbLogger(BaseLogger):
    """Weights & Biases logger implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize wandb logger.
        
        Args:
            config: Wandb configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialized = False
        
        if self.is_available() and config.get('enabled', False):
            self._initialize()
    
    def _initialize(self):
        """Initialize wandb session."""
        try:
            # Check if wandb is already initialized
            if wandb.run is not None:
                self.logger.info("Using existing wandb run")
                self._initialized = True
                return
            
            # Initialize wandb
            wandb_init_kwargs = {
                'project': self.config.get('project', 'jepa'),
                'name': self.config.get('name'),
                'entity': self.config.get('entity'),
                'tags': self.config.get('tags'),
                'notes': self.config.get('notes'),
                'config': self.config.get('hyperparameters', {})
            }
            
            # Remove None values
            wandb_init_kwargs = {k: v for k, v in wandb_init_kwargs.items() if v is not None}
            
            wandb.init(**wandb_init_kwargs)
            self._initialized = True
            self.logger.info(f"Wandb initialized: {wandb.run.url}")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self._initialized = False
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None, prefix: str = ""):
        """Log metrics to wandb."""
        if not self._initialized:
            return
        
        try:
            # Add prefix to metrics if provided
            if prefix:
                metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
            
            wandb.log(metrics, step=step)
        except Exception as e:
            self.logger.warning(f"Failed to log metrics to wandb: {e}")
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to wandb."""
        if not self._initialized:
            return
        
        try:
            wandb.config.update(params)
        except Exception as e:
            self.logger.warning(f"Failed to log hyperparameters to wandb: {e}")
    
    def log_artifact(self, file_path: str, name: str, artifact_type: str = "model"):
        """Log artifact to wandb."""
        if not self._initialized or not self.config.get('log_model', True):
            return
        
        try:
            artifact = wandb.Artifact(
                name=name,
                type=artifact_type,
                description=f"JEPA {artifact_type}"
            )
            artifact.add_file(file_path)
            wandb.log_artifact(artifact)
            self.logger.info(f"Artifact '{name}' logged to wandb")
        except Exception as e:
            self.logger.warning(f"Failed to log artifact to wandb: {e}")
    
    def watch_model(self, model, log_freq: int = 100):
        """Watch model in wandb."""
        if not self._initialized or not self.config.get('watch_model', True):
            return
        
        try:
            log_type = 'all' if self.config.get('log_gradients', False) else 'parameters'
            wandb.watch(model, log=log_type, log_freq=log_freq)
        except Exception as e:
            self.logger.warning(f"Failed to watch model in wandb: {e}")
    
    def finish(self):
        """Finish wandb session."""
        if self._initialized:
            try:
                wandb.finish()
                self.logger.info("Wandb session finished")
            except Exception as e:
                self.logger.warning(f"Failed to finish wandb session: {e}")
    
    def is_available(self) -> bool:
        """Check if wandb is available."""
        return WANDB_AVAILABLE
