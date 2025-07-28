"""
Multi-logger implementation that manages multiple logging backends.
"""

import logging
from typing import Dict, Any, Optional, Union, List

from .base_logger import BaseLogger, LoggerRegistry
from .wandb_logger import WandbLogger
from .tensorboard_logger import TensorBoardLogger
from .console_logger import ConsoleLogger


class MultiLogger(BaseLogger):
    """Composite logger that manages multiple logging backends."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize multi-logger with configuration for different backends.
        
        Args:
            config: Configuration dictionary with backend-specific settings
        """
        self.config = config
        self.registry = LoggerRegistry()
        self.logger = logging.getLogger(__name__)
        
        self._setup_loggers()
    
    def _setup_loggers(self):
        """Setup all configured loggers."""
        # Console logger (always enabled by default)
        console_config = self.config.get('console', {'enabled': True})
        if console_config.get('enabled', True):
            console_logger = ConsoleLogger(console_config)
            self.registry.register('console', console_logger)
        
        # Wandb logger
        wandb_config = self.config.get('wandb', {})
        if wandb_config.get('enabled', False):
            wandb_logger = WandbLogger(wandb_config)
            self.registry.register('wandb', wandb_logger)
        
        # TensorBoard logger
        tensorboard_config = self.config.get('tensorboard', {})
        if tensorboard_config.get('enabled', False):
            tensorboard_logger = TensorBoardLogger(tensorboard_config)
            self.registry.register('tensorboard', tensorboard_logger)
        
        active_loggers = [name for name in self.registry._active_loggers]
        self.logger.info(f"Initialized loggers: {active_loggers}")
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None, prefix: str = ""):
        """Log metrics to all active loggers."""
        for logger in self.registry.get_active_loggers():
            logger.log_metrics(metrics, step, prefix)
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to all active loggers."""
        for logger in self.registry.get_active_loggers():
            logger.log_hyperparameters(params)
    
    def log_artifact(self, file_path: str, name: str, artifact_type: str = "model"):
        """Log artifact to all active loggers."""
        for logger in self.registry.get_active_loggers():
            logger.log_artifact(file_path, name, artifact_type)
    
    def watch_model(self, model, log_freq: int = 100):
        """Watch model in all active loggers."""
        for logger in self.registry.get_active_loggers():
            logger.watch_model(model, log_freq)
    
    def finish(self):
        """Finish all logger sessions."""
        for logger in self.registry.get_active_loggers():
            logger.finish()
    
    def is_available(self) -> bool:
        """Multi-logger is available if at least one logger is active."""
        return len(self.registry.get_active_loggers()) > 0
    
    def get_logger(self, name: str) -> Optional[BaseLogger]:
        """Get a specific logger by name."""
        return self.registry.get_logger(name)
    
    def add_logger(self, name: str, logger: BaseLogger):
        """Add a new logger at runtime."""
        self.registry.register(name, logger)
    
    def remove_logger(self, name: str):
        """Remove a logger at runtime."""
        logger = self.registry.get_logger(name)
        if logger:
            logger.finish()
        self.registry.remove_logger(name)


def create_logger(config: Dict[str, Any]) -> MultiLogger:
    """
    Factory function to create a multi-logger with the given configuration.
    
    Args:
        config: Configuration dictionary for all logging backends
        
    Returns:
        Configured MultiLogger instance
        
    Example:
        config = {
            'console': {'enabled': True, 'level': 'INFO'},
            'wandb': {'enabled': True, 'project': 'jepa-experiments'},
            'tensorboard': {'enabled': True, 'log_dir': './tb_logs'}
        }
        logger = create_logger(config)
    """
    return MultiLogger(config)
