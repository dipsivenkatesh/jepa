"""
Multi-logger implementation that manages multiple logging backends.
"""

import logging
from typing import Dict, Any, Optional, Union, List

from .base_logger import BaseLogger


class MultiLogger(BaseLogger):
    """Composite logger that manages multiple logging backends."""
    
    def __init__(self, loggers: List[BaseLogger]):
        """
        Initialize multi-logger with a list of logger instances.
        
        Args:
            loggers: List of logger instances to manage
        """
        self.loggers = loggers
        self.logger = logging.getLogger(__name__)
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None, prefix: str = ""):
        """Log metrics to all backends."""
        for logger in self.loggers:
            try:
                logger.log_metrics(metrics, step, prefix)
            except Exception as e:
                self.logger.warning(f"Failed to log metrics to {type(logger).__name__}: {e}")
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to all backends."""
        for logger in self.loggers:
            try:
                logger.log_hyperparameters(params)
            except Exception as e:
                self.logger.warning(f"Failed to log hyperparameters to {type(logger).__name__}: {e}")
    
    def log_artifact(self, file_path: str, name: str, artifact_type: str = "model"):
        """Log artifacts to all backends."""
        for logger in self.loggers:
            try:
                logger.log_artifact(file_path, name, artifact_type)
            except Exception as e:
                self.logger.warning(f"Failed to log artifact to {type(logger).__name__}: {e}")
    
    def watch_model(self, model, log_freq: int = 100):
        """Watch model on all backends."""
        for logger in self.loggers:
            try:
                logger.watch_model(model, log_freq)
            except Exception as e:
                self.logger.warning(f"Failed to watch model on {type(logger).__name__}: {e}")
    
    def finish(self):
        """Finish all logging sessions."""
        for logger in self.loggers:
            try:
                logger.finish()
            except Exception as e:
                self.logger.warning(f"Failed to finish {type(logger).__name__}: {e}")
    
    def close(self):
        """Alias for finish."""
        self.finish()
    
    def is_available(self) -> bool:
        """Check if at least one logger is available."""
        return any(logger.is_available() for logger in self.loggers)
    
    def get_loggers(self) -> List[BaseLogger]:
        """Get all managed loggers."""
        return self.loggers.copy()
    
    def get_logger_by_type(self, logger_type: type) -> Optional[BaseLogger]:
        """Get logger by type."""
        for logger in self.loggers:
            if isinstance(logger, logger_type):
                return logger
        return None
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MultiLogger':
        """
        Create MultiLogger from configuration dictionary.
        
        Args:
            config: Configuration dictionary with backend-specific settings
            
        Returns:
            MultiLogger instance
        """
        from .wandb_logger import WandbLogger
        from .tensorboard_logger import TensorBoardLogger
        from .console_logger import ConsoleLogger
        
        loggers = []
        
        # Console logger
        console_config = config.get('console', {})
        if console_config.get('enabled', True):
            loggers.append(ConsoleLogger(console_config))
        
        # Wandb logger
        wandb_config = config.get('wandb', {})
        if wandb_config.get('enabled', False):
            loggers.append(WandbLogger(wandb_config))
        
        # TensorBoard logger
        tensorboard_config = config.get('tensorboard', {})
        if tensorboard_config.get('enabled', False):
            loggers.append(TensorBoardLogger(tensorboard_config))
        
        return cls(loggers)


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
    return MultiLogger.from_config(config)
