"""
Base logger interface for JEPA training and evaluation.

This module provides a scalable logging architecture that supports
multiple backends (wandb, tensorboard, console, etc.) through a 
unified interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
import logging


class BaseLogger(ABC):
    """Abstract base class for all loggers."""
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None, prefix: str = ""):
        """Log metrics to the backend."""
        pass
    
    @abstractmethod
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters/configuration."""
        pass
    
    @abstractmethod
    def log_artifact(self, file_path: str, name: str, artifact_type: str = "model"):
        """Log artifacts (models, files, etc.)."""
        pass
    
    @abstractmethod
    def watch_model(self, model, log_freq: int = 100):
        """Watch model for gradients/parameters."""
        pass
    
    @abstractmethod
    def finish(self):
        """Finish logging session."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if logger backend is available."""
        pass


class LogLevel:
    """Standard logging levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LoggerRegistry:
    """Registry for managing multiple logger instances."""
    
    def __init__(self):
        self._loggers: Dict[str, BaseLogger] = {}
        self._active_loggers: List[str] = []
    
    def register(self, name: str, logger: BaseLogger):
        """Register a logger."""
        if logger.is_available():
            self._loggers[name] = logger
            self._active_loggers.append(name)
        else:
            logging.warning(f"Logger {name} is not available, skipping registration")
    
    def get_logger(self, name: str) -> Optional[BaseLogger]:
        """Get a specific logger."""
        return self._loggers.get(name)
    
    def get_active_loggers(self) -> List[BaseLogger]:
        """Get all active loggers."""
        return [self._loggers[name] for name in self._active_loggers if name in self._loggers]
    
    def remove_logger(self, name: str):
        """Remove a logger."""
        if name in self._loggers:
            del self._loggers[name]
        if name in self._active_loggers:
            self._active_loggers.remove(name)
