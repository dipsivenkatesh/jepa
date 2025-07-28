"""
Console and file logger implementation.
"""

import logging
import os
from typing import Dict, Any, Optional, Union
from datetime import datetime

from .base_logger import BaseLogger


class ConsoleLogger(BaseLogger):
    """Console and file logger implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize console logger.
        
        Args:
            config: Console logger configuration dictionary
        """
        self.config = config
        self.logger = self._setup_logger()
        self._initialized = True
    
    def _setup_logger(self) -> logging.Logger:
        """Setup Python logger with console and file handlers."""
        logger = logging.getLogger('jepa')
        logger.setLevel(getattr(logging, self.config.get('level', 'INFO')))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            self.config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # Console handler
        if self.config.get('console', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, self.config.get('console_level', 'INFO')))
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if self.config.get('file', False):
            log_dir = self.config.get('log_dir', './logs')
            os.makedirs(log_dir, exist_ok=True)
            
            experiment_name = self.config.get('experiment_name', 'jepa_experiment')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, self.config.get('file_level', 'DEBUG')))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            logger.info(f"Logging to file: {log_file}")
        
        return logger
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None, prefix: str = ""):
        """Log metrics to console/file."""
        if not self._initialized:
            return
        
        try:
            metrics_str = ", ".join([f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" 
                                   for k, v in metrics.items()])
            
            step_str = f"Step {step} - " if step is not None else ""
            prefix_str = f"[{prefix}] " if prefix else ""
            
            self.logger.info(f"{prefix_str}{step_str}{metrics_str}")
        except Exception as e:
            self.logger.warning(f"Failed to log metrics: {e}")
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to console/file."""
        if not self._initialized:
            return
        
        try:
            self.logger.info("Hyperparameters:")
            for key, value in params.items():
                self.logger.info(f"  {key}: {value}")
        except Exception as e:
            self.logger.warning(f"Failed to log hyperparameters: {e}")
    
    def log_artifact(self, file_path: str, name: str, artifact_type: str = "model"):
        """Log artifact path to console/file."""
        if not self._initialized:
            return
        
        try:
            self.logger.info(f"Artifact saved - {artifact_type}: {name} -> {file_path}")
        except Exception as e:
            self.logger.warning(f"Failed to log artifact: {e}")
    
    def watch_model(self, model, log_freq: int = 100):
        """Log model information to console/file."""
        if not self._initialized:
            return
        
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
            self.logger.info(f"Model architecture: {model.__class__.__name__}")
        except Exception as e:
            self.logger.warning(f"Failed to log model info: {e}")
    
    def finish(self):
        """Finish console logger session."""
        if self._initialized:
            self.logger.info("Training session completed")
    
    def is_available(self) -> bool:
        """Console logger is always available."""
        return True
