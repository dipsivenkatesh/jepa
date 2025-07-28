"""
Scalable logging system for JEPA.

This module provides a unified logging interface that supports multiple
backends including Weights & Biases, TensorBoard, and console/file logging.

Usage:
    from logging import create_logger
    
    config = {
        'console': {'enabled': True},
        'wandb': {'enabled': True, 'project': 'jepa-experiments'},
        'tensorboard': {'enabled': True, 'log_dir': './tb_logs'}
    }
    
    logger = create_logger(config)
    logger.log_metrics({'loss': 0.5, 'accuracy': 0.8}, step=100)
"""

from .base_logger import BaseLogger, LoggerRegistry, LogLevel
from .wandb_logger import WandbLogger
from .tensorboard_logger import TensorBoardLogger
from .console_logger import ConsoleLogger
from .multi_logger import MultiLogger, create_logger

__all__ = [
    'BaseLogger',
    'LoggerRegistry', 
    'LogLevel',
    'WandbLogger',
    'TensorBoardLogger',
    'ConsoleLogger',
    'MultiLogger',
    'create_logger'
]
