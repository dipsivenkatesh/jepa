"""
Configuration management for JEPA.

This module provides configuration classes and utilities for loading,
saving, and managing JEPA training configurations.

Usage:
    from jepa.config import load_config, JEPAConfig
    
    # Load from YAML file
    config = load_config("config/default_config.yaml")
    
    # Create programmatically
    config = JEPAConfig(
        model=ModelConfig(encoder_dim=768),
        training=TrainingConfig(batch_size=64)
    )
"""

from .config import (
    JEPAConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    WandbConfig,
    TensorBoardConfig,
    ConsoleConfig,
    LoggingConfig,
    load_config,
    save_config
)

__all__ = [
    'JEPAConfig',
    'ModelConfig', 
    'TrainingConfig',
    'DataConfig',
    'WandbConfig',
    'TensorBoardConfig',
    'ConsoleConfig',
    'LoggingConfig',
    'load_config',
    'save_config'
]