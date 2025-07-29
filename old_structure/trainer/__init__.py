"""
Training package for JEPA models.

This package provides a complete training framework for JEPA models including:
- Flexible trainer with customizable optimizers and schedulers
- Comprehensive evaluation metrics and analysis tools
- Utility functions for experiment management and reproducibility
"""

from .trainer import JEPATrainer, create_trainer
from .eval import JEPAEvaluator, quick_evaluate
from .utils import (
    count_parameters,
    plot_training_history,
    save_training_config,
    load_training_config,
    create_data_splits,
    setup_reproducibility,
    get_device_info,
    log_model_summary,
    EarlyStopping,
    create_experiment_dir
)

__all__ = [
    # Trainer classes
    'JEPATrainer',
    'create_trainer',
    
    # Evaluation
    'JEPAEvaluator',
    'quick_evaluate',
    
    # Utilities
    'count_parameters',
    'plot_training_history',
    'save_training_config',
    'load_training_config',
    'create_data_splits',
    'setup_reproducibility',
    'get_device_info',
    'log_model_summary',
    'EarlyStopping',
    'create_experiment_dir'
]
