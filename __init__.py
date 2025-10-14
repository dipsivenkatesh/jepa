"""
JEPA (Joint-Embedding Predictive Architecture) Framework

A powerful self-supervised learning framework for learning representations
by predicting parts of the input from other parts.

Key Features:
- Modular encoder-predictor architecture
- Multi-modal support (vision, NLP, time series, audio)
- High performance with mixed precision and distributed training
- Comprehensive logging and monitoring
- Production-ready CLI interface

Quick Start:
    >>> from jepa import JEPA, JEPATrainer
    >>> from jepa.config import load_config
    >>> 
    >>> # Load configuration and create model
    >>> config = load_config("config/default_config.yaml")
    >>> model = JEPA(config.model)
    >>> trainer = JEPATrainer(model, config)
    >>> trainer.train()

CLI Usage:
    $ python -m jepa.cli train --config config/default_config.yaml
    $ python -m jepa.cli evaluate --config config/default_config.yaml
"""

__version__ = "0.1.0"
__author__ = "Dilip Venkatesh"
__email__ = "your.email@example.com"
__description__ = "Joint-Embedding Predictive Architecture for Self-Supervised Learning"

# Core model components
from .models import JEPA, JEPAAction, BaseModel, Encoder, Predictor

# Training framework
from .trainer import JEPATrainer, JEPAEvaluator, create_trainer

# Loss functions
from .loss_functions import (
    get_loss,
    vicreg_loss,
    vcreg_loss,
    mse_loss,
    LOSS_FUNCTIONS,
)

# Configuration management
from .config import load_config, save_config, JEPAConfig

# Data utilities
from .data import (
    JEPADataset,
    create_dataset,
    JEPATransforms,
    collate_jepa_batch
)

# Logging system
from .loggers import create_logger, MultiLogger

# Utility functions
from .trainer.utils import (
    count_parameters,
    setup_reproducibility,
    get_device_info,
    EarlyStopping
)

# Package metadata
__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',
    '__description__',
    
    # Core models
    'JEPA',
    'JEPAAction',
    'BaseModel', 
    'Encoder',
    'Predictor',
    
    # Training
    'JEPATrainer',
    'JEPAEvaluator',
    'create_trainer',

    # Loss functions
    'get_loss',
    'vicreg_loss',
    'vcreg_loss',
    'mse_loss',
    'LOSS_FUNCTIONS',
    
    # Configuration
    'load_config',
    'save_config',
    'JEPAConfig',
    
    # Data
    'JEPADataset',
    'create_dataset',
    'JEPATransforms',
    'collate_jepa_batch',
    
    # Logging
    'create_logger',
    'MultiLogger',
    
    # Utilities
    'count_parameters',
    'setup_reproducibility', 
    'get_device_info',
    'EarlyStopping'
]

# Convenience imports for common use cases
def quick_start(config_path: str):
    """
    Quick start function to begin training with minimal setup.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        JEPATrainer: Configured trainer ready for training
    """
    config = load_config(config_path)

    encoder = Encoder(config.model.encoder_dim)
    predictor = Predictor(config.model.encoder_dim)
    model = JEPA(encoder=encoder, predictor=predictor)

    loss_name = getattr(config.training, 'loss', 'mse')
    try:
        loss_fn = get_loss(loss_name)
    except KeyError:
        loss_fn = mse_loss

    trainer = create_trainer(
        model=model,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        device=config.device,
        log_interval=config.training.log_interval,
        gradient_clip_norm=config.training.gradient_clip_norm,
        save_dir=config.checkpoint_dir,
        loss_fn=loss_fn,
    )

    return trainer

# Module-level configuration
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
