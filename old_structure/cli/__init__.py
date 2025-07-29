"""
Command Line Interface for JEPA.

This module provides CLI commands for training and evaluating JEPA models.

Usage:
    # Training
    python -m jepa.cli train --config config/default_config.yaml
    
    # Evaluation  
    python -m jepa.cli evaluate --config config/default_config.yaml --checkpoint path/to/model.pth
    
    # Help
    python -m jepa.cli --help
"""

from .train import main as train_main
from .evaluate import main as evaluate_main

__all__ = [
    'train_main',
    'evaluate_main'
]