"""
Models package for OpenJEPA.

This package provides a flexible JEPA implementation that can work with any encoder and predictor.
"""

from .base import BaseModel
from .jepa import JEPA
from .encoder import Encoder
from .predictor import Predictor

__all__ = ['BaseModel', 'JEPA', 'Encoder', 'Predictor']