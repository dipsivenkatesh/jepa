"""
Models package for JEPA.

This package provides a flexible JEPA implementation that can work with any encoder and predictor.
"""

from .base import BaseModel
from .jepa import JEPA, JEPAAction, JEPAInterleaved
from .encoder import Encoder
from .predictor import Predictor

__all__ = ['BaseModel', 'JEPA', 'JEPAAction', 'JEPAInterleaved', 'Encoder', 'Predictor']
