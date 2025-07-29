# Simple Sphinx configuration for Read the Docs
import os
import sys

# Add the project to the path
sys.path.insert(0, os.path.abspath('../../'))

# Mock all the heavy dependencies
from unittest.mock import MagicMock

class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

# Mock modules
MOCK_MODULES = [
    'torch', 'torch.nn', 'torch.nn.functional', 'torch.optim', 'torch.utils', 
    'torch.utils.data', 'torchvision', 'transformers', 'datasets', 'wandb',
    'tensorboard', 'matplotlib', 'matplotlib.pyplot', 'numpy', 'pandas',
    'scipy', 'sklearn', 'scikit-learn', 'tqdm', 'yaml', 'pyyaml'
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = Mock()

# Project information
project = 'JEPA Framework'
copyright = '2025, Dilip Venkatesh'
author = 'Dilip Venkatesh'
release = '0.1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
]

# Add MyST support
source_suffix = {
    '.rst': None,
    '.md': None,
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
]

templates_path = ['_templates']
exclude_patterns = []

# HTML output options
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Autodoc options
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Don't check for broken external links during build
linkcheck_ignore = [r'.*']
