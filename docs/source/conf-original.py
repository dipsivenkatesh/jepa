# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from unittest.mock import MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('../../'))

# Mock dependencies that may not be available during doc builds
autodoc_mock_imports = [
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.optim',
    'torch.utils',
    'torch.utils.data',
    'torchvision',
    'transformers',
    'datasets',
    'wandb',
    'tensorboard',
    'matplotlib',
    'matplotlib.pyplot',
    'seaborn',
    'numpy',
    'pandas',
    'scipy',
    'sklearn',
    'scikit-learn',
    'tqdm',
    'yaml',
    'pyyaml',
    'omegaconf',
    'huggingface_hub',
    'safetensors'
]

# More robust mocking for Read the Docs
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

for mod_name in autodoc_mock_imports:
    sys.modules[mod_name] = Mock()

# -- Project information -----------------------------------------------------
project = 'JEPA Framework'
copyright = '2025, Dilip Venkatesh'
author = 'Dilip Venkatesh'
release = '0.1.0'
version = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode', 
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'myst_parser',
    'sphinx_copybutton',
]

# Only add optional extensions if available
try:
    import sphinx_design
    extensions.append('sphinx_design')
except ImportError:
    pass

try:
    import sphinxcontrib.mermaid
    extensions.append('sphinxcontrib.mermaid')
except ImportError:
    pass

# MyST-Parser Configuration
source_suffix = {
    '.rst': None,
    '.md': None,
}

myst_enable_extensions = [
    "colon_fence",
    "deflist", 
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Template and exclusion configuration
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Autosummary settings
autosummary_generate = True

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Intersphinx settings
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# HTML theme options
html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#2980B9',
}

# Custom CSS
html_css_files = [
    'custom.css',
]

# Logo (commented out until logo is available)
# html_logo = '_static/logo.png'

# Sidebar
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Source suffix (configured earlier)
# source_suffix = {
#     '.rst': None,
#     '.md': 'myst_parser',
# }
