"""
Setup script for JEPA package.

This file is maintained for backward compatibility. The main package configuration
is now in pyproject.toml following modern Python packaging standards.
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read version from package
def get_version():
    """Get version from the package."""
    version_file = this_directory / "jepa" / "__init__.py"
    with open(version_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return "0.1.0"

# Read requirements
def get_requirements():
    """Get requirements from requirements.txt."""
    requirements_file = this_directory / "requirements.txt"
    if requirements_file.exists():
        with open(requirements_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="jepa",
    version=get_version(),
    author="Dilip Venkatesh",
    author_email="your.email@example.com",
    description="Joint-Embedding Predictive Architecture for Self-Supervised Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dipsivenkatesh/jepa",
    project_urls={
        "Bug Tracker": "https://github.com/dipsivenkatesh/jepa/issues",
        "Documentation": "https://jepa.readthedocs.io/",
        "Source Code": "https://github.com/dipsivenkatesh/jepa",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10", 
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0", 
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.950",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
            "sphinx-autodoc-typehints>=1.19.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "jepa-train=jepa.cli.train:main",
            "jepa-evaluate=jepa.cli.evaluate:main",
        ],
    },
    package_data={
        "jepa": [
            "config/*.yaml",
            "config/*.yml",
            "py.typed",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "self-supervised-learning",
        "representation-learning",
        "deep-learning", 
        "pytorch",
        "jepa",
        "joint-embedding",
        "predictive-architecture",
    ],
)