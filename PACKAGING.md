# JEPA Package Build and Distribution Guide

This guide explains how to build and distribute the JEPA package.

## Prerequisites

Make sure you have the latest versions of build tools:

```bash
pip install --upgrade pip build twine
```

## Building the Package

### 1. Development Installation

For development and testing:

```bash
# Install in editable mode
pip install -e .

# Or run the development setup script
./install_dev.sh
```

### 2. Building Distribution Packages

Build both wheel and source distributions:

```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Build the package
python -m build

# This creates:
# - dist/jepa-0.1.0.tar.gz (source distribution)
# - dist/jepa-0.1.0-py3-none-any.whl (wheel distribution)
```

### 3. Verifying the Build

Check the built package:

```bash
# Check package contents
tar -tzf dist/jepa-0.1.0.tar.gz

# Test installation from wheel
pip install dist/jepa-0.1.0-py3-none-any.whl

# Verify installation
python verify_install.py
```

## Testing the Package

### Local Testing

```bash
# Install in a fresh virtual environment
python -m venv test_env
source test_env/bin/activate
pip install dist/jepa-0.1.0-py3-none-any.whl

# Test basic functionality
python -c "import jepa; print(jepa.__version__)"
jepa-train --help
```

### TestPyPI Upload (Optional)

Upload to TestPyPI for testing:

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ jepa
```

## Publishing to PyPI

### 1. Final Checks

Before publishing, ensure:
- [ ] All tests pass: `pytest`
- [ ] Version is correct in `jepa/__init__.py`
- [ ] CHANGELOG.md is updated
- [ ] Documentation is up to date

### 2. Upload to PyPI

```bash
# Upload to PyPI (requires PyPI account and API token)
twine upload dist/*
```

### 3. Installation from PyPI

Once published, users can install with:

```bash
pip install jepa
```

## Package Structure

The package includes:

```
jepa/
├── __init__.py          # Main package interface
├── models/              # Model implementations
├── trainer/             # Training framework
├── config/              # Configuration management
├── data/                # Data utilities
├── loggers/             # Logging system
├── cli/                 # Command line interface
└── py.typed             # Type hint marker
```

## Key Files for Packaging

- `pyproject.toml` - Modern Python packaging configuration
- `setup.py` - Backward compatibility setup script
- `MANIFEST.in` - Additional files to include
- `requirements.txt` - Runtime dependencies
- `py.typed` - Indicates this is a typed package

## Command Line Tools

The package provides these CLI commands:

- `jepa-train` - Train JEPA models
- `jepa-evaluate` - Evaluate trained models

## Development Workflow

1. Make changes to the code
2. Update version in `jepa/__init__.py`
3. Run tests: `pytest`
4. Update documentation if needed
5. Build package: `python -m build`
6. Test locally: `pip install dist/jepa-*.whl`
7. Upload to PyPI: `twine upload dist/*`

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all `__init__.py` files are properly configured
2. **Missing dependencies**: Check `requirements.txt` and `pyproject.toml`
3. **CLI commands not found**: Verify entry points in `pyproject.toml`
4. **Type checking issues**: Ensure `py.typed` file is included

### Useful Commands

```bash
# Check package metadata
python setup.py check

# List package contents
python -c "import jepa; print(jepa.__file__)"

# Check entry points
pip show -f jepa

# Uninstall for clean testing
pip uninstall jepa -y
```
