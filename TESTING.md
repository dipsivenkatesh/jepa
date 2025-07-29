# JEPA Framework Testing Guide

This document provides comprehensive information about the testing system for the JEPA framework.

## Overview

The JEPA framework includes a robust testing suite with:
- **Unit tests** for all core components
- **Integration tests** for end-to-end workflows  
- **Performance benchmarks** for optimization
- **Code quality checks** for maintainability
- **Continuous integration** via GitHub Actions

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── test_config_data.py      # Test configuration constants
├── test_model.py           # Model component tests
├── test_data.py            # Data handling tests  
├── test_trainer.py         # Training system tests
├── test_config.py          # Configuration system tests
├── test_logging.py         # Logging system tests
└── test_cli.py             # CLI interface tests
```

## Running Tests

### Quick Start

```bash
# Run all tests with coverage
python test_jepa.py

# Run specific test category
python test_jepa.py --mode unit --module model

# Run smoke test only
python test_jepa.py --mode smoke

# Check dependencies
python test_jepa.py --mode deps
```

### Using Pytest (Recommended)

```bash
# Install test dependencies
pip install -r test_requirements.txt

# Run all tests with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run specific test file
pytest tests/test_model.py -v

# Run tests with specific markers
pytest tests/ -m "unit" -v
pytest tests/ -m "integration" -v
```

### Using unittest

```bash
# Run all tests
python -m unittest discover tests -v

# Run specific test module
python -m unittest tests.test_model -v

# Run specific test class
python -m unittest tests.test_model.TestJEPA -v
```

### Using the Custom Test Runner

```bash
# Run with coverage reporting
python run_tests.py --mode coverage

# Run tests by category
python run_tests.py --mode category

# Run specific module
python run_tests.py --mode module --module test_model

# Check dependencies first
python run_tests.py --check-deps
```

## Test Categories

### Unit Tests

Test individual components in isolation:

- **Model Tests** (`test_model.py`)
  - BaseModel functionality
  - Encoder architectures
  - Predictor architectures  
  - JEPA core model
  - Model saving/loading

- **Data Tests** (`test_data.py`)
  - Dataset classes
  - Data transforms
  - Data utilities
  - Factory functions

- **Trainer Tests** (`test_trainer.py`)
  - Training loops
  - Evaluation
  - Checkpointing
  - Utilities

- **Config Tests** (`test_config.py`)
  - Configuration classes
  - Loading/saving
  - Validation
  - Merging

- **Logging Tests** (`test_logging.py`)
  - Logger implementations
  - Multi-logger system
  - Backend integrations

- **CLI Tests** (`test_cli.py`)
  - Argument parsing
  - Command execution
  - Utilities

### Integration Tests

Test complete workflows:

- End-to-end training pipeline
- CLI command execution
- Configuration loading and validation
- Multi-logger integration
- Model persistence

### Performance Tests

Benchmark critical paths:

- Model forward/backward passes
- Data loading performance
- Training step timing
- Memory usage

## Test Configuration

### Pytest Configuration (`pytest.ini`)

- Coverage settings (80% minimum)
- Test discovery patterns
- Output formatting
- Marker definitions
- Warning filters

### Coverage Configuration

- Source code inclusion/exclusion
- Report formats (HTML, XML, terminal)
- Coverage thresholds
- Line exclusion patterns

## Dependencies

### Required Dependencies

```
torch>=1.10.0
numpy>=1.20.0
pandas>=1.3.0
Pillow>=8.0.0
PyYAML>=6.0.0
```

### Testing Dependencies

```
pytest>=6.0.0
pytest-cov>=3.0.0
coverage>=6.0.0
pytest-mock>=3.0.0
```

### Optional Dependencies

```
wandb>=0.12.0
tensorboard>=2.8.0
pytest-benchmark>=3.4.0
pytest-html>=3.0.0
```

## Continuous Integration

### GitHub Actions Workflow

The `.github/workflows/tests.yml` file defines:

- **Multi-platform testing** (Ubuntu, Windows, macOS)
- **Multi-Python version testing** (3.8, 3.9, 3.10, 3.11)
- **Code quality checks** (flake8, black, isort)
- **Security scanning** (pip-audit)
- **Documentation testing**
- **Performance benchmarking**

### CI Triggers

- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches
- Manual workflow dispatch

## Test Writing Guidelines

### Unit Test Structure

```python
import unittest
from unittest.mock import patch, MagicMock

class TestComponentName(unittest.TestCase):
    """Test cases for ComponentName class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Initialize test data
        
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up resources
        
    def test_specific_functionality(self):
        """Test specific functionality with descriptive name."""
        # Arrange
        # Act  
        # Assert
```

### Mocking Guidelines

- Mock external dependencies (wandb, tensorboard)
- Mock file I/O operations
- Mock network calls
- Use `patch` for temporary mocking
- Use `MagicMock` for complex objects

### Test Data

- Use small data sizes for fast execution
- Create deterministic test data
- Clean up temporary files
- Use `tempfile` for temporary directories

## Coverage Requirements

- **Minimum coverage**: 80%
- **Target coverage**: 90%+
- **Critical components**: 95%+

### Coverage Exclusions

- Abstract methods
- Debug code
- Platform-specific code
- External library wrappers

## Performance Benchmarks

### Benchmarked Operations

- Model creation and initialization
- Forward pass timing
- Backward pass timing
- Data loading speed
- Training step duration

### Benchmark Execution

```bash
# Run performance benchmarks
pytest tests/ -m "benchmark" --benchmark-only

# Compare with baseline
pytest tests/ -m "benchmark" --benchmark-compare

# Save benchmark results
pytest tests/ -m "benchmark" --benchmark-save=baseline
```

## Debugging Tests

### Verbose Output

```bash
# Detailed test output
pytest tests/ -v -s

# Show local variables on failure
pytest tests/ -l

# Stop on first failure
pytest tests/ -x

# Run specific test with pdb
pytest tests/test_model.py::TestJEPA::test_forward_pass --pdb
```

### Test Debugging Tips

1. **Use descriptive test names**
2. **Add debug prints with `-s` flag**
3. **Check test isolation** - run tests individually
4. **Verify test data** - ensure fixtures are correct
5. **Mock verification** - check mock calls

## Common Issues and Solutions

### Import Errors

```bash
# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use pytest discovery
pytest tests/ --import-mode=importlib
```

### CUDA/GPU Tests

```bash
# Skip GPU tests on CPU-only systems
pytest tests/ -m "not gpu"

# Force CPU testing
CUDA_VISIBLE_DEVICES="" pytest tests/
```

### Memory Issues

```bash
# Run tests with memory profiling
pytest tests/ --memprof

# Limit parallel test execution
pytest tests/ -n 2
```

## Contributing

### Adding New Tests

1. **Create test file** following naming convention
2. **Add to appropriate category** using pytest markers
3. **Include docstrings** for test classes and methods
4. **Add to CI workflow** if needed
5. **Update coverage requirements** if applicable

### Test Review Checklist

- [ ] Tests are deterministic
- [ ] External dependencies are mocked
- [ ] Temporary files are cleaned up
- [ ] Test data is minimal but sufficient
- [ ] Tests run quickly (< 1 second each)
- [ ] Tests have descriptive names
- [ ] Edge cases are covered
- [ ] Error conditions are tested

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [unittest documentation](https://docs.python.org/3/library/unittest.html)
- [coverage.py documentation](https://coverage.readthedocs.io/)
- [Python testing best practices](https://realpython.com/python-testing/)

---

For questions or issues with the testing system, please open an issue on the [GitHub repository](https://github.com/dipsivenkatesh/jepa/issues).
