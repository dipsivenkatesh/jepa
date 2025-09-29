# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- GPU optimizations with Triton kernels
- Additional pre-trained models
- More data format support
- Advanced visualization tools
- Performance benchmarks
- Docker containers

## [0.1.1] - 2025-09-26

### Added
- `JEPAAction` variant with explicit action encoder support for action-conditioned prediction tasks.

### Changed
- Documentation refinements across README and Sphinx config for the action-conditioned workflow.
- Workflow configuration updates tied to the latest documentation pipeline.

## [0.1.0] - 2025-07-29

### Added
- Initial release of JEPA (Joint-Embedding Predictive Architecture) framework
- Core JEPA model implementation with modular encoder-predictor architecture
- Multi-modal support for vision, NLP, time series, and audio data
- Comprehensive training framework with distributed training support
- Flexible configuration system with YAML support
- Advanced logging system supporting Weights & Biases, TensorBoard, and console logging
- Production-ready CLI interface with `jepa-train` and `jepa-evaluate` commands
- Extensive data utilities and transformations
- HuggingFace datasets compatibility
- Complete documentation and examples
- Type hints throughout the codebase

### Features
- **Models**: Flexible JEPA architecture with pluggable encoders and predictors
- **Training**: Full-featured trainer with early stopping, checkpointing, and metrics
- **Configuration**: YAML-based configuration with override capabilities
- **Data**: Support for multiple data formats and automatic batching
- **Logging**: Multi-backend logging with experiment tracking
- **CLI**: Easy-to-use command-line interface for training and evaluation
- **Examples**: Comprehensive examples for different use cases

### Technical Details
- Minimum Python version: 3.8
- PyTorch-based implementation
- Follows modern Python packaging standards
- Comprehensive test suite
- Type-safe implementation with mypy support
- Production-ready with proper error handling and logging
