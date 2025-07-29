# Installation Guide

This guide will help you install and set up the JEPA framework.

## Requirements

### System Requirements

- Python 3.8 or higher
- CUDA 11.0+ (for GPU training)
- 8GB+ RAM (16GB+ recommended)
- 10GB+ disk space

### Core Dependencies

- PyTorch 2.0+
- NumPy
- PyYAML
- tqdm

### Optional Dependencies

- Weights & Biases (for experiment tracking)
- TensorBoard (for visualization)
- Triton (for kernel optimization)
- Hugging Face Transformers (for compatibility)

## Installation Methods

### From Source (Recommended)

1. Clone the repository:

```bash
git clone https://github.com/your-org/jepa.git
cd jepa
```

2. Create a virtual environment:

```bash
python -m venv jepa-env
source jepa-env/bin/activate  # On Windows: jepa-env\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install in development mode:

```bash
pip install -e .
```

### Using pip

```bash
pip install jepa-framework
```

### Using conda

```bash
conda install -c conda-forge jepa-framework
```

## Verification

Verify your installation by running:

```bash
python -c "import jepa; print('JEPA installed successfully!')"
```

Or run the test suite:

```bash
python -m pytest tests/
```

## GPU Setup

### CUDA Installation

1. Install CUDA Toolkit from NVIDIA website
2. Install cuDNN compatible with your CUDA version
3. Install PyTorch with CUDA support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU availability:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Optional Components

### Weights & Biases

1. Create an account at https://wandb.ai
2. Install wandb:

```bash
pip install wandb
```

3. Login:

```bash
wandb login
```

### TensorBoard

Install TensorBoard for visualization:

```bash
pip install tensorboard
```

### Triton (Advanced)

For kernel optimization:

```bash
pip install triton
```

## Development Setup

For contributors and advanced users:

1. Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

2. Install pre-commit hooks:

```bash
pre-commit install
```

3. Run tests:

```bash
python -m pytest tests/ -v
```

## Troubleshooting

### Common Issues

**ImportError: No module named 'torch'**
   Install PyTorch: `pip install torch`

**CUDA out of memory**
   Reduce batch size in your configuration

**Permission denied errors**
   Use `--user` flag: `pip install --user package_name`

**Package conflicts**
   Create a fresh virtual environment

### Getting Help

- Check the [FAQ](faq.md)
- Search [GitHub Issues](https://github.com/your-org/jepa/issues)
- Join our [Discord](https://discord.gg/jepa)

## Next Steps

After installation, proceed to the [Quick Start Guide](quickstart.md) to begin using JEPA.
