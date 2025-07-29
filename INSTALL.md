# JEPA Package Installation and Usage Guide

## 🚀 Quick Installation

```bash
# Install from PyPI (once published)
pip install jepa

# Install from source
pip install git+https://github.com/dipsivenkatesh/jepa.git

# Install from local wheel
pip install jepa-0.1.0-py3-none-any.whl
```

## 📚 Quick Start Guide

### 1. Basic Usage

```python
import jepa
from jepa import JEPA, JEPATrainer, load_config

# Load configuration
config = load_config("config/default_config.yaml")

# Create model and trainer
model = JEPA(config.model)
trainer = JEPATrainer(model, config)

# Train the model
trainer.train()
```

### 2. Using the Quick Start Function

```python
from jepa import quick_start

# One-line setup
trainer = quick_start("config/default_config.yaml")
trainer.train()
```

### 3. CLI Usage

```bash
# Train a model
jepa-train --config config/default_config.yaml

# Evaluate a model
jepa-evaluate --config config/default_config.yaml --checkpoint model.pth

# Generate a default configuration
jepa-train --generate-config my_config.yaml

# Get help
jepa-train --help
```

### 4. Custom Configuration

```python
from jepa.config import JEPAConfig, ModelConfig, TrainingConfig

# Create custom configuration
config = JEPAConfig(
    model=ModelConfig(
        encoder_dim=768,
        predictor_hidden_dim=1024
    ),
    training=TrainingConfig(
        batch_size=64,
        learning_rate=1e-3,
        num_epochs=100
    )
)
```

### 5. Working with Data

```python
from jepa.data import create_dataset, JEPATransforms

# Create dataset
dataset = create_dataset(
    data_path="path/to/data",
    sequence_length=10,
    transforms=JEPATransforms()
)
```

## 🎯 Use Cases

### Computer Vision
```python
# Image classification pretraining
config = load_config("config/vision_config.yaml")
trainer = quick_start("config/vision_config.yaml")
```

### Natural Language Processing
```python
# Language model pretraining
config = load_config("config/nlp_config.yaml")
trainer = quick_start("config/nlp_config.yaml")
```

### Time Series
```python
# Time series forecasting
config = load_config("config/timeseries_config.yaml")
trainer = quick_start("config/timeseries_config.yaml")
```

## 🔧 Advanced Features

### Distributed Training
```python
# Configure for multi-GPU training
config.training.distributed = True
config.training.world_size = 4
```

### Weights & Biases Integration
```python
# Enable W&B logging
config.logging.wandb.enabled = True
config.logging.wandb.project = "jepa-experiments"
```

### Custom Models
```python
from jepa.models import BaseModel

class CustomEncoder(BaseModel):
    def __init__(self, config):
        super().__init__()
        # Your custom encoder implementation
        
# Use with JEPA
model = JEPA(config.model, encoder=CustomEncoder)
```

## 📖 Configuration Reference

### Model Configuration
- `encoder_type`: Type of encoder ('transformer', 'cnn', etc.)
- `encoder_dim`: Encoder output dimension
- `predictor_type`: Type of predictor ('mlp', 'transformer')
- `predictor_hidden_dim`: Hidden dimension for predictor
- `dropout`: Dropout rate

### Training Configuration
- `batch_size`: Training batch size
- `learning_rate`: Learning rate
- `num_epochs`: Number of training epochs
- `warmup_epochs`: Number of warmup epochs
- `weight_decay`: Weight decay for regularization
- `gradient_clip_norm`: Gradient clipping norm

### Data Configuration
- `train_data_path`: Path to training data
- `val_data_path`: Path to validation data
- `sequence_length`: Input sequence length
- `num_workers`: Number of data loading workers

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   ```

2. **CUDA Issues**
   ```python
   # Check CUDA availability
   import torch
   print(torch.cuda.is_available())
   ```

3. **Memory Issues**
   ```python
   # Reduce batch size
   config.training.batch_size = 16
   ```

### Getting Help

- 📚 **Documentation**: Check the `docs/` directory
- 🐛 **Issues**: Report bugs on GitHub
- 💬 **Discussions**: Use GitHub Discussions for questions
- 📧 **Contact**: Reach out to the maintainers

## 🧪 Development

### Installing for Development
```bash
# Clone the repository
git clone https://github.com/dipsivenkatesh/jepa.git
cd jepa

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run type checking
mypy jepa/
```

### Building Documentation
```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
python build_docs.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with PyTorch and HuggingFace Transformers
- Inspired by the Joint-Embedding Predictive Architecture principles
- Thanks to the open-source community for the amazing tools
