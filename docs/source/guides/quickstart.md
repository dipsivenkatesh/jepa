# Quick Start Guide

Get started with JEPA in minutes! This guide will walk you through your first training run.

## Overview

JEPA (Joint Embedding Predictive Architecture) is a self-supervised learning framework that learns representations by predicting parts of the input from other parts. This guide shows you how to:

1. Prepare your data
2. Configure the model
3. Train your first JEPA model
4. Evaluate the results

## 30-Second Start

For the impatient, here's how to train a JEPA model right now:

```bash
# Clone and install
git clone https://github.com/your-org/jepa.git
cd jepa
pip install -e .

# Train with defaults
python -m cli train
```

That's it! The model will train with default settings on synthetic data.

## Step-by-Step Tutorial

### Step 1: Prepare Your Data

JEPA works with various data types. Create a simple dataset:

```python
# data_prep.py
import torch
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, size=1000, dim=128):
        self.data = torch.randn(size, dim)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
```

### Step 2: Basic Configuration

Create a configuration file:

```yaml
# my_config.yaml
model:
  encoder_dim: 128
  predictor_dim: 64
  
training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
  
data:
  dataset_path: "path/to/your/data"
  
logging:
  level: "INFO"
  backends: ["console"]
```

### Step 3: Train the Model

Train using the CLI:

```bash
python -m cli train --config my_config.yaml
```

Or programmatically:

```python
from trainer.trainer import JEPATrainer
from config.config import load_config

config = load_config("my_config.yaml")
trainer = JEPATrainer(config)
trainer.train()
```

### Step 4: Monitor Training

View progress in real-time:

```bash
# With TensorBoard
tensorboard --logdir logs/

# Or check console output
tail -f logs/training.log
```

### Step 5: Evaluate Results

Evaluate your trained model:

```bash
python -m cli evaluate --model-path checkpoints/best_model.pth
```

## Common Use Cases

### Computer Vision

Train on image data:

```bash
python -m cli train --config config/vision_config.yaml
```

### Natural Language Processing

Train on text data:

```bash
python -m cli train --config config/nlp_config.yaml
```

### Time Series Forecasting

Train on sequential data:

```bash
python -m cli train --config config/timeseries_config.yaml
```

## Key Concepts

**Encoder**: Transforms input data into representations
   - Configurable architecture (CNN, Transformer, MLP)
   - Shared across context and target

**Predictor**: Predicts target representations from context
   - Typically smaller than encoder
   - Learns meaningful relationships

**Context/Target**: Input is split into context and target regions
   - Context: What the model sees
   - Target: What the model predicts

**Joint Embedding**: Shared representation space
   - Context and target embeddings
   - Enables transfer learning

## Training Tips

**Start Small**
   Begin with small datasets and simple configurations

**Monitor Loss**
   Watch for decreasing prediction loss

**Experiment**
   Try different encoder architectures

**Use Logging**
   Enable comprehensive logging for debugging

**GPU Acceleration**
   Use CUDA for faster training

## Next Steps

Now that you've run your first model:

1. Read the [Configuration Guide](configuration.md) for advanced settings
2. Explore [Training Guide](training.md) for optimization tips
3. Check out [Examples](../examples/index.md) for real-world use cases
4. Learn about [Data Loading](data.md) for custom datasets

## Need Help?

- Review the [FAQ](faq.md)
- Check [API Documentation](../api/index.md)
- Browse [Examples](../examples/index.md)
- Ask questions on [GitHub Discussions](https://github.com/your-org/jepa/discussions)
