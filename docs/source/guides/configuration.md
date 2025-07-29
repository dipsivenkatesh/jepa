# Configuration Guide

This guide covers all aspects of configuring JEPA for your specific use case.

## Configuration Overview

JEPA uses YAML configuration files to specify:

- Model architecture parameters
- Training hyperparameters  
- Data loading settings
- Logging configuration
- Experiment tracking

## Configuration Structure

A typical configuration file has these sections:

```yaml
# Model architecture
model:
  encoder_dim: 512
  predictor_dim: 256
  num_layers: 6
  
# Training parameters
training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.001
  
# Data settings
data:
  dataset_path: "data/"
  train_split: 0.8
  
# Logging setup
logging:
  backends: ["wandb", "tensorboard"]
  level: "INFO"
```

## Model Configuration

### Core Architecture

```yaml
model:
  # Encoder settings
  encoder_dim: 512           # Hidden dimension
  encoder_layers: 6          # Number of layers
  encoder_heads: 8           # Attention heads (for Transformer)
  encoder_type: "transformer" # Options: transformer, cnn, mlp
  
  # Predictor settings
  predictor_dim: 256         # Predictor hidden size
  predictor_layers: 3        # Predictor depth
  
  # Output settings
  output_dim: 128            # Final embedding dimension
  dropout: 0.1               # Dropout rate
```

### Advanced Architecture Options

**Transformer Encoder**:

```yaml
model:
  encoder_type: "transformer"
  encoder_dim: 512
  encoder_layers: 12
  encoder_heads: 8
  feedforward_dim: 2048
  positional_encoding: true
  layer_norm_eps: 1e-6
```

**CNN Encoder**:

```yaml
model:
  encoder_type: "cnn"
  channels: [64, 128, 256, 512]
  kernel_sizes: [3, 3, 3, 3]
  strides: [2, 2, 2, 2]
  pooling: "adaptive"
```

**MLP Encoder**:

```yaml
model:
  encoder_type: "mlp"
  hidden_dims: [512, 256, 128]
  activation: "relu"
  batch_norm: true
```

## Training Configuration

### Basic Training Settings

```yaml
training:
  epochs: 100               # Training epochs
  batch_size: 64            # Batch size
  learning_rate: 0.001      # Initial learning rate
  weight_decay: 1e-4        # L2 regularization
  gradient_clip: 1.0        # Gradient clipping
  
  # Validation settings
  val_frequency: 5          # Validate every N epochs
  val_patience: 10          # Early stopping patience
```

### Optimizer Configuration

```yaml
training:
  optimizer:
    type: "adamw"           # Options: adam, adamw, sgd, rmsprop
    learning_rate: 0.001
    betas: [0.9, 0.999]     # Adam betas
    weight_decay: 1e-4
    eps: 1e-8
```

### Learning Rate Scheduling

```yaml
training:
  scheduler:
    type: "cosine"          # Options: cosine, step, exponential, plateau
    warmup_epochs: 10       # Warmup period
    min_lr: 1e-6           # Minimum learning rate
    
    # For step scheduler
    step_size: 30
    gamma: 0.1
    
    # For plateau scheduler
    patience: 5
    factor: 0.5
```

## Data Configuration

### Dataset Settings

```yaml
data:
  # Data paths
  dataset_path: "data/train"
  val_path: "data/val"      # Optional separate validation
  test_path: "data/test"    # Optional test set
  
  # Data splits
  train_split: 0.8          # Train/val split if no separate val
  random_seed: 42           # For reproducible splits
  
  # Loading settings
  batch_size: 64            # Can override training batch_size
  num_workers: 4            # DataLoader workers
  pin_memory: true          # GPU optimization
  shuffle: true             # Shuffle training data
```

### Data Transforms

```yaml
data:
  transforms:
    # Image transforms
    resize: [224, 224]
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
    augmentation:
      horizontal_flip: 0.5
      rotation: 15
      color_jitter: 0.1
      
    # Text transforms
    tokenizer: "bert-base-uncased"
    max_length: 512
    truncation: true
    
    # Time series transforms
    window_size: 100
    stride: 50
    normalization: "z-score"
```

### Context/Target Configuration

```yaml
data:
  masking:
    strategy: "random"      # Options: random, block, structured
    mask_ratio: 0.15        # Fraction to mask
    block_size: 16          # For block masking
    
    # Context/target regions
    context_ratio: 0.85     # Fraction for context
    target_overlap: false   # Allow context/target overlap
```

## Logging Configuration

### Basic Logging

```yaml
logging:
  level: "INFO"             # DEBUG, INFO, WARNING, ERROR
  backends: ["console"]     # Active logging backends
  log_dir: "logs"          # Output directory
  
  # Console settings
  console:
    format: "%(asctime)s - %(levelname)s - %(message)s"
    colored: true
```

### Weights & Biases

```yaml
logging:
  backends: ["wandb", "console"]
  
  wandb:
    project: "jepa-experiments"
    entity: "my-team"
    name: "experiment-1"     # Run name
    tags: ["baseline", "transformer"]
    notes: "Initial experiment"
    
    # What to log
    log_frequency: 10        # Log every N steps
    log_gradients: false     # Log gradient histograms
    log_model: false         # Save model artifacts
```

### TensorBoard

```yaml
logging:
  backends: ["tensorboard", "console"]
  
  tensorboard:
    log_dir: "logs/tensorboard"
    log_frequency: 10
    log_images: true         # Log sample images
    log_histograms: true     # Log parameter histograms
```

### Multi-Backend Logging

```yaml
logging:
  backends: ["wandb", "tensorboard", "console"]
  
  # Shared settings
  log_frequency: 10
  save_frequency: 50        # Save checkpoints every N epochs
  
  # Backend-specific settings
  wandb:
    project: "jepa"
  tensorboard:
    log_dir: "logs/tb"
  console:
    level: "INFO"
```

## Environment-Specific Configs

### Development Configuration

```yaml
# dev_config.yaml
model:
  encoder_dim: 128          # Smaller for faster training
  encoder_layers: 2
  
training:
  epochs: 5                 # Quick testing
  batch_size: 16            # Fit on smaller GPUs
  
logging:
  backends: ["console"]     # Simple logging
  level: "DEBUG"            # Verbose output
```

### Production Configuration

```yaml
# prod_config.yaml
model:
  encoder_dim: 1024         # Large model
  encoder_layers: 24
  
training:
  epochs: 1000              # Long training
  batch_size: 128           # Large batches
  gradient_clip: 1.0        # Stability
  
logging:
  backends: ["wandb", "tensorboard"]
  wandb:
    project: "production-runs"
```

## Loading and Merging Configs

### Loading from File

```python
from config.config import load_config

config = load_config("my_config.yaml")
```

### Merging Configurations

```python
from config.config import load_config, merge_configs

base_config = load_config("base_config.yaml")
override_config = load_config("overrides.yaml")

final_config = merge_configs(base_config, override_config)
```

### Command Line Overrides

Override config values from command line:

```bash
python -m cli train \
  --config config/base.yaml \
  --learning-rate 0.01 \
  --batch-size 128 \
  --epochs 50
```

## Validation and Debugging

### Config Validation

Validate your configuration:

```bash
python -m cli config --validate my_config.yaml
```

### View Current Config

```bash
python -m cli config --show my_config.yaml
```

### Create Template

Generate a template configuration:

```bash
python -m cli config --create-template new_config.yaml
```

## Best Practices

**Use Version Control**
   Track configuration files in git

**Environment Variables**
   Use environment variables for sensitive data:
   
   ```yaml
   logging:
     wandb:
       api_key: "${WANDB_API_KEY}"
   ```

**Configuration Inheritance**
   Create base configurations and extend them:
   
   ```yaml
   # child_config.yaml
   base_config: "base_config.yaml"
   
   # Override specific values
   training:
     learning_rate: 0.01
   ```

**Comments and Documentation**
   Document your configurations:
   
   ```yaml
   model:
     encoder_dim: 512  # Chosen based on ablation study
   ```
        
**Naming Conventions**
   Use descriptive configuration names:
   
   - `vision_large.yaml`
   - `nlp_debug.yaml`
   - `production_v2.yaml`

## Troubleshooting

**Config Not Found**
   Check file paths are relative to working directory

**Validation Errors**
   Use `--validate` flag to check configuration

**Memory Issues**
   Reduce `batch_size` or `encoder_dim`

**Slow Training**
   Increase `batch_size` or `num_workers`

**NaN Loss**
   Reduce `learning_rate` or add `gradient_clip`

## Examples

See the [examples directory](../examples/index.md) for complete configuration examples for different use cases.
