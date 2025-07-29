# Configuration API

This section documents the configuration management system.

## Main Configuration Classes

```{eval-rst}
.. automodule:: config.config
   :members:
   :undoc-members:
   :show-inheritance:
```

## Model Configuration

```{eval-rst}
.. autoclass:: config.config.ModelConfig
   :members:
   :special-members: __init__
```

Configure model architecture:

```python
from jepa.config import ModelConfig

config = ModelConfig(
    encoder_type="transformer",
    encoder_dim=768,
    encoder_layers=12,
    predictor_type="mlp",
    predictor_dim=256
)
```

Or via YAML:

```yaml
model:
  encoder_type: "transformer"
  encoder_dim: 768
  encoder_layers: 12
  encoder_heads: 12
  dropout: 0.1
  
  predictor_type: "mlp"
  predictor_dim: 256
  predictor_layers: 3
  
  loss_function: "contrastive"
  temperature: 0.1
```

## Training Configuration

```{eval-rst}
.. autoclass:: config.config.TrainingConfig
   :members:
   :special-members: __init__
```

Configure training parameters:

```python
from jepa.config import TrainingConfig

config = TrainingConfig(
    epochs=100,
    batch_size=64,
    learning_rate=1e-4,
    optimizer="adamw",
    scheduler="cosine"
)
```

Or via YAML:

```yaml
training:
  epochs: 100
  batch_size: 64
  learning_rate: 1e-4
  weight_decay: 1e-2
  
  optimizer:
    type: "adamw"
    betas: [0.9, 0.999]
    eps: 1e-8
  
  scheduler:
    type: "cosine"
    warmup_epochs: 10
    min_lr: 1e-6
  
  mixed_precision: true
  gradient_clip: 1.0
```

## Data Configuration

```{eval-rst}
.. autoclass:: config.config.DataConfig
   :members:
   :special-members: __init__
```

Configure data loading and processing:

```python
from jepa.config import DataConfig

config = DataConfig(
    dataset_type="csv",
    dataset_path="data/train.csv",
    batch_size=64,
    num_workers=4
)
```

Or via YAML:

```yaml
data:
  dataset_type: "csv"
  dataset_path: "data/train.csv"
  data_columns: ["feature1", "feature2", "feature3"]
  temporal_offset: 1
  
  batch_size: 64
  num_workers: 4
  pin_memory: true
  shuffle: true
  
  transforms:
    normalize: true
    mask_ratio: 0.15
    augmentation: true
```

## Logging Configuration

```{eval-rst}
.. autoclass:: config.config.LoggingConfig
   :members:
   :special-members: __init__
```

Configure logging backends:

```python
from jepa.config import LoggingConfig

config = LoggingConfig(
    backends=["wandb", "tensorboard"],
    level="INFO",
    log_frequency=10
)
```

Or via YAML:

```yaml
logging:
  level: "INFO"
  backends: ["wandb", "tensorboard", "console"]
  
  wandb:
    project: "jepa-experiments"
    entity: "my-team"
    tags: ["baseline"]
  
  tensorboard:
    log_dir: "logs/tensorboard"
    log_images: true
  
  console:
    level: "INFO"
    colored: true
```

## Experiment Configuration

```{eval-rst}
.. autoclass:: config.config.ExperimentConfig
   :members:
   :special-members: __init__
```

Complete experiment configuration:

```python
from jepa.config import ExperimentConfig

config = ExperimentConfig(
    name="jepa_baseline",
    model=model_config,
    training=training_config,
    data=data_config,
    logging=logging_config
)
```

## Configuration Loading

### Load from File

```{eval-rst}
.. autofunction:: config.config.load_config
```

Load configuration from YAML files:

```python
from jepa.config import load_config

# Load complete configuration
config = load_config("config/experiment.yaml")

# Load specific section
model_config = load_config("config/model.yaml", section="model")
```

### Default Configurations

```{eval-rst}
.. autofunction:: config.config.get_default_config
```

Get default configurations for different domains:

```python
from jepa.config import get_default_config

# Get default configuration for vision tasks
vision_config = get_default_config("vision")

# Get default configuration for NLP tasks
nlp_config = get_default_config("nlp")

# Get default configuration for time series
timeseries_config = get_default_config("timeseries")
```

### Configuration Merging

```{eval-rst}
.. autofunction:: config.config.merge_configs
```

Merge multiple configurations:

```python
from jepa.config import merge_configs

# Load base and override configurations
base_config = load_config("config/base.yaml")
override_config = load_config("config/overrides.yaml")

# Merge configurations (override takes precedence)
final_config = merge_configs(base_config, override_config)
```

## Available Configuration Templates

### Default Configuration

Complete default configuration for general use:

```{eval-rst}
.. literalinclude:: ../../config/default_config.yaml
   :language: yaml
   :caption: Default Configuration
```

### Vision Configuration

Optimized for computer vision tasks:

```{eval-rst}
.. literalinclude:: ../../config/vision_config.yaml
   :language: yaml
   :caption: Vision Configuration
```

### NLP Configuration

Optimized for natural language processing:

```{eval-rst}
.. literalinclude:: ../../config/nlp_config.yaml
   :language: yaml
   :caption: NLP Configuration
```

### Time Series Configuration

Optimized for time series forecasting:

```{eval-rst}
.. literalinclude:: ../../config/timeseries_config.yaml
   :language: yaml
   :caption: Time Series Configuration
```

## Configuration Validation

### Validation Functions

```{eval-rst}
.. autofunction:: config.validation.validate_config

.. autofunction:: config.validation.validate_model_config

.. autofunction:: config.validation.validate_training_config
```

Validate configuration files:

```python
from jepa.config.validation import validate_config

# Validate complete configuration
is_valid, errors = validate_config(config)

if not is_valid:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
```

### Schema Validation

```{eval-rst}
.. autoclass:: config.validation.ConfigSchema
   :members:
   :special-members: __init__
```

Use schemas for strict validation:

```python
from jepa.config.validation import ConfigSchema

schema = ConfigSchema({
    'model': {
        'encoder_dim': {'type': int, 'min': 1},
        'encoder_layers': {'type': int, 'min': 1, 'max': 50}
    },
    'training': {
        'batch_size': {'type': int, 'min': 1},
        'learning_rate': {'type': float, 'min': 0.0, 'max': 1.0}
    }
})

# Validate against schema
schema.validate(config)
```

## Environment-Specific Configurations

### Development Configuration

```yaml
# dev_config.yaml
model:
  encoder_dim: 128      # Smaller for faster iteration
  encoder_layers: 2
  
training:
  epochs: 5             # Quick testing
  batch_size: 16
  
logging:
  backends: ["console"] # Simple logging
  level: "DEBUG"
```

### Production Configuration

```yaml
# prod_config.yaml
model:
  encoder_dim: 1024     # Large model
  encoder_layers: 24
  
training:
  epochs: 1000          # Long training
  batch_size: 128
  mixed_precision: true
  
logging:
  backends: ["wandb", "tensorboard"]
  wandb:
    project: "production-runs"
```

### Testing Configuration

```yaml
# test_config.yaml
model:
  encoder_dim: 64       # Minimal for testing
  encoder_layers: 1
  
training:
  epochs: 1
  batch_size: 4
  
data:
  dataset_size: 100     # Small test dataset
```

## Configuration Utilities

### Environment Variable Substitution

```{eval-rst}
.. autofunction:: config.utils.substitute_env_vars
```

Use environment variables in configurations:

```yaml
logging:
  wandb:
    project: "${WANDB_PROJECT}"
    api_key: "${WANDB_API_KEY}"
    
data:
  dataset_path: "${DATA_PATH}/train"
```

```python
from jepa.config.utils import substitute_env_vars
import os

os.environ['WANDB_PROJECT'] = 'my-project'
os.environ['DATA_PATH'] = '/path/to/data'

config = load_config("config.yaml")
config = substitute_env_vars(config)
```

### Configuration Templates

```{eval-rst}
.. autofunction:: config.utils.create_config_template

.. autofunction:: config.utils.update_config_template
```

Generate configuration templates:

```python
from jepa.config.utils import create_config_template

# Create template for vision task
template = create_config_template(
    task="vision",
    model_size="large",
    training_type="self_supervised"
)

# Save template
with open("my_config.yaml", "w") as f:
    yaml.dump(template, f)
```

## CLI Configuration

### Command Line Overrides

Override configuration values from the command line:

```bash
# Override single values
python -m cli train \
  --config config/base.yaml \
  --learning-rate 0.01 \
  --batch-size 128

# Override nested values
python -m cli train \
  --config config/base.yaml \
  --model.encoder_dim 1024 \
  --training.optimizer.type adamw
```

### Configuration Generation

```bash
# Generate default configuration
python -m cli config --create-default my_config.yaml

# Generate configuration for specific task
python -m cli config --create-template vision my_vision_config.yaml

# Validate configuration
python -m cli config --validate my_config.yaml

# Show current configuration
python -m cli config --show my_config.yaml
```

## Best Practices

### Configuration Organization

1. **Hierarchical Structure**: Organize configs by domain (model, training, data)
2. **Environment Separation**: Separate configs for dev, test, prod
3. **Version Control**: Track configuration changes with git
4. **Documentation**: Comment configuration options

### Configuration Management

```python
# Good: Use descriptive names
vision_large_config.yaml
nlp_bert_base_config.yaml
timeseries_lstm_config.yaml

# Good: Environment-specific configs
configs/
  base/
    model.yaml
    training.yaml
  environments/
    dev.yaml
    prod.yaml
  experiments/
    exp_001.yaml
    exp_002.yaml
```

### Parameter Sweeps

```python
from jepa.config import ConfigSweep

# Define parameter sweep
sweep = ConfigSweep({
    'model.encoder_dim': [256, 512, 768],
    'training.learning_rate': [1e-4, 1e-3, 1e-2],
    'training.batch_size': [32, 64, 128]
})

# Generate all combinations
for config in sweep.generate():
    train_model(config)
```

## Examples

### Basic Configuration Usage

```python
from jepa.config import load_config
from jepa.trainer import JEPATrainer

# Load configuration
config = load_config("config/my_experiment.yaml")

# Create trainer with config
trainer = JEPATrainer(config)
trainer.train()
```

### Dynamic Configuration

```python
from jepa.config import ExperimentConfig, ModelConfig

# Create configuration programmatically
config = ExperimentConfig(
    name="dynamic_experiment",
    model=ModelConfig(
        encoder_type="transformer",
        encoder_dim=768
    ),
    training=TrainingConfig(
        epochs=100,
        batch_size=64
    )
)

# Modify configuration
config.training.learning_rate = 1e-4
config.model.encoder_layers = 12

# Use configuration
trainer = JEPATrainer(config)
```

### Configuration Inheritance

```python
# base_config.yaml
model:
  encoder_type: "transformer"
  encoder_dim: 768
  
training:
  epochs: 100
  batch_size: 64

# experiment_config.yaml
base_config: "base_config.yaml"

# Override specific values
model:
  encoder_layers: 24  # Larger model
  
training:
  learning_rate: 1e-4  # Specific LR
```

For more examples and detailed usage, see the [Configuration Guide](../guides/configuration.md) and [Examples](../examples/index.md).
