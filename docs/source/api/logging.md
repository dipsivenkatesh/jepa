# Logging API

This section documents the centralized logging system.

## Base Logger

```{eval-rst}
.. automodule:: loggers.base_logger
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: loggers.base_logger.BaseLogger
   :members:
   :special-members: __init__
```

Base functionality for all loggers:

```python
from jepa.loggers.base_logger import BaseLogger

class CustomLogger(BaseLogger):
    def __init__(self, config):
        super().__init__(config)
        # Custom initialization
    
    def log_metrics(self, metrics, step=None):
        # Custom logging implementation
        pass
```

## Logger Registry

```{eval-rst}
.. autoclass:: loggers.base_logger.LoggerRegistry
   :members:
   :special-members: __init__
```

Manage available logger types:

```python
from jepa.loggers.base_logger import LoggerRegistry

# Register custom logger
LoggerRegistry.register("custom", CustomLogger)

# Create logger by name
logger = LoggerRegistry.create("wandb", config)
```

## Console Logger

```{eval-rst}
.. automodule:: loggers.console_logger
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: loggers.console_logger.ConsoleLogger
   :members:
   :special-members: __init__
```

Simple console-based logging:

```python
from jepa.loggers import ConsoleLogger

logger = ConsoleLogger({
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'colored': True
})

logger.log_metrics({'loss': 0.5, 'accuracy': 0.85}, step=100)
```

## Weights & Biases Logger

```{eval-rst}
.. automodule:: loggers.wandb_logger
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: loggers.wandb_logger.WandbLogger
   :members:
   :special-members: __init__
```

Integration with Weights & Biases:

```python
from jepa.loggers import WandbLogger

logger = WandbLogger({
    'project': 'jepa-experiments',
    'entity': 'my-team',
    'name': 'experiment-1',
    'tags': ['baseline', 'transformer'],
    'notes': 'Initial experiment with transformer encoder'
})

# Log metrics
logger.log_metrics({'train/loss': 0.5, 'train/lr': 1e-4}, step=100)

# Log model
logger.log_model(model, 'model_checkpoint')
```

## TensorBoard Logger

```{eval-rst}
.. automodule:: loggers.tensorboard_logger
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: loggers.tensorboard_logger.TensorBoardLogger
   :members:
   :special-members: __init__
```

TensorBoard integration for visualization:

```python
from jepa.loggers import TensorBoardLogger

logger = TensorBoardLogger({
    'log_dir': 'logs/tensorboard',
    'log_frequency': 10,
    'log_images': True,
    'log_histograms': True
})

# Log scalars
logger.log_metrics({'loss': 0.5}, step=100)

# Log images
logger.log_image('input_samples', images, step=100)

# Log histograms
logger.log_histogram('model.encoder.weight', model.encoder.weight, step=100)
```

## Multi Logger

```{eval-rst}
.. automodule:: loggers.multi_logger
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: loggers.multi_logger.MultiLogger
   :members:
   :special-members: __init__
```

Combine multiple logging backends:

```python
from jepa.loggers import MultiLogger

# List-based configuration
logger = MultiLogger([
    ('console', {'level': 'INFO'}),
    ('wandb', {'project': 'jepa'}),
    ('tensorboard', {'log_dir': 'logs'})
])

# Dictionary-based configuration
logger = MultiLogger({
    'backends': ['console', 'wandb', 'tensorboard'],
    'console': {'level': 'INFO'},
    'wandb': {'project': 'jepa'},
    'tensorboard': {'log_dir': 'logs'}
})

# Logs to all backends simultaneously
logger.log_metrics({'loss': 0.5}, step=100)
```

## Logger Factory

```{eval-rst}
.. autofunction:: loggers.multi_logger.create_logger
```

Create loggers from configuration:

```python
from jepa.loggers import create_logger

# From dictionary config
config = {
    'backends': ['wandb', 'tensorboard'],
    'wandb': {'project': 'jepa-experiments'},
    'tensorboard': {'log_dir': 'logs/tb'}
}

logger = create_logger(config)

# From YAML config
logger = create_logger("config/logging_config.yaml")
```

## Logging Utilities

### Metric Formatting

```{eval-rst}
.. autofunction:: loggers.utils.format_metrics

.. autofunction:: loggers.utils.flatten_metrics

.. autofunction:: loggers.utils.add_prefix
```

Utilities for metric processing:

```python
from jepa.loggers.utils import format_metrics, add_prefix

# Format metrics for display
formatted = format_metrics({'loss': 0.123456}, precision=4)
# Output: {'loss': '0.1235'}

# Add prefix to metrics
prefixed = add_prefix({'loss': 0.5, 'acc': 0.9}, 'train')
# Output: {'train/loss': 0.5, 'train/acc': 0.9}
```

### Logging Decorators

```{eval-rst}
.. autofunction:: loggers.decorators.log_execution_time

.. autofunction:: loggers.decorators.log_function_calls

.. autofunction:: loggers.decorators.log_exceptions
```

Decorators for automatic logging:

```python
from jepa.loggers.decorators import log_execution_time, log_exceptions

@log_execution_time(logger)
@log_exceptions(logger)
def training_step(batch):
    # Function execution time and exceptions are automatically logged
    return process_batch(batch)
```

## Advanced Logging Features

### Structured Logging

```{eval-rst}
.. autoclass:: loggers.structured.StructuredLogger
   :members:
   :special-members: __init__
```

Log structured data with schema validation:

```python
from jepa.loggers.structured import StructuredLogger

logger = StructuredLogger({
    'schema': {
        'metrics': {'type': 'dict', 'required': True},
        'metadata': {'type': 'dict', 'required': False}
    }
})

logger.log_structured({
    'metrics': {'loss': 0.5, 'accuracy': 0.9},
    'metadata': {'epoch': 10, 'batch_size': 64}
})
```

### Async Logging

```{eval-rst}
.. autoclass:: loggers.async_logger.AsyncLogger
   :members:
   :special-members: __init__
```

Non-blocking logging for high-throughput scenarios:

```python
from jepa.loggers.async_logger import AsyncLogger

logger = AsyncLogger({
    'backend': 'wandb',
    'buffer_size': 1000,
    'flush_interval': 30  # seconds
})

# Non-blocking log calls
logger.log_metrics({'loss': 0.5}, step=100)
```

### Conditional Logging

```{eval-rst}
.. autoclass:: loggers.conditional.ConditionalLogger
   :members:
   :special-members: __init__
```

Log based on conditions:

```python
from jepa.loggers.conditional import ConditionalLogger

logger = ConditionalLogger({
    'base_logger': wandb_logger,
    'conditions': {
        'min_step': 100,  # Only log after step 100
        'max_frequency': 0.1,  # Max 10% of calls
        'level_filter': 'INFO'  # Only INFO and above
    }
})
```

## Configuration

### Logging Configuration

```{eval-rst}
.. autoclass:: config.config.LoggingConfig
   :members:
   :special-members: __init__
```

Configure logging through YAML:

```yaml
logging:
  level: "INFO"
  backends: ["wandb", "tensorboard", "console"]
  
  # Backend-specific settings
  wandb:
    project: "jepa-experiments"
    entity: "my-team"
    tags: ["baseline", "transformer"]
    log_frequency: 10
    log_gradients: false
    log_model: true
  
  tensorboard:
    log_dir: "logs/tensorboard"
    log_frequency: 10
    log_images: true
    log_histograms: true
  
  console:
    level: "INFO"
    format: "%(asctime)s - %(levelname)s - %(message)s"
    colored: true
  
  # Global settings
  log_frequency: 10
  save_frequency: 50
```

## Custom Logger Development

### Creating Custom Loggers

```python
from jepa.loggers.base_logger import BaseLogger

class CustomAPILogger(BaseLogger):
    def __init__(self, config):
        super().__init__(config)
        self.api_endpoint = config.get('api_endpoint')
        self.api_key = config.get('api_key')
        self.session = requests.Session()
    
    def log_metrics(self, metrics, step=None):
        payload = {
            'metrics': metrics,
            'step': step,
            'timestamp': time.time()
        }
        
        response = self.session.post(
            self.api_endpoint,
            json=payload,
            headers={'Authorization': f'Bearer {self.api_key}'}
        )
        
        if not response.ok:
            self.logger.warning(f"Failed to log metrics: {response.status_code}")
    
    def finish(self):
        self.session.close()

# Register the custom logger
from jepa.loggers.base_logger import LoggerRegistry
LoggerRegistry.register('custom_api', CustomAPILogger)
```

### Logger Middleware

```python
from jepa.loggers.middleware import LoggerMiddleware

class MetricFilterMiddleware(LoggerMiddleware):
    def __init__(self, filter_patterns):
        self.filter_patterns = filter_patterns
    
    def process_metrics(self, metrics, step=None):
        filtered_metrics = {}
        for key, value in metrics.items():
            if any(pattern in key for pattern in self.filter_patterns):
                filtered_metrics[key] = value
        return filtered_metrics, step

# Apply middleware
logger = WandbLogger(config)
logger.add_middleware(MetricFilterMiddleware(['train', 'val']))
```

## Integration Examples

### Basic Logging Setup

```python
from jepa.loggers import create_logger

# Simple console logging
logger = create_logger({'backends': ['console']})

# Multi-backend logging
logger = create_logger({
    'backends': ['wandb', 'tensorboard', 'console'],
    'wandb': {'project': 'my-project'},
    'tensorboard': {'log_dir': 'logs'},
    'console': {'level': 'INFO'}
})

# Use in training
for epoch in range(epochs):
    for batch_idx, batch in enumerate(dataloader):
        loss = training_step(batch)
        
        if batch_idx % log_frequency == 0:
            logger.log_metrics({
                'train/loss': loss,
                'train/epoch': epoch,
                'train/step': global_step
            }, step=global_step)
```

### Advanced Logging with Callbacks

```python
from jepa.trainer.callbacks import LoggingCallback

class CustomLoggingCallback(LoggingCallback):
    def __init__(self, logger, log_frequency=10):
        super().__init__(logger, log_frequency)
        self.best_loss = float('inf')
    
    def on_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if batch_idx % self.log_frequency == 0:
            self.logger.log_metrics({
                'train/loss': outputs['loss'],
                'train/lr': trainer.optimizer.param_groups[0]['lr']
            }, step=trainer.global_step)
    
    def on_validation_end(self, trainer, model, metrics):
        self.logger.log_metrics(metrics, step=trainer.global_step)
        
        if metrics['val_loss'] < self.best_loss:
            self.best_loss = metrics['val_loss']
            self.logger.log_metrics({'best_val_loss': self.best_loss})

# Use with trainer
trainer = JEPATrainer(
    config,
    callbacks=[CustomLoggingCallback(logger)]
)
```

For more examples and detailed usage, see the [Logging Examples](../examples/logging.md) and [Training Guide](../guides/training.md).
