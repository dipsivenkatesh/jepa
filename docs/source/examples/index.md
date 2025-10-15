# Examples

This section contains practical examples showing how to use JEPA for various tasks and scenarios.

## Basic Examples

### Hello World Training

The simplest possible JEPA training example:

```eval-rst
.. literalinclude:: ../../examples/usage_example.py
   :language: python
   :caption: Basic JEPA Usage
```

### CLI Training Example

Training using the command-line interface:

```eval-rst
.. literalinclude:: ../../examples/cli_example.py
   :language: python
   :caption: CLI Training Example
```

## Training Examples

### Custom Training Loop

Implementing a custom training loop with full control:

```eval-rst
.. literalinclude:: ../../examples/training_example.py
   :language: python
   :caption: Custom Training Loop
```

### Advanced Training Configuration

Example with advanced training features:

```python
from jepa.trainer import JEPATrainer
from config.config import load_config
from loggers.multi_logger import create_logger

# Load advanced configuration
config = load_config("config/advanced_config.yaml")

# Setup logging
logger = create_logger(
    backends=["wandb", "tensorboard"],
    project="jepa-advanced",
    name="experiment-1"
)

# Advanced trainer with callbacks
trainer = JEPATrainer(
    config=config,
    logger=logger,
    callbacks=[
        EarlyStopping(patience=10),
        ModelCheckpoint(save_top_k=3),
        LearningRateMonitor()
    ]
)

# Train with custom options
trainer.train(
    resume_from_checkpoint="checkpoints/last.ckpt",
    max_epochs=100,
    limit_train_batches=0.8  # Use 80% of training data
)
```

## Data Examples

### Custom Dataset

Creating a custom dataset for JEPA:

```eval-rst
.. literalinclude:: ../../examples/data_example.py
   :language: python
   :caption: Custom Dataset Example
```

### Hugging Face Integration

Using Hugging Face datasets:

```eval-rst
.. literalinclude:: ../../examples/hf_example.py
   :language: python
   :caption: Hugging Face Integration
```

## Domain-Specific Examples

### Computer Vision

Training JEPA on image data:

```python
# Vision configuration
vision_config = {
    "model": {
        "encoder_type": "vision_transformer",
        "patch_size": 16,
        "encoder_dim": 768,
        "encoder_layers": 12,
        "encoder_heads": 12
    },
    "data": {
        "dataset": "imagenet",
        "image_size": 224,
        "transforms": {
            "resize": 256,
            "center_crop": 224,
            "normalize": True,
            "augmentation": {
                "horizontal_flip": 0.5,
                "color_jitter": 0.1,
                "random_crop": True
            }
        },
        "masking": {
            "strategy": "block",
            "mask_ratio": 0.15,
            "block_size": 16
        }
    },
    "training": {
        "epochs": 100,
        "batch_size": 256,
        "learning_rate": 1e-4,
        "optimizer": "adamw",
        "scheduler": "cosine"
    }
}

# Train vision model
trainer = JEPATrainer(vision_config)
trainer.train()
```

### Natural Language Processing

Training JEPA on text data:

```python
# NLP configuration
nlp_config = {
    "model": {
        "encoder_type": "transformer",
        "vocab_size": 50000,
        "encoder_dim": 512,
        "encoder_layers": 8,
        "encoder_heads": 8,
        "max_length": 512
    },
    "data": {
        "dataset": "bookcorpus",
        "tokenizer": "bert-base-uncased",
        "masking": {
            "strategy": "random",
            "mask_ratio": 0.15,
            "mask_token": "[MASK]",
            "random_token_prob": 0.1
        }
    },
    "training": {
        "epochs": 50,
        "batch_size": 128,
        "learning_rate": 5e-5,
        "warmup_steps": 10000
    }
}

# Train NLP model
trainer = JEPATrainer(nlp_config)
trainer.train()
```

### Time Series Forecasting

Training JEPA on sequential data:

```python
# Time series configuration
timeseries_config = {
    "model": {
        "encoder_type": "temporal_cnn",
        "input_dim": 10,
        "encoder_dim": 256,
        "num_layers": 6,
        "kernel_size": 3
    },
    "data": {
        "dataset": "electricity",
        "window_size": 168,  # 1 week of hourly data
        "stride": 24,        # 1 day stride
        "normalization": "z-score",
        "masking": {
            "strategy": "contiguous",
            "mask_ratio": 0.15,
            "min_mask_length": 12,
            "max_mask_length": 48
        }
    },
    "training": {
        "epochs": 200,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "scheduler": "plateau"
    }
}

# Train time series model
trainer = JEPATrainer(timeseries_config)
trainer.train()
```

## Logging Examples

### Weights & Biases Integration

```eval-rst
.. literalinclude:: ../../examples/wandb_example.py
   :language: python
   :caption: W&B Integration Example
```

### TensorBoard Logging

```eval-rst
.. literalinclude:: ../../examples/logging_example.py
   :language: python
   :caption: TensorBoard Logging
```

### Multi-Backend Logging

```python
from loggers import create_logger

# Setup multiple logging backends
logger = create_logger([
    {
        "type": "wandb",
        "project": "jepa-experiments",
        "entity": "my-team",
        "tags": ["vision", "large-scale"]
    },
    {
        "type": "tensorboard",
        "log_dir": "logs/tensorboard"
    },
    {
        "type": "console",
        "level": "INFO"
    }
])

# Use with trainer
trainer = JEPATrainer(config, logger=logger)
trainer.train()
```

## Evaluation Examples

### Model Evaluation

```python
from jepa.trainer.eval import JEPAEvaluator

# Load trained model
evaluator = JEPAEvaluator.load_from_checkpoint(
    "checkpoints/best_model.ckpt"
)

# Evaluate on test set
test_results = evaluator.evaluate(test_dataloader)
print(f"Test Loss: {test_results['loss']:.4f}")

# Evaluate on custom metrics
custom_results = evaluator.evaluate(
    test_dataloader,
    metrics=["reconstruction_error", "representation_quality"]
)
```

### Transfer Learning

```python
# Load pre-trained JEPA model
pretrained_model = JEPAModel.load_from_checkpoint(
    "checkpoints/pretrained_jepa.ckpt"
)

# Extract encoder for downstream task
encoder = pretrained_model.encoder

# Fine-tune on downstream task
downstream_model = DownstreamClassifier(encoder)
downstream_trainer = Trainer(downstream_model)
downstream_trainer.train(downstream_dataloader)
```

## Advanced Examples

### Multi-GPU Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize distributed training
dist.init_process_group(backend="nccl")

# Setup distributed model
model = JEPAModel(config)
model = DistributedDataParallel(model)

# Distributed trainer
trainer = JEPATrainer(
    model=model,
    config=config,
    distributed=True
)

trainer.train()
```

### Mixed Precision Training

```python
from torch.cuda.amp import GradScaler, autocast

# Setup mixed precision
scaler = GradScaler()

class MixedPrecisionTrainer(JEPATrainer):
    def training_step(self, batch):
        with autocast():
            loss = super().training_step(batch)
        
        # Scale loss and backward
        scaler.scale(loss).backward()
        scaler.step(self.optimizer)
        scaler.update()
        
        return loss
```

### Custom Loss Functions

```python
class ContrastiveJEPALoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, predicted, target, negatives):
        # Positive similarity
        pos_sim = F.cosine_similarity(predicted, target, dim=-1)
        
        # Negative similarities
        neg_sim = F.cosine_similarity(
            predicted.unsqueeze(1), 
            negatives, 
            dim=-1
        )
        
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        logits = logits / self.temperature
        
        labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.long)
        
        return F.cross_entropy(logits, labels)

# Use custom loss
trainer = JEPATrainer(
    config=config,
    loss_fn=ContrastiveJEPALoss(temperature=0.07)
)
```

## Configuration Examples

### Production Configuration

```yaml
# production_config.yaml
model:
  encoder_type: "transformer"
  encoder_dim: 1024
  encoder_layers: 24
  encoder_heads: 16
  dropout: 0.1

training:
  epochs: 1000
  batch_size: 256
  learning_rate: 1e-4
  weight_decay: 1e-4
  gradient_clip_norm: 1.0
  
  optimizer:
    type: "adamw"
    betas: [0.9, 0.999]
    eps: 1e-8
  
  scheduler:
    type: "cosine"
    warmup_epochs: 50
    min_lr: 1e-6

data:
  batch_size: 256
  num_workers: 16
  pin_memory: true
  persistent_workers: true

logging:
  backends: ["wandb", "tensorboard"]
  save_frequency: 10
  
  wandb:
    project: "production-jepa"
    entity: "my-org"
```

### Development Configuration

```yaml
# dev_config.yaml
model:
  encoder_dim: 128
  encoder_layers: 2

training:
  epochs: 5
  batch_size: 8
  learning_rate: 1e-3

data:
  num_workers: 2
  limit_train_batches: 100
  limit_val_batches: 20

logging:
  backends: ["console"]
  level: "DEBUG"
```

## Structured Data Examples

Working with structured/tabular data:

```eval-rst
.. literalinclude:: ../../examples/structured_data_example.py
   :language: python
   :caption: Structured Data Example
```

## Performance Optimization

### Memory Optimization

```python
# Gradient checkpointing
model.gradient_checkpointing = True

# Optimized data loading
dataloader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)

# Memory-efficient attention
torch.backends.cuda.enable_flash_sdp(True)
```

### Speed Optimization

```python
# Compile model (PyTorch 2.0+)
model = torch.compile(model)

# Optimized loss computation
def optimized_loss(pred, target):
    # Use vectorized operations
    return F.mse_loss(pred, target, reduction='mean')
```

## Integration Examples

### FastAPI Service

```python
from fastapi import FastAPI
import torch

app = FastAPI()

# Load trained model
model = JEPAModel.load_from_checkpoint("model.ckpt")
model.eval()

@app.post("/encode")
async def encode_data(data: dict):
    with torch.no_grad():
        # Preprocess input
        input_tensor = preprocess(data)
        
        # Get embeddings
        embeddings = model.encoder(input_tensor)
        
        return {"embeddings": embeddings.tolist()}
```

### Jupyter Notebook

For interactive examples, see the [Jupyter notebooks](https://github.com/your-org/jepa/tree/main/notebooks) in the repository.

## Running Examples

To run any of these examples:

1. **Setup environment**:
   ```bash
   pip install -e .
   pip install -r requirements.txt
   ```

2. **Run basic example**:
   ```bash
   python examples/usage_example.py
   ```

3. **Run with configuration**:
   ```bash
   python examples/training_example.py --config configs/example_config.yaml
   ```

4. **CLI examples**:
   ```bash
   python -m jepa.cli train --config examples/configs/vision.yaml
   ```

## Getting Help

- **Documentation**: Read the [API reference](../api/index.md)
- **Tutorials**: Check out the [guides](../guides/quickstart.md)
- **Community**: Join our [discussions](https://github.com/your-org/jepa/discussions)
- **Issues**: Report bugs on [GitHub](https://github.com/your-org/jepa/issues)

## Contributing Examples

We welcome contributions of new examples! Please:

1. Follow the existing code style
2. Include proper documentation
3. Add configuration files
4. Test your examples
5. Submit a pull request

See our [contribution guidelines](https://github.com/your-org/jepa/blob/main/CONTRIBUTING.md) for details.
