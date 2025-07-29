# Training Guide

This guide covers everything you need to know about training JEPA models effectively.

## Training Overview

JEPA training follows these key principles:

1. **Self-Supervised Learning**: No labeled data required
2. **Context-Target Prediction**: Learn by predicting masked regions
3. **Joint Embedding Space**: Shared representations for context and targets
4. **Scalable Architecture**: Works from small to very large models

## Training Process

The JEPA training loop consists of:

1. **Data Loading**: Load and preprocess input data
2. **Masking**: Create context and target regions
3. **Encoding**: Encode context and targets separately
4. **Prediction**: Predict target embeddings from context
5. **Loss Computation**: Compare predicted and actual target embeddings
6. **Optimization**: Update model parameters

## Basic Training

### Simple Training Script

```python
from trainer.trainer import JEPATrainer
from config.config import load_config

# Load configuration
config = load_config("config/default_config.yaml")

# Create trainer
trainer = JEPATrainer(config)

# Train the model
trainer.train()
```

### CLI Training

```bash
# Basic training
python -m cli train --config my_config.yaml

# With specific parameters
python -m cli train \
  --config my_config.yaml \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 0.001
```

## Advanced Training Techniques

### Distributed Training

```yaml
training:
  distributed: true
  world_size: 4        # Number of GPUs
  backend: "nccl"      # Communication backend
```

```bash
# Multi-GPU training
torchrun --nproc_per_node=4 -m cli train --config config.yaml
```

### Mixed Precision Training

```yaml
training:
  mixed_precision: true
  grad_scaler: true
  amp_backend: "native"  # or "apex"
```

### Gradient Accumulation

```yaml
training:
  batch_size: 32           # Effective batch size
  micro_batch_size: 8      # Actual batch size per step
  gradient_accumulation_steps: 4  # 32/8 = 4
```

### Learning Rate Scheduling

```yaml
training:
  scheduler:
    type: "cosine"
    warmup_epochs: 10
    min_lr: 1e-6
    
  # Or step scheduling
  scheduler:
    type: "step"
    step_size: 30
    gamma: 0.1
```

## Loss Functions

### Standard JEPA Loss

```python
def jepa_loss(predicted_targets, actual_targets, mask=None):
    """
    Standard JEPA prediction loss
    """
    loss = F.mse_loss(predicted_targets, actual_targets, reduction='none')
    if mask is not None:
        loss = loss * mask.unsqueeze(-1)
    return loss.mean()
```

### Contrastive Loss

```yaml
training:
  loss:
    type: "contrastive"
    temperature: 0.1
    negative_samples: 64
```

### Multi-Scale Loss

```yaml
training:
  loss:
    type: "multiscale"
    scales: [1, 2, 4]
    weights: [1.0, 0.5, 0.25]
```

## Optimization Strategies

### Optimizer Selection

```yaml
training:
  optimizer:
    type: "adamw"
    learning_rate: 0.001
    weight_decay: 1e-4
    betas: [0.9, 0.999]
```

### Learning Rate Finding

```python
from trainer.utils import find_learning_rate

# Find optimal learning rate
trainer = JEPATrainer(config)
optimal_lr = find_learning_rate(trainer, min_lr=1e-6, max_lr=1e-1)
print(f"Optimal LR: {optimal_lr}")
```

### Gradient Clipping

```yaml
training:
  gradient_clip_norm: 1.0     # Clip by norm
  gradient_clip_value: 0.5    # Clip by value
```

## Training Monitoring

### Logging Configuration

```yaml
logging:
  backends: ["wandb", "tensorboard", "console"]
  log_frequency: 10
  
  wandb:
    project: "jepa-training"
    tags: ["experiment-1"]
    
  tensorboard:
    log_dir: "logs/tensorboard"
```

### Key Metrics to Track

- **Training Loss**: Should decrease over time
- **Validation Loss**: Should decrease without overfitting
- **Learning Rate**: Track scheduling
- **Gradient Norm**: Monitor for gradient explosion
- **Memory Usage**: Ensure efficient GPU utilization

### Early Stopping

```yaml
training:
  early_stopping:
    patience: 10
    min_delta: 1e-4
    monitor: "val_loss"
    mode: "min"
```

## Checkpointing and Resuming

### Automatic Checkpointing

```yaml
training:
  save_frequency: 10        # Save every 10 epochs
  save_top_k: 3            # Keep best 3 checkpoints
  checkpoint_dir: "checkpoints/"
```

### Resume Training

```bash
python -m cli train \
  --config my_config.yaml \
  --resume checkpoints/epoch_50.pth
```

### Custom Checkpointing

```python
# Save custom checkpoint
trainer.save_checkpoint("my_checkpoint.pth", include_optimizer=True)

# Load checkpoint
trainer.load_checkpoint("my_checkpoint.pth", load_optimizer=True)
```

## Data-Specific Training

### Vision Training

```yaml
model:
  encoder_type: "vision_transformer"
  patch_size: 16
  
data:
  transforms:
    resize: [224, 224]
    normalize: true
    augmentation:
      horizontal_flip: 0.5
      rotation: 15
```

### NLP Training

```yaml
model:
  encoder_type: "transformer"
  vocab_size: 50000
  
data:
  tokenizer: "bert-base-uncased"
  max_length: 512
  masking:
    mask_ratio: 0.15
    random_token_prob: 0.1
```

### Time Series Training

```yaml
model:
  encoder_type: "temporal_cnn"
  
data:
  window_size: 100
  stride: 50
  masking:
    strategy: "contiguous"
    mask_length: 20
```

## Performance Optimization

### Memory Optimization

```yaml
training:
  # Use gradient checkpointing
  gradient_checkpointing: true
  
  # Optimize data loading
  data:
    num_workers: 8
    pin_memory: true
    prefetch_factor: 2
```

### Speed Optimization

```python
# Use compiled models (PyTorch 2.0+)
model = torch.compile(model)

# Enable optimized attention
torch.backends.cuda.enable_flash_sdp(True)
```

### Batch Size Optimization

```python
# Find optimal batch size
from trainer.utils import find_optimal_batch_size

optimal_batch_size = find_optimal_batch_size(
    model, 
    dataloader, 
    device="cuda"
)
```

## Hyperparameter Tuning

### Grid Search

```yaml
# hyperparams.yaml
sweep:
  learning_rate: [0.001, 0.01, 0.1]
  batch_size: [32, 64, 128]
  encoder_dim: [256, 512, 1024]
```

### Random Search

```python
import wandb

# Initialize sweep
sweep_config = {
    'method': 'random',
    'parameters': {
        'learning_rate': {'values': [0.001, 0.01, 0.1]},
        'batch_size': {'values': [32, 64, 128]}
    }
}

sweep_id = wandb.sweep(sweep_config)
wandb.agent(sweep_id, function=train_model)
```

## Troubleshooting

### Common Issues

**Loss not decreasing**
- Check learning rate (try lower values)
- Verify data preprocessing
- Check for gradient clipping

**GPU out of memory**
- Reduce batch size
- Enable gradient checkpointing
- Use mixed precision

**Training too slow**
- Increase batch size
- Use more workers
- Enable model compilation

**Model overfitting**
- Add dropout
- Reduce model size
- Use data augmentation

### Debugging Tips

```python
# Debug mode
config.debug = True
config.training.epochs = 1
config.training.batch_size = 2

# Log gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

## Best Practices

1. **Start Simple**: Begin with small models and datasets
2. **Monitor Closely**: Watch loss curves and metrics
3. **Save Often**: Regular checkpointing prevents data loss
4. **Experiment**: Try different architectures and hyperparameters
5. **Document**: Keep detailed logs of experiments
6. **Validate**: Always use validation data for model selection

## Advanced Topics

### Custom Training Loops

```python
class CustomJEPATrainer(JEPATrainer):
    def training_step(self, batch, batch_idx):
        # Custom training logic
        context, targets, mask = self.prepare_batch(batch)
        
        # Encode
        context_emb = self.model.encoder(context)
        target_emb = self.model.encoder(targets)
        
        # Predict
        predicted_targets = self.model.predictor(context_emb, mask)
        
        # Compute loss
        loss = self.criterion(predicted_targets, target_emb)
        
        return loss
```

### Multi-Task Training

```yaml
training:
  tasks:
    - name: "reconstruction"
      weight: 1.0
      loss: "mse"
    - name: "contrastive"
      weight: 0.5
      loss: "infonce"
```

## Examples

For complete training examples, see:
- [Vision Training](../examples/vision.md)
- [NLP Training](../examples/nlp.md)
- [Time Series Training](../examples/timeseries.md)
- [Multi-Modal Training](../examples/multimodal.md)
