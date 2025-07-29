# Training API

This section documents the training and evaluation modules.

## Core Trainer

```{eval-rst}
.. automodule:: trainer.trainer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: trainer.trainer.JEPATrainer
   :members:
   :special-members: __init__
```

The main training class for JEPA models:

```python
from jepa.trainer import JEPATrainer
from jepa.config import load_config

# Create trainer from config
config = load_config("config/default_config.yaml")
trainer = JEPATrainer(config)

# Train the model
trainer.train()

# Evaluate the model
metrics = trainer.evaluate()
```

## Evaluation

```{eval-rst}
.. automodule:: trainer.eval
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: trainer.eval.JEPAEvaluator
   :members:
   :special-members: __init__
```

Comprehensive model evaluation:

```python
from jepa.trainer.eval import JEPAEvaluator

# Create evaluator
evaluator = JEPAEvaluator.from_checkpoint("checkpoints/best_model.pth")

# Evaluate on test set
test_metrics = evaluator.evaluate(test_dataloader)

# Generate embeddings
embeddings = evaluator.encode(data_samples)

# Downstream task evaluation
downstream_results = evaluator.evaluate_downstream(
    task="classification",
    train_data=train_data,
    test_data=test_data
)
```

## Training Utilities

```{eval-rst}
.. automodule:: trainer.utils
   :members:
   :undoc-members:
   :show-inheritance:
```

### Training Helpers

```{eval-rst}
.. autofunction:: trainer.utils.setup_training

.. autofunction:: trainer.utils.create_optimizer

.. autofunction:: trainer.utils.create_scheduler
```

Utilities for setting up training:

```python
from jepa.trainer.utils import setup_training, create_optimizer

# Setup training environment
device, model, dataloader = setup_training(config)

# Create optimizer
optimizer = create_optimizer(
    model.parameters(),
    optimizer_type="adamw",
    learning_rate=1e-4,
    weight_decay=1e-2
)
```

### Checkpoint Management

```{eval-rst}
.. autofunction:: trainer.utils.save_checkpoint

.. autofunction:: trainer.utils.load_checkpoint

.. autofunction:: trainer.utils.get_best_checkpoint
```

Handle model checkpoints:

```python
from jepa.trainer.utils import save_checkpoint, load_checkpoint

# Save checkpoint
save_checkpoint(
    model, optimizer, scheduler, epoch, loss,
    checkpoint_path="checkpoints/epoch_10.pth"
)

# Load checkpoint
model, optimizer, scheduler, epoch = load_checkpoint(
    "checkpoints/epoch_10.pth", model, optimizer, scheduler
)
```

### Training Metrics

```{eval-rst}
.. autofunction:: trainer.utils.compute_metrics

.. autofunction:: trainer.utils.log_metrics

.. autofunction:: trainer.utils.track_progress
```

Track and log training progress:

```python
from jepa.trainer.utils import compute_metrics, log_metrics

# Compute evaluation metrics
metrics = compute_metrics(predictions, targets, metric_types=["mse", "cosine_sim"])

# Log metrics to various backends
log_metrics(metrics, step=global_step, logger=logger)
```

## Advanced Training Features

### Distributed Training

```{eval-rst}
.. autoclass:: trainer.distributed.DistributedTrainer
   :members:
   :special-members: __init__
```

Multi-GPU and multi-node training:

```python
from jepa.trainer.distributed import DistributedTrainer

# Initialize distributed training
trainer = DistributedTrainer(
    config,
    world_size=4,
    rank=local_rank
)

# Train with data parallel
trainer.train()
```

### Mixed Precision Training

```{eval-rst}
.. autoclass:: trainer.mixed_precision.MixedPrecisionTrainer
   :members:
   :special-members: __init__
```

Automatic mixed precision for faster training:

```python
from jepa.trainer.mixed_precision import MixedPrecisionTrainer

trainer = MixedPrecisionTrainer(
    config,
    use_amp=True,
    grad_scaler=True
)
```

### Curriculum Learning

```{eval-rst}
.. autoclass:: trainer.curriculum.CurriculumTrainer
   :members:
   :special-members: __init__
```

Progressive difficulty training:

```python
from jepa.trainer.curriculum import CurriculumTrainer

trainer = CurriculumTrainer(
    config,
    curriculum_schedule="linear",
    difficulty_levels=5
)
```

## Custom Training Loops

### Basic Custom Trainer

```python
from jepa.trainer.trainer import BaseTrainer

class CustomTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.custom_loss = CustomLoss()
    
    def training_step(self, batch, batch_idx):
        context, target = batch
        pred, target_emb = self.model(context, target)
        
        # Custom loss computation
        loss = self.custom_loss(pred, target_emb)
        
        # Additional regularization
        reg_loss = self.compute_regularization()
        total_loss = loss + 0.1 * reg_loss
        
        return {
            'loss': total_loss,
            'main_loss': loss,
            'reg_loss': reg_loss
        }
    
    def validation_step(self, batch, batch_idx):
        # Custom validation logic
        pass
```

### Multi-Task Training

```python
from jepa.trainer.multitask import MultiTaskTrainer

class JEPAMultiTaskTrainer(MultiTaskTrainer):
    def __init__(self, config, tasks):
        super().__init__(config)
        self.tasks = tasks
        self.task_weights = config.task_weights
    
    def compute_loss(self, outputs, targets):
        total_loss = 0
        losses = {}
        
        for task_name, task in self.tasks.items():
            task_loss = task.compute_loss(
                outputs[task_name], 
                targets[task_name]
            )
            weight = self.task_weights.get(task_name, 1.0)
            total_loss += weight * task_loss
            losses[f"{task_name}_loss"] = task_loss
        
        losses['total_loss'] = total_loss
        return losses
```

## Training Callbacks

### Base Callback System

```{eval-rst}
.. autoclass:: trainer.callbacks.BaseCallback
   :members:
   :special-members: __init__
```

### Built-in Callbacks

```{eval-rst}
.. autoclass:: trainer.callbacks.EarlyStopping
   :members:
   :special-members: __init__

.. autoclass:: trainer.callbacks.ModelCheckpoint
   :members:
   :special-members: __init__

.. autoclass:: trainer.callbacks.LearningRateScheduler
   :members:
   :special-members: __init__
```

Use callbacks for training control:

```python
from jepa.trainer.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    ),
    ModelCheckpoint(
        dirpath='checkpoints/',
        filename='best_model',
        monitor='val_loss',
        save_top_k=3
    )
]

trainer = JEPATrainer(config, callbacks=callbacks)
```

### Custom Callbacks

```python
from jepa.trainer.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def on_epoch_start(self, trainer, model):
        # Custom logic at epoch start
        print(f"Starting epoch {trainer.current_epoch}")
    
    def on_batch_end(self, trainer, model, outputs, batch, batch_idx):
        # Custom logic after each batch
        if batch_idx % 100 == 0:
            self.log_intermediate_results(outputs)
    
    def on_validation_end(self, trainer, model, metrics):
        # Custom validation logic
        if metrics['val_accuracy'] > self.best_accuracy:
            self.save_best_model(model)
```

## Loss Functions

### JEPA Losses

```{eval-rst}
.. automodule:: trainer.losses
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: trainer.losses.JEPALoss
   :members:
   :special-members: __init__

.. autoclass:: trainer.losses.ContrastiveLoss
   :members:
   :special-members: __init__

.. autoclass:: trainer.losses.ReconstructionLoss
   :members:
   :special-members: __init__
```

Different loss formulations for JEPA:

```python
from jepa.trainer.losses import ContrastiveLoss, ReconstructionLoss

# Contrastive learning loss
contrastive_loss = ContrastiveLoss(
    temperature=0.1,
    negative_sampling='random'
)

# Direct reconstruction loss
reconstruction_loss = ReconstructionLoss(
    loss_type='mse',
    reduction='mean'
)
```

## Optimizers and Schedulers

### Optimizer Factory

```{eval-rst}
.. autofunction:: trainer.optimizers.create_optimizer
```

Create optimizers with proper configurations:

```python
from jepa.trainer.optimizers import create_optimizer

optimizer = create_optimizer(
    model.parameters(),
    optimizer_type="adamw",
    learning_rate=1e-4,
    weight_decay=1e-2,
    betas=(0.9, 0.999)
)
```

### Scheduler Factory

```{eval-rst}
.. autofunction:: trainer.schedulers.create_scheduler
```

Create learning rate schedulers:

```python
from jepa.trainer.schedulers import create_scheduler

scheduler = create_scheduler(
    optimizer,
    scheduler_type="cosine",
    max_epochs=100,
    warmup_epochs=10,
    min_lr=1e-6
)
```

## Configuration

### Training Configuration

```{eval-rst}
.. autoclass:: config.config.TrainingConfig
   :members:
   :special-members: __init__
```

Configure training through YAML:

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
  gradient_accumulation: 1
  
  validation:
    frequency: 5
    patience: 10
```

## Examples

### Basic Training

```python
from jepa.trainer import JEPATrainer
from jepa.config import load_config

# Load configuration
config = load_config("config/training_config.yaml")

# Create and train
trainer = JEPATrainer(config)
history = trainer.train()

# Evaluate
test_metrics = trainer.evaluate(test_dataloader)
```

### Advanced Training with Callbacks

```python
from jepa.trainer import JEPATrainer
from jepa.trainer.callbacks import *

trainer = JEPATrainer(
    config,
    callbacks=[
        EarlyStopping(patience=10),
        ModelCheckpoint(save_top_k=3),
        LearningRateScheduler(schedule="cosine"),
        CustomLogging(log_frequency=100)
    ]
)

trainer.train()
```

### Distributed Training

```python
import torch.distributed as dist
from jepa.trainer.distributed import DistributedTrainer

# Initialize distributed
dist.init_process_group(backend='nccl')

# Create distributed trainer
trainer = DistributedTrainer(
    config,
    world_size=torch.distributed.get_world_size(),
    rank=torch.distributed.get_rank()
)

trainer.train()
```

For more examples and detailed usage, see the [Training Guide](../guides/training.md) and [Examples](../examples/index.md).
