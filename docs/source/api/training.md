# Training API

This page documents the primary training interfaces exposed by the ``jepa.trainer`` package.

## Trainer utilities

```{eval-rst}
.. automodule:: jepa.trainer
   :members: JEPATrainer, JEPAEvaluator, create_trainer
   :undoc-members:
   :show-inheritance:
```

### Creating a trainer

```python
import torch
from torch.utils.data import DataLoader

from jepa.models import JEPA
from jepa.models.encoder import Encoder
from jepa.models.predictor import Predictor
from jepa.trainer import create_trainer

encoder = Encoder(hidden_dim=256)
predictor = Predictor(hidden_dim=256)
model = JEPA(encoder=encoder, predictor=predictor)

train_dataset = ...  # yields (state_t, state_t1) pairs
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

trainer = create_trainer(model, learning_rate=5e-4)
trainer.train(train_loader, num_epochs=50)
```

The convenience ``create_trainer`` helper wires up a default ``AdamW`` optimizer and a cosine scheduler. For full control, instantiate ``JEPATrainer`` directly by providing your own optimizer, scheduler, and logger instances.

### Distributed training

```python
trainer = create_trainer(
    model,
    distributed=True,
    world_size=int(os.environ["WORLD_SIZE"]),
    local_rank=int(os.environ.get("LOCAL_RANK", 0)),
)
```

With ``distributed=True`` the trainer automatically wraps the model in ``DistributedDataParallel``, synchronizes losses across workers, and limits logging/checkpointing to rank zero.

Launch multi-GPU runs with ``torchrun``:

```bash
torchrun --nproc_per_node=4 python -m jepa.cli train --config config/train.yaml --distributed true
```

### Saving and loading checkpoints

```python
trainer.save_checkpoint("checkpoint_epoch_10.pt")
trainer.load_checkpoint("checkpoint_epoch_10.pt")
```

For lightweight inference artifacts, use the HuggingFace-style helpers on any ``BaseModel`` subclass:

```python
model.save_pretrained("artifacts/jepa-small")
restored = JEPA.from_pretrained("artifacts/jepa-small", encoder=encoder, predictor=predictor)
restored.eval()
```

## Evaluator

```{eval-rst}
.. automodule:: jepa.trainer.eval
   :members: JEPAEvaluator
   :undoc-members:
   :show-inheritance:
```

Instantiate directly or from a checkpoint:

```python
from jepa.trainer.eval import JEPAEvaluator

evaluator = JEPAEvaluator.from_checkpoint("checkpoints/best_model.pt")
metrics = evaluator.evaluate(test_loader)
```

## Supporting utilities

```{eval-rst}
.. automodule:: jepa.trainer.utils
   :members: count_parameters, plot_training_history, save_training_config, load_training_config, create_data_splits, setup_reproducibility, get_device_info, log_model_summary
   :undoc-members:
   :show-inheritance:
```

These helpers provide quick access to parameter counts, plotting, configuration persistence, dataset splitting, and device inspection.
