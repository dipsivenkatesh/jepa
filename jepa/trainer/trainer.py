"""
JEPA Trainer - A flexible training module for JEPA models.

This trainer is designed to work with any JEPA model configuration and provides
a clean, reusable interface for training joint-embedding predictive architectures.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from typing import Dict, Any, Optional, Callable, Union
import os
from copy import deepcopy
import logging
from collections.abc import Mapping
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:  # Progress bar support
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None  # type: ignore

try:  # Optional dependency for HuggingFace compatibility
    from transformers import PreTrainedModel
except ImportError:  # pragma: no cover - transformers is optional
    PreTrainedModel = None  # type: ignore

from ..loggers import create_logger, BaseLogger, WandbLogger
from ..loss_functions import mse_loss


LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class JEPATrainer:
    """
    A flexible trainer for JEPA models that can work with any encoder/predictor combination.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "auto",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        gradient_clip_norm: Optional[float] = None,
        log_interval: int = 100,
        save_dir: str = "./checkpoints",
        logger: Optional[BaseLogger] = None,
        loss_fn: Optional[LossFunction] = None,
        progress_bar: bool = True,
        distributed: bool = False,
        local_rank: Optional[int] = None,
        global_rank: Optional[int] = None,
        world_size: Optional[int] = None,
        dist_backend: str = "nccl",
        sync_batchnorm: bool = False,
        logger_project: Optional[str] = None,
        logger_run_name: Optional[str] = None,
    ):
        """
        Initialize the JEPA trainer.
        
        Args:
            model: JEPA model instance
            optimizer: PyTorch optimizer
            device: Device to train on ("auto", "cuda", "cpu")
            scheduler: Optional learning rate scheduler
            gradient_clip_norm: Optional gradient clipping value
            log_interval: How often to log training progress
            save_dir: Directory to save checkpoints
            logger: Centralized logger instance
            loss_fn: Optional callable that computes loss from prediction and target tensors
            progress_bar: Enable tqdm-based progress bars during training and validation
            distributed: Enable torch.distributed distributed data parallel training
            local_rank: Local process rank (per node) when using distributed training
            global_rank: Global process rank when using distributed training
            world_size: Total number of processes participating in training
            dist_backend: Backend to use for distributed communication ("nccl" or "gloo")
            sync_batchnorm: Convert BatchNorm layers to SyncBatchNorm when distributed
            logger_project: Optional experiment project name (e.g. for Weights & Biases)
            logger_run_name: Optional run name/alias for logger backends that support it
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_clip_norm = gradient_clip_norm
        self.log_interval = log_interval
        self.save_dir = save_dir
        self.custom_logger = self._initialize_logger(logger, logger_project, logger_run_name)
        self._progress_bar_requested = progress_bar
        self.distributed = distributed
        if distributed:
            if local_rank is None:
                env_local_rank = os.environ.get("LOCAL_RANK")
                self.local_rank = int(env_local_rank) if env_local_rank is not None else 0
            else:
                self.local_rank = local_rank
            if global_rank is None:
                env_rank = os.environ.get("RANK")
                self.global_rank = int(env_rank) if env_rank is not None else 0
            else:
                self.global_rank = global_rank
        else:
            self.local_rank = None
            self.global_rank = 0
        if distributed:
            if world_size is not None:
                self.world_size = world_size
            else:
                env_world_size = os.environ.get("WORLD_SIZE")
                self.world_size = int(env_world_size) if env_world_size is not None else None
        else:
            self.world_size = 1
        self.dist_backend = dist_backend
        self.sync_batchnorm = sync_batchnorm
        self.is_hf_model = bool(PreTrainedModel and isinstance(model, PreTrainedModel))
        if loss_fn is None and not self.is_hf_model:
            self.loss_fn = mse_loss
        else:
            self.loss_fn = loss_fn

        # Set device
        self.device = self._resolve_device(device)
        self._init_distributed_if_needed()

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.use_progress_bar = bool(self._progress_bar_requested and tqdm is not None)
        if self._progress_bar_requested and tqdm is None:
            self.logger.warning("tqdm is not available; disabling progress bars.")

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Initialize logger if provided
        if self.custom_logger and hasattr(self.custom_logger, "watch_model"):
            self.custom_logger.watch_model(self._model_to_watch)

    def _move_to_device(self, data: Any) -> Any:
        """Recursively move supported batch structures to the configured device."""
        if torch.is_tensor(data):
            return data.to(self.device)
        if hasattr(data, "to") and not isinstance(data, nn.Module):
            try:
                moved = data.to(self.device)
                return moved
            except TypeError:
                pass  # Fall back to manual traversal when .to signature differs
        if isinstance(data, Mapping):
            return {k: self._move_to_device(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return type(data)(self._move_to_device(v) for v in data)
        return data

    def _prepare_hf_batch(self, batch: Any) -> Dict[str, Any]:
        """Normalize HuggingFace style batches into a dict of tensors on the device."""
        prepared = self._move_to_device(batch)
        if isinstance(prepared, Mapping):
            return {k: v for k, v in prepared.items()}
        raise TypeError(
            "Expected a mapping-like structure for HuggingFace models but received "
            f"{type(prepared)}"
        )

    def _compute_loss(self, batch: Any) -> torch.Tensor:
        """Compute the training/validation loss for the provided batch."""
        if self.is_hf_model:
            model_inputs = self._prepare_hf_batch(batch)
            outputs = self.model(**model_inputs)
            if hasattr(outputs, "loss") and outputs.loss is not None:
                return outputs.loss
            if self.loss_fn is not None:
                logits = getattr(outputs, "logits", None)
                labels = model_inputs.get("labels")
                if logits is None or labels is None:
                    raise ValueError(
                        "HuggingFace model outputs do not expose `loss`. Provide `labels` in the "
                        "batch and/or a custom `loss_fn`."
                    )
                return self.loss_fn(logits, labels)
            raise ValueError(
                "Unable to compute loss: HuggingFace model output has no `loss` attribute and "
                "no custom `loss_fn` was provided."
            )

        # Default JEPA-style tuple batches
        if not isinstance(batch, (list, tuple)):
            raise TypeError(
                "Non-HuggingFace models expect DataLoader batches to be tuple/list structures. "
                f"Received type {type(batch)}."
            )
        if len(batch) == 3:
            state_t, action_t, state_t1 = batch
            state_t = self._move_to_device(state_t)
            action_t = self._move_to_device(action_t)
            state_t1 = self._move_to_device(state_t1)
            prediction, target = self.model(state_t, action_t, state_t1)
        elif len(batch) == 2:
            state_t, state_t1 = batch
            state_t = self._move_to_device(state_t)
            state_t1 = self._move_to_device(state_t1)
            prediction, target = self.model(state_t, state_t1)
        else:
            raise ValueError(
                "Expected batches of length 2 or 3 for JEPA models. "
                f"Received length {len(batch)}."
            )

        if self.loss_fn is None:
            raise ValueError(
                "A `loss_fn` must be provided when using non-HuggingFace models."
            )
        return self.loss_fn(prediction, target)

    @property
    def _model_to_watch(self) -> nn.Module:
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model

    def _initialize_logger(
        self,
        logger: Optional[Union[BaseLogger, Dict[str, Any], str]],
        project: Optional[str],
        run_name: Optional[str],
    ) -> Optional[BaseLogger]:
        if logger is None:
            return None

        if isinstance(logger, BaseLogger):
            return logger

        if isinstance(logger, str):
            backend = logger.lower()
            if backend == "wandb":
                wandb_config: Dict[str, Any] = {"enabled": True}
                if project:
                    wandb_config["project"] = project
                if run_name:
                    wandb_config["name"] = run_name
                return WandbLogger(wandb_config)
            raise ValueError(f"Unsupported logger backend '{logger}'.")

        if isinstance(logger, dict):
            config = deepcopy(logger)

            # Determine whether this is a composite logger config (console/wandb/tensorboard keys)
            composite_keys = {"console", "wandb", "tensorboard"}
            if composite_keys.intersection(config.keys()):
                wandb_conf = config.setdefault("wandb", {})
                if project and "project" not in wandb_conf:
                    wandb_conf["project"] = project
                if run_name and "name" not in wandb_conf:
                    wandb_conf["name"] = run_name
                return create_logger(config)

            # Treat as single-backend config (assume wandb for now)
            config.setdefault("enabled", True)
            if project and "project" not in config:
                config["project"] = project
            if run_name and "name" not in config:
                config["name"] = run_name
            return WandbLogger(config)

        raise TypeError(
            "logger must be a BaseLogger instance, a configuration dict, or a supported backend string."
        )

    def _resolve_device(self, device: str) -> torch.device:
        if self.distributed:
            if device != "auto" and device is not None:
                resolved = torch.device(device)
            else:
                if torch.cuda.is_available():
                    if self.local_rank is None:
                        raise ValueError(
                            "`local_rank` must be provided when `distributed=True` and using CUDA."
                        )
                    resolved = torch.device(f"cuda:{self.local_rank}")
                else:
                    resolved = torch.device("cpu")
        else:
            if device == "auto":
                resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                resolved = torch.device(device)
        return resolved

    def _init_distributed_if_needed(self):
        if not self.distributed:
            self.model.to(self.device)
            self.world_size = 1
            self.global_rank = 0
            return

        if not dist.is_available():
            raise RuntimeError("torch.distributed is not available but distributed training was requested")

        backend = self.dist_backend
        if backend == "nccl" and self.device.type != "cuda":
            backend = "gloo"

        if not dist.is_initialized():
            if self.global_rank is None or self.world_size is None:
                raise ValueError("`global_rank` and `world_size` must be provided when initializing distributed training")
            dist.init_process_group(backend=backend, rank=self.global_rank, world_size=self.world_size)
        self.dist_backend = backend

        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)

        self.world_size = dist.get_world_size()
        self.global_rank = dist.get_rank()

        if self.sync_batchnorm:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)  # type: ignore[arg-type]

        if hasattr(self.model, "to"):
            self.model.to(self.device)

        device_ids = [self.device.index] if self.device.type == "cuda" else None
        self.model = DDP(self.model, device_ids=device_ids, broadcast_buffers=False, find_unused_parameters=False)

    @property
    def is_main_process(self) -> bool:
        return (not self.distributed) or (self.global_rank == 0)

    def _distributed_reduce(self, value: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        if not self.distributed:
            return value
        value = value.clone()
        dist.all_reduce(value, op=op)
        if op == dist.ReduceOp.SUM:
            value /= self.world_size
        return value

    def _set_sampler_epoch(self, dataloader: DataLoader, epoch: int):
        sampler = getattr(dataloader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader that yields either JEPA tuples (state_t, state_t1)
                or HuggingFace-style dictionaries containing model inputs (e.g. input_ids,
                attention_mask, labels)
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
        progress_bar = None
        iterator = dataloader
        if self.use_progress_bar and self.is_main_process:
            desc = f"Train Epoch {self.current_epoch + 1}"
            progress_bar = tqdm(iterator, total=total_batches, desc=desc, leave=False)
            iterator = progress_bar

        for batch_idx, batch in enumerate(iterator):
            self.optimizer.zero_grad()

            loss = self._compute_loss(batch)

            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            
            self.optimizer.step()
            
            reduced_loss = self._distributed_reduce(loss.detach())
            loss_value = reduced_loss.item()

            total_loss += loss_value
            num_batches += 1
            self.global_step += 1

            if progress_bar is not None:
                postfix = {"loss": f"{loss_value:.4f}"}
                if self.optimizer.param_groups:
                    postfix["lr"] = f"{self.optimizer.param_groups[0]['lr']:.6f}"
                progress_bar.set_postfix(postfix, refresh=False)

            # Log progress
            if self.is_main_process and batch_idx % self.log_interval == 0:
                log_msg = (
                    f"Epoch {self.current_epoch}, Batch {batch_idx + 1}/{total_batches if total_batches is not None else '?'}, "
                    f"Loss: {loss_value:.6f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
                self.logger.info(log_msg)

                # Log to centralized logger
                if self.custom_logger:
                    metrics = {
                        'batch_loss': loss_value,
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'epoch': self.current_epoch,
                    }
                    self.custom_logger.log_metrics(metrics, step=self.global_step, prefix='train')

        if progress_bar is not None:
            progress_bar.close()

        avg_loss = total_loss / num_batches if num_batches else float('nan')
        return {
            "train_loss": avg_loss,
            "loss": avg_loss,
            "num_batches": num_batches,
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: Validation DataLoader producing JEPA tuples or HuggingFace-style
                dictionaries
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
        progress_bar = None
        iterator = dataloader
        if self.use_progress_bar and self.is_main_process:
            desc = f"Eval Epoch {self.current_epoch + 1}"
            progress_bar = tqdm(iterator, total=total_batches, desc=desc, leave=False)
            iterator = progress_bar

        with torch.no_grad():
            for batch in iterator:
                loss = self._compute_loss(batch)
                reduced_loss = self._distributed_reduce(loss.detach())
                loss_value = reduced_loss.item()
                total_loss += loss_value
                num_batches += 1

                if progress_bar is not None:
                    progress_bar.set_postfix({"loss": f"{loss_value:.4f}"}, refresh=False)

        if progress_bar is not None:
            progress_bar.close()

        avg_loss = total_loss / num_batches if num_batches else float('nan')
        return {
            "val_loss": avg_loss,
            "val_num_batches": num_batches,
        }
    
    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        val_dataloader: Optional[DataLoader] = None,
        save_every: int = 10,
        early_stopping_patience: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            train_dataloader: Training data loader
            num_epochs: Number of epochs to train
            val_dataloader: Optional validation data loader
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Stop if validation doesn't improve for N epochs
            
        Returns:
            Training history dictionary
        """
        history = {"train_loss": [], "val_loss": []}
        epochs_without_improvement = 0
        
        if self.is_main_process:
            self.logger.info(f"Starting training for {num_epochs} epochs on {self.device}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start_time = time.time()

            if self.distributed:
                self._set_sampler_epoch(train_dataloader, epoch)
                if val_dataloader is not None:
                    self._set_sampler_epoch(val_dataloader, epoch)
            
            # Train
            train_metrics = self.train_epoch(train_dataloader)
            history["train_loss"].append(train_metrics["train_loss"])
            
            # Validate
            val_metrics = {}
            if val_dataloader is not None:
                val_metrics = self.validate(val_dataloader)
                history["val_loss"].append(val_metrics["val_loss"])
                
                # Check for improvement
                val_loss = val_metrics["val_loss"]
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    epochs_without_improvement = 0
                    if self.is_main_process:
                        self.save_checkpoint(f"best_model.pt")
                else:
                    epochs_without_improvement += 1
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_dataloader is not None:
                        self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()
            
            # Log epoch results
            epoch_time = time.time() - start_time
            if self.is_main_process:
                log_msg = f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s - "
                log_msg += f"Train Loss: {train_metrics['train_loss']:.6f}"
                if val_dataloader is not None:
                    log_msg += f", Val Loss: {val_metrics['val_loss']:.6f}"
                self.logger.info(log_msg)

            # Log to centralized logger
            if self.custom_logger and self.is_main_process:
                epoch_metrics = {
                    'epoch_loss': train_metrics['train_loss'],
                    'epoch': epoch + 1,
                    'epoch_time': epoch_time,
                }
                if val_dataloader is not None:
                    epoch_metrics.update({
                        'val_loss': val_metrics['val_loss'],
                        'best_loss': self.best_loss,
                        'epochs_without_improvement': epochs_without_improvement,
                    })
                if self.scheduler is not None:
                    epoch_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
                
                self.custom_logger.log_metrics(epoch_metrics, step=epoch + 1, prefix='train')
                if val_dataloader is not None:
                    val_only_metrics = {
                        'epoch_loss': val_metrics['val_loss'],
                        'best_loss': self.best_loss,
                    }
                    self.custom_logger.log_metrics(val_only_metrics, step=epoch + 1, prefix='val')
            
            # Save checkpoint
            if self.is_main_process and (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
            
            # Early stopping
            if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                if self.is_main_process:
                    self.logger.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
                break
        
        if self.is_main_process:
            self.logger.info("Training completed!")
        
        # Finish logging session
        if self.custom_logger and self.is_main_process:
            self.custom_logger.finish()
            
        return history

    def save_checkpoint(self, filename: Optional[str] = None) -> str:
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch + 1}.pt"

        model_state = self._model_to_watch.state_dict()

        checkpoint = {
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        checkpoint_path = os.path.join(self.save_dir, filename)
        if self.is_main_process:
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Checkpoint saved: {filename}")

        # Log checkpoint to centralized logger if it's the best model
        if self.custom_logger and filename == "best_model.pt" and self.is_main_process:
            try:
                self.custom_logger.log_artifact(
                    checkpoint_path, 
                    name=f"best_model_epoch_{self.current_epoch + 1}",
                    artifact_type="model"
                )
            except Exception as e:
                self.logger.warning(f"Failed to log model artifact: {e}")
        return checkpoint_path
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self._model_to_watch.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.logger.info(f"Checkpoint loaded: {filename}")


def create_trainer(
    model: nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "auto",
    logger: Optional[BaseLogger] = None,
    loss_fn: Optional[LossFunction] = None,
    distributed: bool = False,
    local_rank: Optional[int] = None,
    global_rank: Optional[int] = None,
    world_size: Optional[int] = None,
    dist_backend: str = "nccl",
    sync_batchnorm: bool = False,
    logger_project: Optional[str] = None,
    logger_run_name: Optional[str] = None,
    **trainer_kwargs
) -> JEPATrainer:
    """
    Convenience function to create a trainer with sensible defaults.
    
    Args:
        model: JEPA model instance
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        device: Device to train on
        logger: Centralized logger instance
        loss_fn: Optional loss function callable
        **trainer_kwargs: Additional arguments for JEPATrainer
        
    Returns:
        Configured JEPATrainer instance
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=1000,  # Will be adjusted based on actual training
        eta_min=learning_rate * 0.01
    )
    
    return JEPATrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        logger=logger,
        loss_fn=loss_fn,
        distributed=distributed,
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
        dist_backend=dist_backend,
        sync_batchnorm=sync_batchnorm,
        logger_project=logger_project,
        logger_run_name=logger_run_name,
        **trainer_kwargs
    )
