"""
JEPA Trainer - A flexible training module for JEPA models.

This trainer is designed to work with any JEPA model configuration and provides
a clean, reusable interface for training joint-embedding predictive architectures.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from typing import Dict, Any, Optional, Callable
import os
import logging

from ..logging import create_logger, BaseLogger


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
        logger: Optional[BaseLogger] = None
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
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_clip_norm = gradient_clip_norm
        self.log_interval = log_interval
        self.save_dir = save_dir
        self.custom_logger = logger
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Initialize logger if provided
        if self.custom_logger:
            self.custom_logger.watch_model(self.model)
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader that yields (state_t, state_t1) pairs
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (state_t, state_t1) in enumerate(dataloader):
            # Move data to device
            state_t = state_t.to(self.device)
            state_t1 = state_t1.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            prediction, target = self.model(state_t, state_t1)
            loss = self.model.loss(prediction, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log progress
            if batch_idx % self.log_interval == 0:
                log_msg = (
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(dataloader)}, "
                    f"Loss: {loss.item():.6f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
                self.logger.info(log_msg)
                
                # Log to centralized logger
                if self.custom_logger:
                    metrics = {
                        'batch_loss': loss.item(),
                        'learning_rate': self.optimizer.param_groups[0]['lr'],
                        'epoch': self.current_epoch,
                    }
                    self.custom_logger.log_metrics(metrics, step=self.global_step, prefix='train')
        
        avg_loss = total_loss / num_batches
        return {"train_loss": avg_loss}
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: Validation DataLoader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for state_t, state_t1 in dataloader:
                state_t = state_t.to(self.device)
                state_t1 = state_t1.to(self.device)
                
                prediction, target = self.model(state_t, state_t1)
                loss = self.model.loss(prediction, target)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {"val_loss": avg_loss}
    
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
        
        self.logger.info(f"Starting training for {num_epochs} epochs on {self.device}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
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
            log_msg = f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s - "
            log_msg += f"Train Loss: {train_metrics['train_loss']:.6f}"
            if val_dataloader is not None:
                log_msg += f", Val Loss: {val_metrics['val_loss']:.6f}"
            self.logger.info(log_msg)
            
            # Log to centralized logger
            if self.custom_logger:
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
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
            
            # Early stopping
            if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
                break
        
        self.logger.info("Training completed!")
        
        # Finish logging session
        if self.custom_logger:
            self.custom_logger.finish()
            
        return history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        checkpoint_path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {filename}")
        
        # Log checkpoint to centralized logger if it's the best model
        if self.custom_logger and filename == "best_model.pt":
            try:
                self.custom_logger.log_artifact(
                    checkpoint_path, 
                    name=f"best_model_epoch_{self.current_epoch + 1}",
                    artifact_type="model"
                )
            except Exception as e:
                self.logger.warning(f"Failed to log model artifact: {e}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
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
        **trainer_kwargs
    )
