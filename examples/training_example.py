"""
Complete training example showing how to use the JEPA training framework.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Import JEPA components
from models import JEPA, Encoder, Predictor
from trainer import JEPATrainer, create_trainer, quick_evaluate
from trainer.utils import setup_reproducibility, log_model_summary, create_experiment_dir


class DummyDataset:
    """
    Create dummy sequential data for demonstration.
    In practice, you would replace this with your actual dataset.
    """
    
    def __init__(self, num_samples: int = 1000, seq_length: int = 10, hidden_dim: int = 256):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        
        # Generate random sequential data where state_t1 is slightly related to state_t
        self.data_t = torch.randn(num_samples, seq_length, hidden_dim)
        
        # Add some temporal correlation
        noise = torch.randn(num_samples, seq_length, hidden_dim) * 0.3
        self.data_t1 = self.data_t + noise
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data_t[idx], self.data_t1[idx]


def create_sample_model(hidden_dim: int = 256) -> JEPA:
    """Create a sample JEPA model for demonstration."""
    encoder = Encoder(hidden_dim)
    predictor = Predictor(hidden_dim)
    return JEPA(encoder, predictor)


def training_example():
    """Complete training example."""
    
    # Setup
    setup_reproducibility(42)
    experiment_dir = create_experiment_dir("./experiments", "jepa_demo")
    print(f"Experiment directory: {experiment_dir}")
    
    # Model configuration
    hidden_dim = 256
    model = create_sample_model(hidden_dim)
    
    # Log model info
    log_model_summary(model, input_shape=(10, hidden_dim))
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = DummyDataset(num_samples=800, seq_length=10, hidden_dim=hidden_dim)
    val_dataset = DummyDataset(num_samples=200, seq_length=10, hidden_dim=hidden_dim)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create trainer with custom settings
    print("Setting up trainer...")
    trainer = create_trainer(
        model=model,
        learning_rate=1e-3,
        weight_decay=1e-4,
        save_dir=f"{experiment_dir}/checkpoints",
        log_interval=10,
        gradient_clip_norm=1.0
    )
    
    # Train the model
    print("Starting training...")
    history = trainer.train(
        train_dataloader=train_loader,
        num_epochs=50,
        val_dataloader=val_loader,
        save_every=10,
        early_stopping_patience=15
    )
    
    # Evaluate the model
    print("Evaluating model...")
    final_metrics = quick_evaluate(model, val_loader)
    print("Final validation metrics:")
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    # Save training history and config
    from trainer.utils import save_training_config, plot_training_history
    
    config = {
        "model_type": "JEPA",
        "hidden_dim": hidden_dim,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "num_epochs": 50,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
    }
    
    save_training_config(config, f"{experiment_dir}/config.json")
    
    # Plot and save training curves
    try:
        plot_training_history(history, f"{experiment_dir}/plots/training_history.png")
    except ImportError:
        print("Matplotlib not available, skipping plot generation")
    
    print(f"Training completed! Results saved to {experiment_dir}")
    return model, history


def custom_trainer_example():
    """Example showing how to create a custom trainer configuration."""
    
    # Create model
    model = create_sample_model(hidden_dim=128)
    
    # Custom optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Custom trainer
    trainer = JEPATrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="auto",
        gradient_clip_norm=0.5,
        log_interval=50,
        save_dir="./custom_checkpoints"
    )
    
    print("Custom trainer created with:")
    print(f"  Optimizer: {type(optimizer).__name__}")
    print(f"  Scheduler: {type(scheduler).__name__}")
    print(f"  Device: {trainer.device}")
    
    return trainer


def main():
    """Run training examples."""
    print("JEPA Training Framework Demo")
    print("=" * 40)
    
    # Run basic training example
    print("\n1. Running basic training example...")
    model, history = training_example()
    
    print(f"\nTraining completed with final validation loss: {history['val_loss'][-1]:.6f}")
    
    # Show custom trainer example
    print("\n2. Custom trainer configuration example...")
    custom_trainer = custom_trainer_example()
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()
