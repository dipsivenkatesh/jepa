"""
Complete training example showing how to use the JEPA training framework with centralized logging.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import JEPA components
from models.jepa import JEPA
from models.encoder import create_encoder
from models.predictor import create_predictor
from trainer.trainer import JEPATrainer, create_trainer
from loggers.multi_logger import MultiLogger
from loggers.console_logger import ConsoleLogger
from loggers.tensorboard_logger import TensorBoardLogger


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
    encoder = create_encoder("mlp", hidden_dim)
    predictor = create_predictor("mlp", hidden_dim)
    return JEPA(encoder, predictor)


def training_example():
    """Complete training example with centralized logging."""
    
    # Setup
    torch.manual_seed(42)
    experiment_dir = "./experiments/jepa_demo"
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Experiment directory: {experiment_dir}")
    
    # Model configuration
    hidden_dim = 256
    model = create_sample_model(hidden_dim)
    
    # Print model info
    print(f"Model: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = DummyDataset(num_samples=800, seq_length=10, hidden_dim=hidden_dim)
    val_dataset = DummyDataset(num_samples=200, seq_length=10, hidden_dim=hidden_dim)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create centralized logger
    print("Setting up logging...")
    loggers = []
    
    # Console logger
    console_logger = ConsoleLogger(
        log_file=f"{experiment_dir}/training.log",
        level="INFO"
    )
    loggers.append(console_logger)
    
    # TensorBoard logger
    tensorboard_logger = TensorBoardLogger(
        log_dir=f"{experiment_dir}/tensorboard",
        comment="jepa_demo"
    )
    loggers.append(tensorboard_logger)
    
    # Create multi-logger
    multi_logger = MultiLogger(loggers)
    
    # Log hyperparameters
    hyperparams = {
        "model_type": "JEPA",
        "hidden_dim": hidden_dim,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "num_epochs": 50
    }
    multi_logger.log_hyperparameters(hyperparams)
    
    # Create trainer with centralized logging
    print("Setting up trainer...")
    trainer = create_trainer(
        model=model,
        learning_rate=1e-3,
        weight_decay=1e-4,
        logger=multi_logger  # Pass the centralized logger
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
    
    # Simple evaluation
    print("Evaluating model...")
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for state_t, state_t1 in val_loader:
            loss = trainer._compute_loss(state_t, state_t1)
            total_loss += loss.item()
            num_batches += 1
    
    final_val_loss = total_loss / num_batches
    print(f"Final validation loss: {final_val_loss:.6f}")
    
    # Close loggers
    multi_logger.close()
    
    # Save training history
    import json
    config = {
        "model_type": "JEPA",
        "hidden_dim": hidden_dim,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "num_epochs": 50,
        "final_val_loss": final_val_loss,
    }
    
    with open(f"{experiment_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Training completed! Results saved to {experiment_dir}")
    return model, {"final_val_loss": final_val_loss}


def custom_trainer_example():
    """Example showing how to create a custom trainer configuration with centralized logging."""
    
    # Create model
    model = create_sample_model(hidden_dim=128)
    
    # Create simple console logger
    console_logger = ConsoleLogger(level="INFO")
    
    # Custom optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Custom trainer
    trainer = JEPATrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="auto",
        logger=console_logger
    )
    
    print("Custom trainer created with:")
    print(f"  Optimizer: {type(optimizer).__name__}")
    print(f"  Scheduler: {type(scheduler).__name__}")
    print(f"  Device: {trainer.device}")
    
    return trainer


def main():
    """Run training examples."""
    print("JEPA Training Framework Demo with Centralized Logging")
    print("=" * 55)
    
    # Run basic training example
    print("\n1. Running basic training example...")
    model, results = training_example()
    
    print(f"\nTraining completed with final validation loss: {results['final_val_loss']:.6f}")
    
    # Show custom trainer example
    print("\n2. Custom trainer configuration example...")
    custom_trainer = custom_trainer_example()
    
    print("\nDemo completed!")
    print("\n💡 Key features demonstrated:")
    print("   • Centralized logging with MultiLogger")
    print("   • Console and TensorBoard logging")
    print("   • Custom trainer configurations")
    print("   • Automatic experiment directory setup")
    print("   • Hyperparameter logging")


if __name__ == "__main__":
    main()
