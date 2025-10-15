#!/usr/bin/env python3
"""
Comprehensive example demonstrating the centralized logging system.

This example shows how to use all available logging backends together
and demonstrates the flexibility of the centralized logging architecture.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.jepa import JEPA
from models.encoder import create_encoder
from models.predictor import create_predictor
from trainer.trainer import create_trainer
from loggers.multi_logger import MultiLogger
from loggers.wandb_logger import WandbLogger
from loggers.tensorboard_logger import TensorBoardLogger
from loggers.console_logger import ConsoleLogger


def create_dummy_data(num_samples=100, seq_length=10, hidden_dim=64):
    """Create dummy data for demonstration."""
    x = torch.randn(num_samples, seq_length, hidden_dim)
    y = x + torch.randn_like(x) * 0.1  # Add small noise
    return TensorDataset(x, y)


def example_console_only():
    """Example using only console logging."""
    print("\n🖥️  Console-only logging example")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create simple model
        model = JEPA(
            create_encoder("mlp", 64),
            create_predictor("mlp", 64)
        )
        
        # Console logger with file output
        logger = ConsoleLogger(
            log_file=os.path.join(tmp_dir, "console_only.log"),
            level="INFO"
        )
        
        # Create trainer
        trainer = create_trainer(model, logger=logger)
        
        # Log some metrics
        logger.log_metrics({"loss": 0.5, "accuracy": 0.8}, step=1)
        logger.log_hyperparameters({"lr": 0.001, "batch_size": 32})
        
        print("✅ Console logging completed")
        print(f"   Log file: {os.path.join(tmp_dir, 'console_only.log')}")


def example_tensorboard_only():
    """Example using only TensorBoard logging."""
    print("\n📊 TensorBoard-only logging example")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create simple model
        model = JEPA(
            create_encoder("mlp", 64),
            create_predictor("mlp", 64)
        )
        
        # TensorBoard logger
        logger = TensorBoardLogger(
            log_dir=os.path.join(tmp_dir, "tensorboard"),
            comment="demo"
        )
        
        # Create trainer
        trainer = create_trainer(model, logger=logger)
        
        # Log some metrics
        logger.log_metrics({"loss": 0.3, "accuracy": 0.9}, step=1)
        logger.log_hyperparameters({"lr": 0.001, "batch_size": 32})
        
        logger.close()
        print("✅ TensorBoard logging completed")
        print(f"   TensorBoard logs: {os.path.join(tmp_dir, 'tensorboard')}")
        print("   Run: tensorboard --logdir <path> to view")


def example_multi_logger():
    """Example using multiple loggers together."""
    print("\n🔗 Multi-logger example (Console + TensorBoard)")
    print("-" * 50)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create simple model
        model = JEPA(
            create_encoder("mlp", 64),
            create_predictor("mlp", 64)
        )
        
        # Create multiple loggers
        loggers = []
        
        # Console logger
        console_logger = ConsoleLogger(
            log_file=os.path.join(tmp_dir, "multi_example.log"),
            level="INFO"
        )
        loggers.append(console_logger)
        
        # TensorBoard logger
        tb_logger = TensorBoardLogger(
            log_dir=os.path.join(tmp_dir, "tensorboard"),
            comment="multi_demo"
        )
        loggers.append(tb_logger)
        
        # Create multi-logger
        multi_logger = MultiLogger(loggers)
        
        # Create trainer
        trainer = create_trainer(model, logger=multi_logger)
        
        # Log metrics to all backends simultaneously
        multi_logger.log_metrics({
            "train_loss": 0.4,
            "val_loss": 0.5,
            "accuracy": 0.85
        }, step=1)
        
        multi_logger.log_hyperparameters({
            "model": "JEPA",
            "encoder": "MLP",
            "predictor": "MLP",
            "hidden_dim": 64,
            "lr": 0.001
        })
        
        # Simulate training progress
        for step in range(5):
            multi_logger.log_metrics({
                "train_loss": 0.4 - step * 0.05,
                "val_loss": 0.5 - step * 0.04,
                "learning_rate": 0.001 * (0.9 ** step)
            }, step=step + 1)
        
        multi_logger.close()
        print("✅ Multi-logger example completed")
        print("   Metrics logged to both console and TensorBoard")


def example_wandb_simulation():
    """Example showing wandb configuration (simulation)."""
    print("\n🔥 Wandb logging example (configuration only)")
    print("-" * 50)
    
    # Note: This doesn't actually log to wandb to avoid requiring login
    # but shows how it would be configured
    
    try:
        # Create model
        model = JEPA(
            create_encoder("mlp", 64),
            create_predictor("mlp", 64)
        )
        
        # Create wandb logger (this will check availability)
        wandb_logger = WandbLogger(
            project="jepa-demo",
            name="test-run",
            tags=["demo", "test"],
            notes="Demo run for logging system"
        )
        
        if wandb_logger.is_available():
            print("✅ Wandb is available and configured")
            print("   This would log to wandb.ai")
            wandb_logger.close()
        else:
            print("ℹ️  Wandb not available (not logged in or installed)")
            print("   Install wandb and run 'wandb login' to enable")
            
    except Exception as e:
        print(f"ℹ️  Wandb configuration example: {e}")
        print("   This is expected if wandb is not installed")


def example_mini_training():
    """Example with actual mini training loop."""
    print("\n🚀 Mini training example with centralized logging")
    print("-" * 55)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create model and data
        model = JEPA(
            create_encoder("mlp", 64),
            create_predictor("mlp", 64)
        )
        
        dataset = create_dummy_data(num_samples=32, seq_length=8, hidden_dim=64)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Create multi-logger
        loggers = []
        loggers.append(ConsoleLogger(level="INFO"))
        loggers.append(TensorBoardLogger(
            log_dir=os.path.join(tmp_dir, "mini_training"),
            comment="mini_demo"
        ))
        
        multi_logger = MultiLogger(loggers)
        
        # Create trainer
        trainer = create_trainer(
            model=model,
            learning_rate=0.01,
            logger=multi_logger
        )
        
        # Log hyperparameters
        multi_logger.log_hyperparameters({
            "model_type": "JEPA",
            "hidden_dim": 64,
            "learning_rate": 0.01,
            "batch_size": 8,
            "epochs": 3
        })
        
        # Simple training loop
        model.train()
        for epoch in range(3):
            epoch_loss = 0
            for batch_idx, (state_t, state_t1) in enumerate(dataloader):
                loss = trainer._compute_loss(state_t, state_t1)
                epoch_loss += loss.item()
                
                # Log batch metrics
                step = epoch * len(dataloader) + batch_idx
                multi_logger.log_metrics({
                    "batch_loss": loss.item(),
                    "epoch": epoch
                }, step=step)
            
            # Log epoch metrics
            avg_loss = epoch_loss / len(dataloader)
            multi_logger.log_metrics({
                "epoch_loss": avg_loss,
                "epoch": epoch
            }, step=epoch)
            
            print(f"   Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
        
        multi_logger.close()
        print("✅ Mini training completed")


def main():
    """Run all logging examples."""
    print("🧪 JEPA Centralized Logging System Examples")
    print("=" * 60)
    
    print("\nThis demo shows different ways to use the centralized logging system:")
    
    # Run examples
    example_console_only()
    example_tensorboard_only()
    example_multi_logger()
    example_wandb_simulation()
    example_mini_training()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed successfully!")
    
    print("\n💡 Key takeaways:")
    print("   • Use individual loggers for simple cases")
    print("   • Use MultiLogger to combine multiple backends")
    print("   • All loggers implement the same BaseLogger interface")
    print("   • Easy to add new logging backends")
    print("   • Centralized configuration through config files")
    print("   • Automatic availability checking")
    
    print("\n📚 Next steps:")
    print("   • Try the CLI: python -m jepa.cli train --config config/default_config.yaml")
    print("   • Modify config files to enable/disable specific loggers")
    print("   • Add your own custom logger by implementing BaseLogger")
    print("   • Check training_example.py for full training demo")


if __name__ == "__main__":
    main()
