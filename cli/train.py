"""
JEPA Training CLI

Command-line interface for training JEPA models with flexible configuration.
"""

import argparse
import os
import sys
import torch
import yaml
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import load_config, create_default_config, save_config, override_config_with_args
from models.jepa import JEPA
from models.encoder import create_encoder
from models.predictor import create_predictor
from trainer.trainer import JEPATrainer, create_trainer
from data.dataset import create_dataloader
from logging.multi_logger import MultiLogger
from logging.wandb_logger import WandbLogger
from logging.tensorboard_logger import TensorBoardLogger
from logging.console_logger import ConsoleLogger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train JEPA model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration YAML file'
    )
    
    # Data arguments
    parser.add_argument(
        '--train-data',
        type=str,
        help='Path to training data'
    )
    parser.add_argument(
        '--val-data',
        type=str,
        help='Path to validation data'
    )
    
    # Training arguments
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        help='Learning rate'
    )
    parser.add_argument(
        '--num-epochs', '-e',
        type=int,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        help='Device to train on'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for results'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        help='Name of the experiment'
    )
    
    # Resume training
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume from'
    )
    
    # Generate default config
    parser.add_argument(
        '--generate-config',
        type=str,
        help='Generate a default configuration file at specified path'
    )
    
    # Wandb arguments
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        help='Wandb project name'
    )
    parser.add_argument(
        '--wandb-entity',
        type=str,
        help='Wandb entity (username or team)'
    )
    parser.add_argument(
        '--wandb-name',
        type=str,
        help='Wandb run name'
    )
    parser.add_argument(
        '--wandb-tags',
        type=str,
        nargs='+',
        help='Wandb tags'
    )
    
    return parser.parse_args()


def create_model(config):
    """Create JEPA model from configuration."""
    # Create encoder
    encoder = create_encoder(
        encoder_type=config.model.encoder_type,
        input_dim=config.data.input_dim,
        hidden_dim=config.model.encoder_dim,
        dropout=config.model.dropout
    )
    
    # Create predictor
    predictor = create_predictor(
        predictor_type=config.model.predictor_type,
        input_dim=config.model.encoder_dim,
        hidden_dim=config.model.predictor_hidden_dim,
        output_dim=config.model.predictor_output_dim,
        dropout=config.model.dropout
    )
    
    # Create JEPA model
    model = JEPA(encoder=encoder, predictor=predictor)
    
    return model


def setup_directories(config):
    """Create necessary directories."""
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Create experiment subdirectories
    exp_output_dir = os.path.join(config.output_dir, config.experiment_name)
    exp_checkpoint_dir = os.path.join(config.checkpoint_dir, config.experiment_name)
    
    os.makedirs(exp_output_dir, exist_ok=True)
    os.makedirs(exp_checkpoint_dir, exist_ok=True)
    
    return exp_output_dir, exp_checkpoint_dir


def main():
    """Main training function."""
    args = parse_args()
    
    # Generate default config if requested
    if args.generate_config:
        config = create_default_config()
        save_config(config, args.generate_config)
        print(f"Default configuration saved to: {args.generate_config}")
        return
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    else:
        config = create_default_config()
        print("Using default configuration")
    
    # Override with command line arguments
    config = override_config_with_args(config, args)
    
    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Setup directories
    exp_output_dir, exp_checkpoint_dir = setup_directories(config)
    
    # Save final configuration
    config_save_path = os.path.join(exp_output_dir, "config.yaml")
    save_config(config, config_save_path)
    print(f"Configuration saved to: {config_save_path}")
    
    # Validate data paths
    if not config.data.train_data_path:
        raise ValueError("Training data path must be provided via config file or --train-data argument")
    
    if not os.path.exists(config.data.train_data_path):
        raise FileNotFoundError(f"Training data not found: {config.data.train_data_path}")
    
    print(f"Training data: {config.data.train_data_path}")
    if config.data.val_data_path:
        print(f"Validation data: {config.data.val_data_path}")
    
    # Create model
    print("Creating JEPA model...")
    model = create_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = create_dataloader(
        data_path=config.data.train_data_path,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        shuffle=True
    )
    
    val_loader = None
    if config.data.val_data_path and os.path.exists(config.data.val_data_path):
        val_loader = create_dataloader(
            data_path=config.data.val_data_path,
            batch_size=config.training.batch_size,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            shuffle=False
        )
    
    # Create trainer
    print("Creating trainer...")
    
    # Create centralized logger
    loggers = []
    
    # Add console logger
    if config.logging.console.enabled:
        console_logger = ConsoleLogger(
            log_file=str(Path(exp_output_dir) / "training.log") if config.logging.console.file else None,
            level=config.logging.console.level
        )
        loggers.append(console_logger)
    
    # Add wandb logger
    if config.logging.wandb.enabled:
        wandb_logger = WandbLogger(
            project=config.logging.wandb.project,
            entity=config.logging.wandb.entity,
            name=config.logging.wandb.name or config.experiment_name,
            tags=config.logging.wandb.tags,
            notes=config.logging.wandb.notes,
            config={
                'model': config.model.__dict__,
                'training': config.training.__dict__,
                'data': {k: v for k, v in config.data.__dict__.items() if not k.endswith('_path')},
                'experiment_name': config.experiment_name,
            }
        )
        loggers.append(wandb_logger)
    
    # Add tensorboard logger
    if config.logging.tensorboard.enabled:
        tensorboard_logger = TensorBoardLogger(
            log_dir=str(Path(exp_output_dir) / "tensorboard"),
            comment=config.experiment_name
        )
        loggers.append(tensorboard_logger)
    
    # Create multi-logger
    logger = MultiLogger(loggers) if loggers else None
    
    trainer = create_trainer(
        model=model,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        device=config.device,
        logger=logger,
        gradient_clip_norm=config.training.gradient_clip_norm,
        log_interval=config.training.log_interval,
        save_dir=exp_checkpoint_dir
    )
    
    # Resume from checkpoint if requested
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print(f"Starting training for {config.training.num_epochs} epochs...")
    print(f"Device: {trainer.device}")
    print(f"Experiment: {config.experiment_name}")
    print("-" * 50)
    
    try:
        history = trainer.train(
            train_dataloader=train_loader,
            num_epochs=config.training.num_epochs,
            val_dataloader=val_loader,
            save_every=config.training.save_every,
            early_stopping_patience=config.training.early_stopping_patience
        )
        
        # Save training history
        history_path = os.path.join(exp_output_dir, "training_history.yaml")
        with open(history_path, 'w') as f:
            yaml.dump(history, f, default_flow_style=False)
        
        print(f"Training completed successfully!")
        print(f"Results saved to: {exp_output_dir}")
        print(f"Checkpoints saved to: {exp_checkpoint_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving current checkpoint...")
        trainer.save_checkpoint("interrupted_checkpoint.pt")
        print("Checkpoint saved. You can resume training with --resume")
    
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
