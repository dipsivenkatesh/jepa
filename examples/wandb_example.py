#!/usr/bin/env python3
"""
Example of using JEPA with Weights & Biases integration through centralized logging.

This script demonstrates how to set up and use wandb logging with JEPA training
using the new centralized logging architecture.
"""

import os
import sys
import yaml

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_wandb_config_example():
    """Create an example configuration with centralized logging including wandb."""
    
    config = {
        # Model configuration
        'model': {
            'encoder_type': 'transformer',
            'encoder_dim': 256,
            'predictor_type': 'mlp',
            'predictor_hidden_dim': 512,
            'predictor_output_dim': 256,
            'dropout': 0.1
        },
        
        # Training configuration
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'num_epochs': 50,
            'warmup_epochs': 5,
            'gradient_clip_norm': 1.0,
            'save_every': 5,
            'early_stopping_patience': 10,
            'log_interval': 50
        },
        
        # Data configuration
        'data': {
            'train_data_path': 'data/train.npy',
            'val_data_path': 'data/val.npy',
            'test_data_path': 'data/test.npy',
            'num_workers': 4,
            'pin_memory': True,
            'sequence_length': 10,
            'input_dim': 784
        },
        
        # Centralized Logging Configuration
        'logging': {
            'wandb': {
                'enabled': True,
                'project': 'jepa-examples',
                'entity': None,  # Replace with your wandb username/team
                'name': 'jepa-demo-run',
                'tags': ['demo', 'example', 'jepa'],
                'notes': 'Demo run showing JEPA with centralized wandb integration',
                'log_model': True,
                'log_gradients': False,
                'log_freq': 50,
                'watch_model': True
            },
            'tensorboard': {
                'enabled': True,
                'log_dir': './logs/tensorboard',
                'comment': 'jepa-wandb-demo'
            },
            'console': {
                'enabled': True,
                'level': 'INFO',
                'file': True
            }
        },
        
        # General configuration
        'device': 'auto',
        'seed': 42,
        'output_dir': './outputs',
        'checkpoint_dir': './checkpoints',
        'experiment_name': 'jepa_wandb_demo'
    }
    
    return config


def main():
    """Demonstrate wandb integration with JEPA using centralized logging."""
    
    print("JEPA + Weights & Biases Integration Example (Centralized Logging)")
    print("=" * 70)
    
    # Create example config with centralized logging including wandb
    config = create_wandb_config_example()
    
    # Save the config
    config_path = 'wandb_example_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Created example config: {config_path}")
    print("\n📋 Centralized Logging Configuration:")
    print(f"   Wandb Project: {config['logging']['wandb']['project']}")
    print(f"   Wandb Run Name: {config['logging']['wandb']['name']}")
    print(f"   Wandb Tags: {config['logging']['wandb']['tags']}")
    print(f"   Wandb Model Logging: {config['logging']['wandb']['log_model']}")
    print(f"   TensorBoard Enabled: {config['logging']['tensorboard']['enabled']}")
    print(f"   Console Logging: {config['logging']['console']['enabled']}")
    
    print("\n🚀 To run training with centralized wandb:")
    print(f"   python -m jepa.cli train --config {config_path}")
    
    print("\n💡 Alternative CLI approach:")
    print("   python -m jepa.cli train \\")
    print("       --config config/default_config.yaml \\")
    print("       --wandb \\")
    print("       --wandb-project jepa-experiments \\")
    print("       --wandb-name my-experiment \\")
    print("       --wandb-tags transformer vision \\")
    print("       --train-data data/train.npy")
    
    print("\n📊 What you'll see with centralized logging:")
    print("   • Wandb: Real-time loss curves, system metrics, model checkpoints")
    print("   • TensorBoard: Local visualization, scalars, hyperparameters")
    print("   • Console: Real-time terminal output + log files")
    print("   • Unified metrics across all backends")
    
    print("\n🔗 Benefits of centralized logging:")
    print("   • Single point of control for all logging")
    print("   • Easy to enable/disable specific backends")
    print("   • Consistent metrics across platforms")
    print("   • Extensible to new logging backends")
    print("   • Clean separation of concerns")
    
    print("\n⚙️ Setup steps:")
    print("   1. pip install wandb tensorboard")
    print("   2. wandb login")
    print("   3. Update the 'entity' field in config")
    print("   4. Run training!")
    
    print(f"\n📁 Config file created: {config_path}")
    print("   Edit this file to customize your logging setup.")
    
    print("\n🔧 Programmatic example:")
    print("   See training_example.py for how to create loggers programmatically")
    
    # Create a small programmatic example
    print("\n💻 Quick programmatic setup:")
    print("""
from loggers.multi_logger import MultiLogger
from loggers.wandb_logger import WandbLogger
from loggers.tensorboard_logger import TensorBoardLogger
from loggers.console_logger import ConsoleLogger

# Create individual loggers
loggers = []
loggers.append(WandbLogger(project="my-project", name="my-run"))
loggers.append(TensorBoardLogger(log_dir="./logs"))
loggers.append(ConsoleLogger(log_file="training.log"))

# Create unified logger
multi_logger = MultiLogger(loggers)

# Use with trainer
trainer = create_trainer(model, logger=multi_logger)
""")


if __name__ == "__main__":
    main()
