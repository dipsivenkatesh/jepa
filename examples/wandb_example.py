#!/usr/bin/env python3
"""
Example of using JEPA with Weights & Biases integration.

This script demonstrates how to set up and use wandb logging with JEPA training.
"""

import os
import sys
import yaml

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_wandb_config_example():
    """Create an example configuration with wandb enabled."""
    
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
        
        # Wandb configuration - ENABLED
        'wandb': {
            'enabled': True,
            'project': 'jepa-examples',
            'entity': None,  # Replace with your wandb username/team
            'name': 'jepa-demo-run',
            'tags': ['demo', 'example', 'jepa'],
            'notes': 'Demo run showing JEPA with wandb integration',
            'log_model': True,
            'log_gradients': False,
            'log_freq': 50,
            'watch_model': True
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
    """Demonstrate wandb integration with JEPA."""
    
    print("JEPA + Weights & Biases Integration Example")
    print("=" * 50)
    
    # Create example config with wandb enabled
    config = create_wandb_config_example()
    
    # Save the config
    config_path = 'wandb_example_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Created example config: {config_path}")
    print("\n📋 Wandb Configuration:")
    print(f"   Project: {config['wandb']['project']}")
    print(f"   Run Name: {config['wandb']['name']}")
    print(f"   Tags: {config['wandb']['tags']}")
    print(f"   Model Logging: {config['wandb']['log_model']}")
    print(f"   Gradient Logging: {config['wandb']['log_gradients']}")
    
    print("\n🚀 To run training with wandb:")
    print(f"   python -m jepa.cli train --config {config_path}")
    
    print("\n💡 Alternative CLI approach:")
    print("   python -m jepa.cli train \\")
    print("       --config config/default_config.yaml \\")
    print("       --wandb \\")
    print("       --wandb-project jepa-experiments \\")
    print("       --wandb-name my-experiment \\")
    print("       --wandb-tags transformer vision \\")
    print("       --train-data data/train.npy")
    
    print("\n📊 What you'll see in wandb:")
    print("   • Real-time loss curves")
    print("   • Learning rate schedules")
    print("   • System metrics (GPU, memory)")
    print("   • Model architecture visualization")
    print("   • Hyperparameter tracking")
    print("   • Model checkpoints (if log_model=True)")
    print("   • Custom charts and comparisons")
    
    print("\n🔗 Useful wandb features:")
    print("   • wandb.ai dashboard for monitoring")
    print("   • Hyperparameter sweeps")
    print("   • Team collaboration")
    print("   • Experiment comparison")
    print("   • Model registry")
    
    print("\n⚙️ Setup steps:")
    print("   1. pip install wandb")
    print("   2. wandb login")
    print("   3. Update the 'entity' field in config")
    print("   4. Run training!")
    
    print(f"\n📁 Config file created: {config_path}")
    print("   Edit this file to customize your wandb setup.")


if __name__ == "__main__":
    main()
