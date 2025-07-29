"""
Utility functions for JEPA training and evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import json
import os


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params
    }


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training metrics over time
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, len(history), figsize=(5 * len(history), 4))
    if len(history) == 1:
        axes = [axes]
    
    for idx, (metric_name, values) in enumerate(history.items()):
        axes[idx].plot(values)
        axes[idx].set_title(f'{metric_name.replace("_", " ").title()}')
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel(metric_name)
        axes[idx].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_training_config(config: Dict[str, Any], save_path: str):
    """
    Save training configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the configuration
    """
    # Convert non-serializable objects to strings
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, (str, int, float, bool, list, dict)):
            serializable_config[key] = value
        else:
            serializable_config[key] = str(value)
    
    with open(save_path, 'w') as f:
        json.dump(serializable_config, f, indent=2)


def load_training_config(config_path: str) -> Dict[str, Any]:
    """
    Load training configuration from JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def create_data_splits(
    dataset: torch.utils.data.Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Dict[str, torch.utils.data.Dataset]:
    """
    Split a dataset into train/validation/test sets.
    
    Args:
        dataset: PyTorch dataset to split
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with split datasets
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }


def setup_reproducibility(seed: int = 42):
    """
    Set up reproducible training environment.
    
    Args:
        seed: Random seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory
    
    return info


def log_model_summary(model: nn.Module, input_shape: tuple, device: str = "cpu"):
    """
    Log a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (without batch dimension)
        device: Device to run the model on
    """
    model.eval()
    
    # Count parameters
    param_info = count_parameters(model)
    
    print("=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)
    print(f"Total parameters: {param_info['total_parameters']:,}")
    print(f"Trainable parameters: {param_info['trainable_parameters']:,}")
    print(f"Non-trainable parameters: {param_info['non_trainable_parameters']:,}")
    print()
    
    # Try to get output shape
    try:
        dummy_input = torch.randn(1, *input_shape).to(device)
        model.to(device)
        
        with torch.no_grad():
            if hasattr(model, 'encoder') and hasattr(model, 'predictor'):
                # For JEPA models
                z = model.encoder(dummy_input)
                pred = model.predictor(z)
                print(f"Input shape: {dummy_input.shape}")
                print(f"Encoder output shape: {z.shape}")
                print(f"Predictor output shape: {pred.shape}")
            else:
                output = model(dummy_input)
                print(f"Input shape: {dummy_input.shape}")
                print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Could not determine output shapes: {e}")
    
    print("=" * 50)


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore the best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.wait = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.stopped_epoch = 0
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights for
            
        Returns:
            True if training should be stopped
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = self.wait
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False


def create_experiment_dir(base_dir: str = "./experiments", experiment_name: Optional[str] = None) -> str:
    """
    Create a directory for an experiment with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Optional name for the experiment
        
    Returns:
        Path to the created experiment directory
    """
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    else:
        exp_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    
    return exp_dir
