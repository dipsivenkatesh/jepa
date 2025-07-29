"""
Data utilities for JEPA datasets.

Provides utility functions for data loading, preprocessing, validation, and analysis.
"""

import torch
from torch.utils.data import DataLoader as TorchDataLoader, Dataset, Sampler
from torch.utils.data.sampler import WeightedRandomSampler
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import json
from PIL import Image


class DataSplitter:
    """Utility class for splitting datasets into train/validation/test sets."""
    
    @staticmethod
    def random_split(
        dataset: Dataset,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: Optional[int] = None
    ) -> Dict[str, Dataset]:
        """
        Randomly split dataset into train/validation/test sets.
        
        Args:
            dataset: Dataset to split
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
        
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )
        
        return {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        }
    
    @staticmethod
    def temporal_split(
        dataset: Dataset,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Dict[str, Dataset]:
        """
        Split dataset temporally (useful for time series data).
        
        Args:
            dataset: Dataset to split
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            
        Returns:
            Dictionary with split datasets
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        dataset_size = len(dataset)
        train_end = int(train_ratio * dataset_size)
        val_end = train_end + int(val_ratio * dataset_size)
        
        train_indices = list(range(0, train_end))
        val_indices = list(range(train_end, val_end))
        test_indices = list(range(val_end, dataset_size))
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        
        return {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset
        }


class DataLoader:
    """Enhanced DataLoader factory with JEPA-specific configurations."""
    
    @staticmethod
    def create_jepa_dataloader(
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
        **kwargs
    ) -> TorchDataLoader:
        """
        Create DataLoader optimized for JEPA training.
        
        Args:
            dataset: Dataset to load
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            drop_last: Whether to drop last incomplete batch
            collate_fn: Custom collate function
            **kwargs: Additional DataLoader arguments
            
        Returns:
            PyTorch DataLoader
        """
        from .dataset import collate_jepa_batch
        
        if collate_fn is None:
            collate_fn = collate_jepa_batch
        
        return TorchDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=collate_fn,
            **kwargs
        )
    
    @staticmethod
    def create_evaluation_dataloader(
        dataset: Dataset,
        batch_size: int = 64,
        num_workers: int = 4,
        **kwargs
    ) -> TorchDataLoader:
        """Create DataLoader for evaluation (no shuffling, larger batch size)."""
        return DataLoader.create_jepa_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            **kwargs
        )


def check_data_integrity(dataset: Dataset, num_samples: int = 100) -> Dict[str, Any]:
    """
    Check data integrity by sampling from dataset.
    
    Args:
        dataset: Dataset to check
        num_samples: Number of samples to check
        
    Returns:
        Dictionary with integrity check results
    """
    results = {
        "total_samples": len(dataset),
        "checked_samples": min(num_samples, len(dataset)),
        "errors": [],
        "shapes": [],
        "dtypes": [],
        "value_ranges": []
    }
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        try:
            sample = dataset[idx]
            
            if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                state_t, state_t1 = sample[0], sample[1]
                
                # Check shapes
                if isinstance(state_t, torch.Tensor) and isinstance(state_t1, torch.Tensor):
                    results["shapes"].append((state_t.shape, state_t1.shape))
                    results["dtypes"].append((state_t.dtype, state_t1.dtype))
                    
                    # Check value ranges
                    results["value_ranges"].append({
                        "state_t": (state_t.min().item(), state_t.max().item()),
                        "state_t1": (state_t1.min().item(), state_t1.max().item())
                    })
                
        except Exception as e:
            results["errors"].append(f"Sample {idx}: {str(e)}")
    
    # Summary statistics
    if results["shapes"]:
        unique_shapes = list(set(results["shapes"]))
        results["unique_shapes"] = unique_shapes
        results["shape_consistency"] = len(unique_shapes) == 1
    
    if results["dtypes"]:
        unique_dtypes = list(set(results["dtypes"]))
        results["unique_dtypes"] = unique_dtypes
        results["dtype_consistency"] = len(unique_dtypes) == 1
    
    results["error_rate"] = len(results["errors"]) / results["checked_samples"]
    
    return results


def get_data_stats(dataset: Dataset, num_samples: int = 1000) -> Dict[str, Any]:
    """
    Compute statistics for dataset.
    
    Args:
        dataset: Dataset to analyze
        num_samples: Number of samples to use for statistics
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        "total_samples": len(dataset),
        "analyzed_samples": min(num_samples, len(dataset))
    }
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    state_t_values = []
    state_t1_values = []
    
    for idx in indices:
        try:
            sample = dataset[idx]
            if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                state_t, state_t1 = sample[0], sample[1]
                
                if isinstance(state_t, torch.Tensor) and isinstance(state_t1, torch.Tensor):
                    state_t_values.append(state_t.flatten())
                    state_t1_values.append(state_t1.flatten())
        except Exception:
            continue
    
    if state_t_values:
        all_state_t = torch.cat(state_t_values)
        all_state_t1 = torch.cat(state_t1_values)
        
        stats.update({
            "state_t_mean": all_state_t.mean().item(),
            "state_t_std": all_state_t.std().item(),
            "state_t_min": all_state_t.min().item(),
            "state_t_max": all_state_t.max().item(),
            "state_t1_mean": all_state_t1.mean().item(),
            "state_t1_std": all_state_t1.std().item(),
            "state_t1_min": all_state_t1.min().item(),
            "state_t1_max": all_state_t1.max().item(),
            "correlation": torch.corrcoef(torch.stack([all_state_t, all_state_t1]))[0, 1].item()
        })
    
    return stats


def visualize_batch(
    dataloader: TorchDataLoader,
    num_samples: int = 8,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Visualize a batch of data from dataloader.
    
    Args:
        dataloader: DataLoader to sample from
        num_samples: Number of samples to visualize
        save_path: Optional path to save the visualization
        figsize: Figure size for the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available for visualization")
        return
    
    # Get a batch
    batch = next(iter(dataloader))
    state_t, state_t1 = batch[0], batch[1]
    
    # Limit to requested number of samples
    num_samples = min(num_samples, state_t.size(0))
    
    fig, axes = plt.subplots(2, num_samples, figsize=figsize)
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_samples):
        # State t
        if state_t[i].dim() == 3 and state_t[i].size(0) in [1, 3]:  # Image data
            img_t = state_t[i].permute(1, 2, 0).cpu().numpy()
            if img_t.shape[2] == 1:
                img_t = img_t.squeeze(2)
            axes[0, i].imshow(img_t)
        else:  # Other data types
            if state_t[i].dim() > 1:
                data = state_t[i].cpu().numpy().flatten()[:100]  # Show first 100 elements
            else:
                data = state_t[i].cpu().numpy()
            axes[0, i].plot(data)
        
        axes[0, i].set_title(f'State t (sample {i})')
        axes[0, i].axis('off')
        
        # State t+1
        if state_t1[i].dim() == 3 and state_t1[i].size(0) in [1, 3]:  # Image data
            img_t1 = state_t1[i].permute(1, 2, 0).cpu().numpy()
            if img_t1.shape[2] == 1:
                img_t1 = img_t1.squeeze(2)
            axes[1, i].imshow(img_t1)
        else:  # Other data types
            if state_t1[i].dim() > 1:
                data = state_t1[i].cpu().numpy().flatten()[:100]  # Show first 100 elements
            else:
                data = state_t1[i].cpu().numpy()
            axes[1, i].plot(data)
        
        axes[1, i].set_title(f'State t+1 (sample {i})')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_balanced_sampler(
    dataset: Dataset,
    target_key: str = "target",
    replacement: bool = True
) -> WeightedRandomSampler:
    """
    Create a balanced sampler for datasets with class imbalance.
    
    Args:
        dataset: Dataset to sample from
        target_key: Key for target labels (if dataset returns dict)
        replacement: Whether to sample with replacement
        
    Returns:
        WeightedRandomSampler
    """
    # Extract targets/labels
    targets = []
    for i in range(len(dataset)):
        sample = dataset[i]
        if isinstance(sample, dict) and target_key in sample:
            targets.append(sample[target_key])
        elif isinstance(sample, (tuple, list)) and len(sample) > 2:
            targets.append(sample[2])  # Assume third element is target
        else:
            # For datasets without explicit targets, use index as proxy
            targets.append(i % 10)  # Create 10 pseudo-classes
    
    # Count occurrences
    target_counts = Counter(targets)
    
    # Compute weights
    total_samples = len(targets)
    num_classes = len(target_counts)
    
    class_weights = {
        cls: total_samples / (num_classes * count)
        for cls, count in target_counts.items()
    }
    
    sample_weights = [class_weights[target] for target in targets]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=total_samples,
        replacement=replacement
    )


def save_dataset_info(
    dataset: Dataset,
    save_path: str,
    include_samples: bool = False,
    num_sample_paths: int = 10
):
    """
    Save dataset information to JSON file.
    
    Args:
        dataset: Dataset to document
        save_path: Path to save JSON file
        include_samples: Whether to include sample file paths
        num_sample_paths: Number of sample paths to include
    """
    info = {
        "dataset_type": type(dataset).__name__,
        "total_samples": len(dataset),
        "dataset_path": getattr(dataset, 'data_path', 'N/A'),
    }
    
    # Add integrity check
    integrity = check_data_integrity(dataset, num_samples=50)
    info["integrity_check"] = {
        "error_rate": integrity["error_rate"],
        "shape_consistency": integrity.get("shape_consistency", False),
        "dtype_consistency": integrity.get("dtype_consistency", False)
    }
    
    # Add statistics
    stats = get_data_stats(dataset, num_samples=100)
    info["statistics"] = stats
    
    # Add sample paths if requested
    if include_samples and hasattr(dataset, 'data_files'):
        sample_paths = dataset.data_files[:num_sample_paths]
        info["sample_paths"] = sample_paths
    
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Dataset info saved to {save_path}")


def load_dataset_from_config(config_path: str) -> Dataset:
    """
    Load dataset from configuration file.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        Instantiated dataset
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    dataset_type = config.get("dataset_type")
    dataset_args = config.get("dataset_args", {})
    
    # Import the appropriate dataset class
    from .dataset import create_dataset
    
    return create_dataset(dataset_type, **dataset_args)


# Utility functions for common data operations
def compute_dataset_mean_std(
    dataset: Dataset,
    num_samples: int = 1000
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and standard deviation for normalization.
    
    Args:
        dataset: Dataset to analyze
        num_samples: Number of samples to use
        
    Returns:
        Tuple of (mean, std) tensors
    """
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    values = []
    for idx in indices:
        sample = dataset[idx]
        if isinstance(sample, (tuple, list)):
            state_t = sample[0]
            if isinstance(state_t, torch.Tensor):
                values.append(state_t.flatten())
    
    if values:
        all_values = torch.cat(values)
        return all_values.mean(), all_values.std()
    else:
        return torch.tensor(0.0), torch.tensor(1.0)


def validate_dataset_compatibility(dataset: Dataset, model_input_shape: Tuple[int, ...]) -> bool:
    """
    Check if dataset output is compatible with model input.
    
    Args:
        dataset: Dataset to check
        model_input_shape: Expected input shape (without batch dimension)
        
    Returns:
        True if compatible, False otherwise
    """
    try:
        sample = dataset[0]
        if isinstance(sample, (tuple, list)):
            state_t = sample[0]
            if isinstance(state_t, torch.Tensor):
                return state_t.shape == model_input_shape
    except Exception:
        pass
    
    return False
