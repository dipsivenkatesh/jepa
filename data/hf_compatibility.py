"""
Hugging Face Datasets compatibility for JEPA.

Provides adapters and utilities to work with Hugging Face datasets
while maintaining compatibility with the existing JEPA data pipeline.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional, Callable, Union, List
import numpy as np
from PIL import Image

try:
    import datasets
    from datasets import load_dataset, Dataset as HFDataset, DatasetDict
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    # Create dummy classes to prevent import errors
    class HFDataset:
        pass
    class DatasetDict:
        pass

from .dataset import JEPADataset


class HuggingFaceJEPADataset(JEPADataset):
    """
    Adapter to use Hugging Face datasets with JEPA models.
    Converts HF dataset format to JEPA (state_t, state_t1) pairs.
    """
    
    def __init__(
        self,
        hf_dataset: Union[HFDataset, str],
        state_t_column: str = "image",
        state_t1_column: Optional[str] = None,
        temporal_offset: int = 1,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize HuggingFace JEPA dataset.
        
        Args:
            hf_dataset: HuggingFace dataset instance or dataset name
            state_t_column: Column name for state_t data
            state_t1_column: Column name for state_t1 data (if None, uses temporal offset)
            temporal_offset: Offset for creating pairs when state_t1_column is None
            transform: Transform for state_t
            target_transform: Transform for state_t1
            cache_dir: Cache directory for HF datasets
            **kwargs: Additional arguments
        """
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace datasets not available. Install with: pip install datasets")
        
        self.state_t_column = state_t_column
        self.state_t1_column = state_t1_column
        self.temporal_offset = temporal_offset
        self.cache_dir = cache_dir
        
        # Load dataset if string provided
        if isinstance(hf_dataset, str):
            self.hf_dataset = load_dataset(hf_dataset, cache_dir=cache_dir)
            # Use train split by default, or first available split
            if isinstance(self.hf_dataset, DatasetDict):
                split_name = 'train' if 'train' in self.hf_dataset else list(self.hf_dataset.keys())[0]
                self.hf_dataset = self.hf_dataset[split_name]
        else:
            self.hf_dataset = hf_dataset
        
        # Validate columns
        if state_t_column not in self.hf_dataset.column_names:
            raise ValueError(f"Column '{state_t_column}' not found in dataset. Available: {self.hf_dataset.column_names}")
        
        if state_t1_column and state_t1_column not in self.hf_dataset.column_names:
            raise ValueError(f"Column '{state_t1_column}' not found in dataset. Available: {self.hf_dataset.column_names}")
        
        # Don't call super().__init__() as we handle data loading differently
        self.transform = transform
        self.target_transform = target_transform
        self.return_indices = kwargs.get('return_indices', False)
        self.data_files = self._load_data_files()
    
    def _load_data_files(self) -> List[int]:
        """Return valid indices for the dataset."""
        if self.state_t1_column is not None:
            return list(range(len(self.hf_dataset)))
        else:
            # Account for temporal offset
            return list(range(len(self.hf_dataset) - self.temporal_offset))
    
    def __len__(self) -> int:
        if self.state_t1_column is not None:
            return len(self.hf_dataset)
        else:
            return len(self.hf_dataset) - self.temporal_offset
    
    def __getitem__(self, idx: int) -> tuple:
        """Get JEPA pair from HuggingFace dataset."""
        # Get state_t
        sample_t = self.hf_dataset[idx]
        state_t = self._process_sample(sample_t[self.state_t_column])
        
        # Get state_t1
        if self.state_t1_column is not None:
            state_t1 = self._process_sample(sample_t[self.state_t1_column])
        else:
            sample_t1 = self.hf_dataset[idx + self.temporal_offset]
            state_t1 = self._process_sample(sample_t1[self.state_t_column])
        
        # Apply transforms
        if self.transform:
            state_t = self.transform(state_t)
        if self.target_transform:
            state_t1 = self.target_transform(state_t1)
        elif self.transform:
            state_t1 = self.transform(state_t1)
        
        if self.return_indices:
            return state_t, state_t1, idx
        return state_t, state_t1
    
    def _process_sample(self, sample: Any) -> Union[torch.Tensor, Image.Image]:
        """Process individual sample based on type."""
        if isinstance(sample, Image.Image):
            # PIL Image - return as-is for transforms to handle
            return sample
        elif isinstance(sample, np.ndarray):
            return torch.from_numpy(sample).float()
        elif isinstance(sample, torch.Tensor):
            return sample.float()
        elif isinstance(sample, (list, tuple)):
            return torch.tensor(sample).float()
        elif isinstance(sample, dict) and 'bytes' in sample:
            # Handle HF image format
            from PIL import Image
            import io
            return Image.open(io.BytesIO(sample['bytes']))
        else:
            # Try to convert to tensor
            try:
                return torch.tensor(sample).float()
            except Exception:
                raise ValueError(f"Unsupported data type: {type(sample)}")


class JEPAToHuggingFace:
    """
    Converter to create HuggingFace datasets from JEPA-compatible data.
    """
    
    @staticmethod
    def from_image_directory(
        image_dir: str,
        dataset_name: str = "jepa_images",
        push_to_hub: bool = False,
        hub_repo_id: Optional[str] = None
    ) -> HFDataset:
        """
        Create HuggingFace dataset from image directory.
        
        Args:
            image_dir: Directory containing images
            dataset_name: Name for the dataset
            push_to_hub: Whether to push to HuggingFace Hub
            hub_repo_id: Repository ID for Hub upload
            
        Returns:
            HuggingFace Dataset
        """
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace datasets not available. Install with: pip install datasets")
        
        from .dataset import ImageSequenceDataset
        
        # Create JEPA dataset
        jepa_dataset = ImageSequenceDataset(image_dir)
        
        # Convert to HF format
        data_dict = {
            "image_path": jepa_dataset.data_files,
            "image": [Image.open(path) for path in jepa_dataset.data_files]
        }
        
        hf_dataset = HFDataset.from_dict(data_dict)
        
        if push_to_hub and hub_repo_id:
            hf_dataset.push_to_hub(hub_repo_id)
        
        return hf_dataset
    
    @staticmethod
    def from_tensor_data(
        data: torch.Tensor,
        dataset_name: str = "jepa_tensors",
        feature_names: Optional[List[str]] = None
    ) -> HFDataset:
        """
        Create HuggingFace dataset from tensor data.
        
        Args:
            data: Tensor data of shape [N, ...]
            dataset_name: Name for the dataset
            feature_names: Names for features
            
        Returns:
            HuggingFace Dataset
        """
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace datasets not available. Install with: pip install datasets")
        
        # Convert tensor to numpy for HF compatibility
        np_data = data.numpy()
        
        if len(np_data.shape) == 2:
            # 2D data - treat as feature vectors
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(np_data.shape[1])]
            
            data_dict = {name: np_data[:, i] for i, name in enumerate(feature_names)}
        else:
            # Multi-dimensional data
            data_dict = {"data": np_data}
        
        return HFDataset.from_dict(data_dict)


# Utility functions for HF integration
def load_jepa_hf_dataset(
    dataset_name: str,
    split: str = "train",
    state_t_column: str = "image",
    **kwargs
) -> HuggingFaceJEPADataset:
    """
    Load a HuggingFace dataset for JEPA training.
    
    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to use
        state_t_column: Column containing the data
        **kwargs: Additional arguments for HuggingFaceJEPADataset
        
    Returns:
        JEPA-compatible dataset
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace datasets not available. Install with: pip install datasets")
    
    hf_dataset = load_dataset(dataset_name, split=split)
    return HuggingFaceJEPADataset(
        hf_dataset=hf_dataset,
        state_t_column=state_t_column,
        **kwargs
    )


def create_jepa_dataset_card(
    dataset: HFDataset,
    description: str,
    usage_example: str
) -> str:
    """
    Create a dataset card for JEPA datasets on HuggingFace Hub.
    
    Args:
        dataset: HuggingFace dataset
        description: Dataset description
        usage_example: Example usage code
        
    Returns:
        Dataset card markdown
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace datasets not available. Install with: pip install datasets")
    
    card = f"""
# JEPA Dataset

## Description
{description}

## Dataset Structure
- **Total samples**: {len(dataset)}
- **Features**: {list(dataset.features.keys())}
- **Data types**: {dict(dataset.features)}

## Usage with JEPA

```python
from openjepa.data.hf_compatibility import HuggingFaceJEPADataset
from datasets import load_dataset

# Load dataset
hf_dataset = load_dataset("your_dataset_name")
jepa_dataset = HuggingFaceJEPADataset(hf_dataset["train"])

# Use with JEPA models
{usage_example}
```

## Data Format
This dataset is compatible with the OpenJEPA library and provides (state_t, state_t1) pairs for joint-embedding predictive architectures.
"""
    return card


# Integration with popular HF datasets
class PopularHFDatasets:
    """Pre-configured loaders for popular HuggingFace datasets."""
    
    @staticmethod
    def load_cifar10_jepa(**kwargs) -> HuggingFaceJEPADataset:
        """Load CIFAR-10 for JEPA training."""
        return load_jepa_hf_dataset(
            "cifar10",
            state_t_column="img",
            **kwargs
        )
    
    @staticmethod
    def load_imagenet_jepa(**kwargs) -> HuggingFaceJEPADataset:
        """Load ImageNet for JEPA training."""
        return load_jepa_hf_dataset(
            "imagenet-1k", 
            state_t_column="image",
            **kwargs
        )
    
    @staticmethod
    def load_coco_jepa(**kwargs) -> HuggingFaceJEPADataset:
        """Load COCO for JEPA training."""
        return load_jepa_hf_dataset(
            "detection-datasets/coco",
            state_t_column="image", 
            **kwargs
        )
    
    @staticmethod
    def load_food101_jepa(**kwargs) -> HuggingFaceJEPADataset:
        """Load Food-101 for JEPA training."""
        return load_jepa_hf_dataset(
            "food101",
            state_t_column="image",
            **kwargs
        )
    
    @staticmethod
    def load_oxford_pets_jepa(**kwargs) -> HuggingFaceJEPADataset:
        """Load Oxford-IIIT Pet Dataset for JEPA training."""
        return load_jepa_hf_dataset(
            "oxford-iiit-pet",
            state_t_column="image",
            **kwargs
        )


def check_hf_availability() -> bool:
    """Check if HuggingFace datasets is available."""
    return HF_AVAILABLE


def get_hf_dataset_info(dataset_name: str, split: str = "train") -> Dict[str, Any]:
    """
    Get information about a HuggingFace dataset.
    
    Args:
        dataset_name: Name of the HF dataset
        split: Dataset split to examine
        
    Returns:
        Dictionary with dataset information
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace datasets not available. Install with: pip install datasets")
    
    try:
        dataset = load_dataset(dataset_name, split=split)
        
        info = {
            "name": dataset_name,
            "split": split,
            "num_samples": len(dataset),
            "features": list(dataset.features.keys()),
            "feature_types": {k: str(v) for k, v in dataset.features.items()},
        }
        
        # Get sample for inspection
        if len(dataset) > 0:
            sample = dataset[0]
            info["sample_keys"] = list(sample.keys())
            info["sample_types"] = {k: type(v).__name__ for k, v in sample.items()}
        
        return info
        
    except Exception as e:
        return {"error": str(e), "dataset_name": dataset_name}
