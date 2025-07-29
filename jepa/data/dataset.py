"""
Dataset classes for JEPA training.

Provides flexible dataset implementations that can work with various data types
and create the (state_t, state_t1) pairs needed for JEPA training.
"""

import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Callable, Any, List, Union
import os
import numpy as np
from PIL import Image
import json
import pickle
import random
import csv
import pandas as pd


class JEPADataset(Dataset):
    """
    Base dataset class for JEPA that creates (state_t, state_t1) pairs.
    All JEPA datasets should inherit from this class.
    """
    
    def __init__(
        self,
        data_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_indices: bool = False
    ):
        """
        Initialize JEPA dataset.
        
        Args:
            data_path: Path to data directory or file
            transform: Optional transform for state_t
            target_transform: Optional transform for state_t1 (if None, uses transform)
            return_indices: Whether to return sample indices along with data
        """
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform
        self.return_indices = return_indices
        self.data_files = self._load_data_files()
        
        if len(self.data_files) == 0:
            raise ValueError(f"No data found in {data_path}")
    
    def _load_data_files(self) -> List[str]:
        """Load list of data files. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _load_data_files")
    
    def __len__(self) -> int:
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, int]]:
        """Return (state_t, state_t1) pair, optionally with index."""
        raise NotImplementedError("Subclasses must implement __getitem__")


class ImageSequenceDataset(JEPADataset):
    """
    Dataset for image sequences (e.g., video frames, consecutive images).
    Creates pairs of consecutive images for temporal prediction.
    """
    
    def __init__(
        self,
        data_path: str,
        sequence_gap: int = 1,
        valid_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'),
        **kwargs
    ):
        """
        Initialize image sequence dataset.
        
        Args:
            data_path: Path to directory containing images
            sequence_gap: Gap between state_t and state_t1 (1 = consecutive frames)
            valid_extensions: Tuple of valid image file extensions
            **kwargs: Additional arguments for parent class
        """
        self.sequence_gap = sequence_gap
        self.valid_extensions = valid_extensions
        super().__init__(data_path, **kwargs)
    
    def _load_data_files(self) -> List[str]:
        """Load image file paths."""
        if not os.path.isdir(self.data_path):
            raise ValueError(f"Data path {self.data_path} is not a directory")
        
        files = []
        for file in os.listdir(self.data_path):
            if file.lower().endswith(self.valid_extensions):
                files.append(os.path.join(self.data_path, file))
        
        return sorted(files)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, int]]:
        """Return consecutive image pair."""
        # Get current and next image (wrap around if at end)
        img1_path = self.data_files[idx]
        img2_idx = (idx + self.sequence_gap) % len(self.data_files)
        img2_path = self.data_files[img2_idx]
        
        # Load images
        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading images {img1_path} or {img2_path}: {e}")
        
        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
        if self.target_transform:
            img2 = self.target_transform(img2)
        elif self.transform:  # Use same transform if target_transform not provided
            img2 = self.transform(img2)
        
        if self.return_indices:
            return img1, img2, idx
        return img1, img2


class VideoFrameDataset(JEPADataset):
    """
    Dataset for video frames with temporal relationships.
    Loads frames from video files and creates temporal pairs.
    """
    
    def __init__(
        self,
        data_path: str,
        frame_interval: int = 1,
        max_frames: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize video frame dataset.
        
        Args:
            data_path: Path to directory containing video files or frame directories
            frame_interval: Interval between selected frames
            max_frames: Maximum number of frames to load per video
            **kwargs: Additional arguments for parent class
        """
        self.frame_interval = frame_interval
        self.max_frames = max_frames
        super().__init__(data_path, **kwargs)
    
    def _load_data_files(self) -> List[str]:
        """Load video frame paths."""
        # This is a simplified implementation
        # In practice, you might use cv2 or other libraries to extract frames
        valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        files = []
        
        if os.path.isdir(self.data_path):
            for file in os.listdir(self.data_path):
                if file.lower().endswith(valid_extensions):
                    files.append(os.path.join(self.data_path, file))
        
        return sorted(files)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, int]]:
        """Return frame pair from video."""
        # This is a placeholder implementation
        # In practice, you would extract frames from video files
        raise NotImplementedError("Video frame extraction requires additional dependencies (cv2, etc.)")


class SyntheticDataset(JEPADataset):
    """
    Synthetic dataset for testing and experimentation.
    Generates correlated synthetic data pairs.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        data_shape: Tuple[int, ...] = (64,),
        correlation: float = 0.8,
        noise_std: float = 0.1,
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize synthetic dataset.
        
        Args:
            num_samples: Number of synthetic samples to generate
            data_shape: Shape of each data sample (excluding batch dimension)
            correlation: Correlation between state_t and state_t1 (0-1)
            noise_std: Standard deviation of noise added to state_t1
            seed: Random seed for reproducibility
            **kwargs: Additional arguments for parent class
        """
        self.num_samples = num_samples
        self.data_shape = data_shape
        self.correlation = correlation
        self.noise_std = noise_std
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Pre-generate all data for consistency
        self._generate_data()
        
        # Use empty string as data_path since we don't load from files
        super().__init__(data_path="", **kwargs)
    
    def _generate_data(self):
        """Pre-generate all synthetic data."""
        self.data_t = torch.randn(self.num_samples, *self.data_shape)
        
        # Generate correlated state_t1
        noise = torch.randn(self.num_samples, *self.data_shape) * self.noise_std
        self.data_t1 = self.correlation * self.data_t + np.sqrt(1 - self.correlation**2) * noise
    
    def _load_data_files(self) -> List[int]:
        """Return list of sample indices."""
        return list(range(self.num_samples))
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, int]]:
        """Return synthetic (state_t, state_t1) pair."""
        state_t = self.data_t[idx]
        state_t1 = self.data_t1[idx]
        
        # Apply transforms if provided
        if self.transform:
            state_t = self.transform(state_t)
        if self.target_transform:
            state_t1 = self.target_transform(state_t1)
        elif self.transform:
            state_t1 = self.transform(state_t1)
        
        if self.return_indices:
            return state_t, state_t1, idx
        return state_t, state_t1


class TensorDataset(JEPADataset):
    """
    Dataset for pre-loaded tensor data.
    Useful when you have data already in tensor format.
    """
    
    def __init__(
        self,
        data_t: torch.Tensor,
        data_t1: Optional[torch.Tensor] = None,
        temporal_offset: int = 1,
        **kwargs
    ):
        """
        Initialize tensor dataset.
        
        Args:
            data_t: Tensor of shape [N, ...] containing all time steps
            data_t1: Optional tensor for state_t1. If None, uses temporal_offset
            temporal_offset: Offset for creating pairs when data_t1 is None
            **kwargs: Additional arguments for parent class
        """
        self.data_t_tensor = data_t
        self.data_t1_tensor = data_t1
        self.temporal_offset = temporal_offset
        
        super().__init__(data_path="", **kwargs)
    
    def _load_data_files(self) -> List[int]:
        """Return list of valid sample indices."""
        if self.data_t1_tensor is not None:
            return list(range(min(len(self.data_t_tensor), len(self.data_t1_tensor))))
        else:
            # Account for temporal offset
            return list(range(len(self.data_t_tensor) - self.temporal_offset))
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, int]]:
        """Return tensor pair."""
        state_t = self.data_t_tensor[idx]
        
        if self.data_t1_tensor is not None:
            state_t1 = self.data_t1_tensor[idx]
        else:
            state_t1 = self.data_t_tensor[idx + self.temporal_offset]
        
        # Apply transforms if provided
        if self.transform:
            state_t = self.transform(state_t)
        if self.target_transform:
            state_t1 = self.target_transform(state_t1)
        elif self.transform:
            state_t1 = self.transform(state_t1)
        
        if self.return_indices:
            return state_t, state_t1, idx
        return state_t, state_t1


class NumpyDataset(JEPADataset):
    """
    Dataset for numpy arrays saved as .npy or .npz files.
    """
    
    def __init__(
        self,
        data_path: str,
        data_key: str = "data",
        temporal_offset: int = 1,
        **kwargs
    ):
        """
        Initialize numpy dataset.
        
        Args:
            data_path: Path to .npy or .npz file
            data_key: Key to use for .npz files
            temporal_offset: Offset for creating temporal pairs
            **kwargs: Additional arguments for parent class
        """
        self.data_key = data_key
        self.temporal_offset = temporal_offset
        super().__init__(data_path, **kwargs)
    
    def _load_data_files(self) -> List[int]:
        """Load numpy data and return indices."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file {self.data_path} not found")
        
        if self.data_path.endswith('.npy'):
            self.data = np.load(self.data_path)
        elif self.data_path.endswith('.npz'):
            data_file = np.load(self.data_path)
            self.data = data_file[self.data_key]
        else:
            raise ValueError(f"Unsupported file format for {self.data_path}")
        
        # Convert to tensor
        self.data = torch.from_numpy(self.data).float()
        
        return list(range(len(self.data) - self.temporal_offset))
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, int]]:
        """Return numpy data pair."""
        state_t = self.data[idx]
        state_t1 = self.data[idx + self.temporal_offset]
        
        # Apply transforms if provided
        if self.transform:
            state_t = self.transform(state_t)
        if self.target_transform:
            state_t1 = self.target_transform(state_t1)
        elif self.transform:
            state_t1 = self.transform(state_t1)
        
        if self.return_indices:
            return state_t, state_t1, idx
        return state_t, state_t1


class JSONDataset(JEPADataset):
    """
    Dataset for data stored in JSON format.
    Supports various JSON structures for JEPA training.
    """
    
    def __init__(
        self,
        data_path: str,
        data_t_key: str = "data",
        data_t1_key: Optional[str] = None,
        temporal_offset: int = 1,
        nested_keys: Optional[List[str]] = None,
        data_format: str = "list",  # "list", "dict", "nested"
        **kwargs
    ):
        """
        Initialize JSON dataset.
        
        Args:
            data_path: Path to JSON file
            data_t_key: Key for state_t data in JSON
            data_t1_key: Key for state_t1 data (if None, uses temporal offset)
            temporal_offset: Offset for creating pairs when data_t1_key is None
            nested_keys: Keys to navigate nested JSON structure
            data_format: Format of JSON data ("list", "dict", "nested")
            **kwargs: Additional arguments for parent class
            
        JSON Format Examples:
            1. List format: {"data_t": [[1,2,3], [4,5,6], ...]}
            2. Dict format: {"samples": [{"data": [1,2,3], "id": 0}, ...]}
            3. Nested format: {"dataset": {"train": {"data_t": [...], "data_t1": [...]}}}
        """
        self.data_t_key = data_t_key
        self.data_t1_key = data_t1_key
        self.temporal_offset = temporal_offset
        self.nested_keys = nested_keys or []
        self.data_format = data_format
        
        super().__init__(data_path, **kwargs)
    
    def _load_data_files(self) -> List[int]:
        """Load JSON data and return indices."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"JSON file {self.data_path} not found")
        
        try:
            with open(self.data_path, 'r') as f:
                json_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file {self.data_path}: {e}")
        
        # Navigate nested structure if needed
        current_data = json_data
        for key in self.nested_keys:
            if key not in current_data:
                raise KeyError(f"Key '{key}' not found in JSON structure")
            current_data = current_data[key]
        
        # Parse based on data format
        if self.data_format == "list":
            self._parse_list_format(current_data)
        elif self.data_format == "dict":
            self._parse_dict_format(current_data)
        elif self.data_format == "nested":
            self._parse_nested_format(current_data)
        else:
            raise ValueError(f"Unknown data format: {self.data_format}")
        
        # Return valid indices
        if hasattr(self, 'data_t1') and self.data_t1 is not None:
            return list(range(min(len(self.data_t), len(self.data_t1))))
        else:
            return list(range(len(self.data_t) - self.temporal_offset))
    
    def _parse_list_format(self, json_data: dict):
        """Parse JSON with list format: {"data_t": [...], "data_t1": [...]}"""
        if self.data_t_key not in json_data:
            raise KeyError(f"Key '{self.data_t_key}' not found in JSON data")
        
        self.data_t = torch.tensor(json_data[self.data_t_key]).float()
        
        if self.data_t1_key and self.data_t1_key in json_data:
            self.data_t1 = torch.tensor(json_data[self.data_t1_key]).float()
        else:
            self.data_t1 = None
    
    def _parse_dict_format(self, json_data: dict):
        """Parse JSON with dict format: {"samples": [{"data": [...], "target": [...]}, ...]}"""
        if "samples" not in json_data:
            raise KeyError("Expected 'samples' key in dict format JSON")
        
        samples = json_data["samples"]
        if not isinstance(samples, list):
            raise ValueError("Samples must be a list in dict format")
        
        data_t_list = []
        data_t1_list = []
        
        for i, sample in enumerate(samples):
            if self.data_t_key not in sample:
                raise KeyError(f"Key '{self.data_t_key}' not found in sample {i}")
            
            data_t_list.append(sample[self.data_t_key])
            
            if self.data_t1_key and self.data_t1_key in sample:
                data_t1_list.append(sample[self.data_t1_key])
        
        self.data_t = torch.tensor(data_t_list).float()
        self.data_t1 = torch.tensor(data_t1_list).float() if data_t1_list else None
    
    def _parse_nested_format(self, json_data: dict):
        """Parse JSON with nested format: complex nested structure"""
        # For nested format, we expect the data to be directly accessible
        # This is a flexible parser for custom nested structures
        
        def extract_nested_data(data, key_path):
            """Extract data from nested dictionary using dot notation"""
            keys = key_path.split('.')
            current = data
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                elif isinstance(current, list) and key.isdigit():
                    current = current[int(key)]
                else:
                    raise KeyError(f"Key path '{key_path}' not found")
            return current
        
        try:
            data_t_raw = extract_nested_data(json_data, self.data_t_key)
            self.data_t = torch.tensor(data_t_raw).float()
            
            if self.data_t1_key:
                data_t1_raw = extract_nested_data(json_data, self.data_t1_key)
                self.data_t1 = torch.tensor(data_t1_raw).float()
            else:
                self.data_t1 = None
        except Exception as e:
            raise ValueError(f"Error parsing nested JSON format: {e}")
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, int]]:
        """Return JSON data pair."""
        state_t = self.data_t[idx]
        
        if self.data_t1 is not None:
            state_t1 = self.data_t1[idx]
        else:
            state_t1 = self.data_t[idx + self.temporal_offset]
        
        # Apply transforms if provided
        if self.transform:
            state_t = self.transform(state_t)
        if self.target_transform:
            state_t1 = self.target_transform(state_t1)
        elif self.transform:
            state_t1 = self.transform(state_t1)
        
        if self.return_indices:
            return state_t, state_t1, idx
        return state_t, state_t1


class CSVDataset(JEPADataset):
    """
    Dataset for loading data from CSV files.
    
    Supports two formats:
    1. Single data columns with temporal offsets
    2. Separate data_t and data_t1 columns
    """
    
    def __init__(
        self,
        csv_path: str,
        time_offset: int = 1,
        transform: Optional[Callable] = None,
        data_columns: Optional[List[str]] = None,
        data_t_columns: Optional[List[str]] = None,
        data_t1_columns: Optional[List[str]] = None,
        skip_header: bool = True
    ):
        """
        Args:
            csv_path: Path to CSV file
            time_offset: Temporal offset for single data format
            transform: Optional transform to apply to data
            data_columns: Column names for data (when using single format)
            data_t_columns: Column names for data_t (when using separate format)
            data_t1_columns: Column names for data_t1 (when using separate format)
            skip_header: Whether to skip the first row (header)
        """
        self.csv_path = csv_path
        self.time_offset = time_offset
        self.transform = transform
        
        # Load CSV data
        self.df = pd.read_csv(csv_path)
        
        # Determine format
        self.separate_columns = data_t_columns is not None and data_t1_columns is not None
        
        if self.separate_columns:
            self.data_t = torch.tensor(self.df[data_t_columns].values, dtype=torch.float32)
            self.data_t1 = torch.tensor(self.df[data_t1_columns].values, dtype=torch.float32)
            if len(self.data_t) != len(self.data_t1):
                raise ValueError("data_t and data_t1 must have the same length")
            self.length = len(self.data_t)
        else:
            if data_columns is None:
                # Use all numeric columns
                numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
                data_columns = numeric_columns
            self.data = torch.tensor(self.df[data_columns].values, dtype=torch.float32)
            self.length = len(self.data) - self.time_offset
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.separate_columns:
            data_t = self.data_t[idx]
            data_t1 = self.data_t1[idx]
        else:
            data_t = self.data[idx]
            data_t1 = self.data[idx + self.time_offset]
        
        if self.transform:
            data_t = self.transform(data_t)
            data_t1 = self.transform(data_t1)
        
        return data_t, data_t1


class PickleDataset(JEPADataset):
    """
    Dataset for loading data from pickle files.
    
    Supports two formats:
    1. Single data array with temporal offsets
    2. Separate data_t and data_t1 arrays
    """
    
    def __init__(
        self,
        pickle_path: str,
        time_offset: int = 1,
        transform: Optional[Callable] = None,
        data_key: str = "data",
        data_t_key: Optional[str] = None,
        data_t1_key: Optional[str] = None
    ):
        """
        Args:
            pickle_path: Path to pickle file
            time_offset: Temporal offset for single data array format
            transform: Optional transform to apply to data
            data_key: Key for data array (when using single array format)
            data_t_key: Key for data_t array (when using separate arrays)
            data_t1_key: Key for data_t1 array (when using separate arrays)
        """
        self.pickle_path = pickle_path
        self.time_offset = time_offset
        self.transform = transform
        
        with open(pickle_path, 'rb') as f:
            self.data_dict = pickle.load(f)
        
        # Determine which format we're using
        self.separate_arrays = data_t_key is not None and data_t1_key is not None
        
        if self.separate_arrays:
            self.data_t = torch.tensor(self.data_dict[data_t_key], dtype=torch.float32)
            self.data_t1 = torch.tensor(self.data_dict[data_t1_key], dtype=torch.float32)
            if len(self.data_t) != len(self.data_t1):
                raise ValueError("data_t and data_t1 must have the same length")
            self.length = len(self.data_t)
        else:
            if isinstance(self.data_dict, dict):
                self.data = torch.tensor(self.data_dict[data_key], dtype=torch.float32)
            else:
                # Assume the pickle file contains the data directly
                self.data = torch.tensor(self.data_dict, dtype=torch.float32)
            self.length = len(self.data) - self.time_offset
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.separate_arrays:
            data_t = self.data_t[idx]
            data_t1 = self.data_t1[idx]
        else:
            data_t = self.data[idx]
            data_t1 = self.data[idx + self.time_offset]
        
        if self.transform:
            data_t = self.transform(data_t)
            data_t1 = self.transform(data_t1)
        
        return data_t, data_t1


# Utility functions
def create_dataset(
    data_type: str,
    data_path: str,
    **kwargs
) -> JEPADataset:
    """
    Factory function to create appropriate dataset based on data type.
    
    Args:
        data_type: Type of dataset ('image', 'synthetic', 'tensor', 'numpy', 'json', 'csv', 'pickle')
        data_path: Path to data
        **kwargs: Additional arguments for dataset
        
    Returns:
        Appropriate dataset instance
    """
    dataset_map = {
        'image': ImageSequenceDataset,
        'synthetic': SyntheticDataset,
        'tensor': TensorDataset,
        'numpy': NumpyDataset,
        'json': JSONDataset,
        'csv': CSVDataset,
        'pickle': PickleDataset
    }
    
    if data_type not in dataset_map:
        raise ValueError(f"Unknown data type: {data_type}. Available: {list(dataset_map.keys())}")
    
    return dataset_map[data_type](data_path, **kwargs)


def collate_jepa_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for JEPA datasets.
    
    Args:
        batch: List of (state_t, state_t1) tuples
        
    Returns:
        Batched tensors
    """
    states_t = torch.stack([item[0] for item in batch])
    states_t1 = torch.stack([item[1] for item in batch])
    return states_t, states_t1
