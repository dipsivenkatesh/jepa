"""
Data utilities for OpenJEPA.

Provides dataset classes, transforms, and utilities for loading and preprocessing
data for JEPA training. Includes optional HuggingFace compatibility.
"""

from .dataset import (
    JEPADataset,
    ImageSequenceDataset,
    VideoFrameDataset,
    SyntheticDataset,
    TensorDataset,
    NumpyDataset,
    JSONDataset,
    CSVDataset,
    PickleDataset,
    create_dataset,
    collate_jepa_batch
)

from .transforms import (
    JEPATransforms,
    TemporalAugmentation,
    MaskingTransform,
    AddGaussianNoise,
    TensorNormalize,
    RandomCrop3D,
    SimCLRTransform,
    DualTransform,
    ComposeTransforms,
    TransformPresets,
    get_training_transforms,
    get_validation_transforms,
    get_tensor_training_transforms,
    create_augmentation_pair
)

from .utils import (
    DataSplitter,
    DataLoader,
    check_data_integrity,
    get_data_stats,
    visualize_batch,
    create_balanced_sampler,
    save_dataset_info,
    load_dataset_from_config,
    compute_dataset_mean_std,
    validate_dataset_compatibility
)

# HuggingFace compatibility (optional import)
try:
    from .hf_compatibility import (
        HuggingFaceJEPADataset,
        JEPAToHuggingFace,
        load_jepa_hf_dataset,
        PopularHFDatasets,
        create_jepa_dataset_card,
        check_hf_availability,
        get_hf_dataset_info
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

__all__ = [
    # Dataset classes
    'JEPADataset',
    'ImageSequenceDataset',
    'VideoFrameDataset',
    'SyntheticDataset',
    'TensorDataset',
    'NumpyDataset',
    'JSONDataset',
    'CSVDataset',
    'PickleDataset',
    'create_dataset',
    'collate_jepa_batch',
    
    # Transform classes
    'JEPATransforms',
    'TemporalAugmentation',
    'MaskingTransform',
    'AddGaussianNoise',
    'TensorNormalize',
    'RandomCrop3D',
    'SimCLRTransform',
    'DualTransform',
    'ComposeTransforms',
    'TransformPresets',
    
    # Transform functions
    'get_training_transforms',
    'get_validation_transforms',
    'get_tensor_training_transforms',
    'create_augmentation_pair',
    
    # Utilities
    'DataSplitter',
    'DataLoader',
    'check_data_integrity',
    'get_data_stats',
    'visualize_batch',
    'create_balanced_sampler',
    'save_dataset_info',
    'load_dataset_from_config',
    'compute_dataset_mean_std',
    'validate_dataset_compatibility'
]

# Add HF compatibility if available
if HF_AVAILABLE:
    __all__.extend([
        'HuggingFaceJEPADataset',
        'JEPAToHuggingFace',
        'load_jepa_hf_dataset',
        'PopularHFDatasets',
        'create_jepa_dataset_card',
        'check_hf_availability',
        'get_hf_dataset_info'
    ])

# Convenience function to check HF availability
def is_huggingface_available() -> bool:
    """Check if HuggingFace datasets integration is available."""
    return HF_AVAILABLE
