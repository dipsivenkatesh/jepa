"""
Data transformations for JEPA training.

Provides various data augmentation and preprocessing transforms that can be used
with JEPA datasets to improve model robustness and performance.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from typing import Tuple, List, Optional, Union, Callable, Any
import random
import numpy as np
from PIL import Image


class JEPATransforms:
    """Collection of transforms specifically designed for JEPA training."""
    
    @staticmethod
    def get_image_transforms(
        image_size: Union[int, Tuple[int, int]] = 224,
        normalize: bool = True,
        augment: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ) -> transforms.Compose:
        """
        Get standard image transforms for JEPA.
        
        Args:
            image_size: Target image size (int or (height, width))
            normalize: Whether to normalize with provided stats
            augment: Whether to apply data augmentation
            mean: Normalization mean values
            std: Normalization std values
            
        Returns:
            Composed transforms
        """
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        
        transform_list = []
        
        # Resize and crop
        transform_list.extend([
            transforms.Resize(max(image_size) + 32),
            transforms.CenterCrop(image_size)
        ])
        
        # Augmentation
        if augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.2, 
                    contrast=0.2, 
                    saturation=0.2, 
                    hue=0.1
                ),
                transforms.RandomResizedCrop(
                    image_size, 
                    scale=(0.8, 1.0), 
                    ratio=(0.9, 1.1)
                )
            ])
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalize
        if normalize:
            transform_list.append(transforms.Normalize(mean=mean, std=std))
        
        return transforms.Compose(transform_list)
    
    @staticmethod
    def get_minimal_transforms(
        image_size: Union[int, Tuple[int, int]] = 224,
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ) -> transforms.Compose:
        """Get minimal transforms for validation/testing."""
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        
        transform_list = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ]
        
        if normalize:
            transform_list.append(transforms.Normalize(mean=mean, std=std))
        
        return transforms.Compose(transform_list)
    
    @staticmethod
    def get_tensor_transforms(
        normalize: bool = True,
        add_noise: bool = False,
        noise_std: float = 0.01
    ) -> transforms.Compose:
        """Get transforms for tensor data."""
        transform_list = []
        
        if add_noise:
            transform_list.append(AddGaussianNoise(std=noise_std))
        
        if normalize:
            transform_list.append(TensorNormalize())
        
        return transforms.Compose(transform_list)


class TemporalAugmentation:
    """Temporal-specific augmentations for sequence data."""
    
    def __init__(
        self, 
        temporal_jitter_prob: float = 0.1,
        temporal_jitter_std: float = 0.01,
        temporal_dropout_prob: float = 0.05,
        temporal_dropout_rate: float = 0.1
    ):
        """
        Initialize temporal augmentation.
        
        Args:
            temporal_jitter_prob: Probability of applying temporal jittering
            temporal_jitter_std: Standard deviation of temporal jitter noise
            temporal_dropout_prob: Probability of applying temporal dropout
            temporal_dropout_rate: Rate of temporal dropout
        """
        self.temporal_jitter_prob = temporal_jitter_prob
        self.temporal_jitter_std = temporal_jitter_std
        self.temporal_dropout_prob = temporal_dropout_prob
        self.temporal_dropout_rate = temporal_dropout_rate
    
    def __call__(self, sequence: torch.Tensor) -> torch.Tensor:
        """Apply temporal augmentations to sequence."""
        # Temporal jittering
        if random.random() < self.temporal_jitter_prob:
            jitter = torch.randn_like(sequence) * self.temporal_jitter_std
            sequence = sequence + jitter
        
        # Temporal dropout
        if random.random() < self.temporal_dropout_prob:
            mask = torch.rand(sequence.shape[0]) > self.temporal_dropout_rate
            sequence = sequence * mask.unsqueeze(-1)
        
        return sequence


class MaskingTransform:
    """Random masking transform for self-supervised learning."""
    
    def __init__(
        self, 
        mask_ratio: float = 0.15, 
        patch_size: Optional[int] = None,
        mask_value: float = 0.0
    ):
        """
        Initialize masking transform.
        
        Args:
            mask_ratio: Fraction of elements to mask
            patch_size: Size of patches for patch-based masking (None for random masking)
            mask_value: Value to use for masked elements
        """
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.mask_value = mask_value
    
    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random masking and return masked input + mask."""
        if self.patch_size is not None:
            return self._patch_masking(x)
        else:
            return self._random_masking(x)
    
    def _random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random element-wise masking."""
        mask = torch.rand(x.shape) < self.mask_ratio
        masked_x = x.clone()
        masked_x[mask] = self.mask_value
        return masked_x, mask.float()
    
    def _patch_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply patch-based masking (for images)."""
        if len(x.shape) != 3:  # Assuming CHW format
            raise ValueError("Patch masking requires 3D tensor (CHW format)")
        
        C, H, W = x.shape
        patch_h, patch_w = self.patch_size, self.patch_size
        
        num_patches_h = H // patch_h
        num_patches_w = W // patch_w
        total_patches = num_patches_h * num_patches_w
        num_masked = int(total_patches * self.mask_ratio)
        
        # Randomly select patches to mask
        patch_indices = torch.randperm(total_patches)[:num_masked]
        
        masked_x = x.clone()
        mask = torch.zeros_like(x)
        
        for idx in patch_indices:
            i = idx // num_patches_w
            j = idx % num_patches_w
            
            start_h = i * patch_h
            end_h = start_h + patch_h
            start_w = j * patch_w
            end_w = start_w + patch_w
            
            masked_x[:, start_h:end_h, start_w:end_w] = self.mask_value
            mask[:, start_h:end_h, start_w:end_w] = 1.0
        
        return masked_x, mask


class AddGaussianNoise:
    """Add Gaussian noise to tensors."""
    
    def __init__(self, mean: float = 0.0, std: float = 0.01):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise


class TensorNormalize:
    """Normalize tensors to zero mean and unit variance."""
    
    def __init__(self, dim: Optional[Union[int, Tuple[int, ...]]] = None):
        self.dim = dim
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.dim is not None:
            mean = tensor.mean(dim=self.dim, keepdim=True)
            std = tensor.std(dim=self.dim, keepdim=True)
        else:
            mean = tensor.mean()
            std = tensor.std()
        
        return (tensor - mean) / (std + 1e-8)


class RandomCrop3D:
    """Random crop for 3D tensors (e.g., video data)."""
    
    def __init__(self, size: Tuple[int, int, int]):
        self.size = size
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply random 3D crop."""
        if len(tensor.shape) != 3:
            raise ValueError("Expected 3D tensor")
        
        d, h, w = tensor.shape
        target_d, target_h, target_w = self.size
        
        if d < target_d or h < target_h or w < target_w:
            raise ValueError("Tensor smaller than crop size")
        
        start_d = random.randint(0, d - target_d)
        start_h = random.randint(0, h - target_h)
        start_w = random.randint(0, w - target_w)
        
        return tensor[
            start_d:start_d + target_d,
            start_h:start_h + target_h,
            start_w:start_w + target_w
        ]


class SimCLRTransform:
    """SimCLR-style augmentations for contrastive learning."""
    
    def __init__(
        self,
        image_size: int = 224,
        color_jitter_strength: float = 0.5,
        blur_prob: float = 0.5
    ):
        self.image_size = image_size
        self.color_jitter_strength = color_jitter_strength
        self.blur_prob = blur_prob
        
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    0.8 * color_jitter_strength,
                    0.8 * color_jitter_strength,
                    0.8 * color_jitter_strength,
                    0.2 * color_jitter_strength
                )
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=blur_prob),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        return self.transform(x)


class DualTransform:
    """Apply different transforms to state_t and state_t1."""
    
    def __init__(
        self,
        transform_t: Callable,
        transform_t1: Optional[Callable] = None
    ):
        self.transform_t = transform_t
        self.transform_t1 = transform_t1 or transform_t
    
    def __call__(self, x: Any) -> Tuple[Any, Any]:
        """Apply transforms to create two views."""
        return self.transform_t(x), self.transform_t1(x)


# Convenience functions
def get_training_transforms(
    image_size: int = 224,
    augment: bool = True
) -> transforms.Compose:
    """Get transforms for training."""
    return JEPATransforms.get_image_transforms(
        image_size=image_size,
        normalize=True,
        augment=augment
    )


def get_validation_transforms(image_size: int = 224) -> transforms.Compose:
    """Get transforms for validation."""
    return JEPATransforms.get_minimal_transforms(
        image_size=image_size,
        normalize=True
    )


def get_tensor_training_transforms(
    add_noise: bool = True,
    noise_std: float = 0.01
) -> transforms.Compose:
    """Get transforms for tensor training data."""
    return JEPATransforms.get_tensor_transforms(
        normalize=True,
        add_noise=add_noise,
        noise_std=noise_std
    )


def create_augmentation_pair(
    base_transform: Callable,
    augment_prob: float = 0.8
) -> DualTransform:
    """Create a dual transform for augmentation pairs."""
    
    # Strong augmentation
    strong_transform = transforms.Compose([
        base_transform,
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=augment_prob),
        transforms.RandomGrayscale(p=0.2)
    ])
    
    return DualTransform(base_transform, strong_transform)


# Transform composition utilities
class ComposeTransforms:
    """Utility class for composing multiple transform types."""
    
    @staticmethod
    def for_images_and_tensors(
        image_transform: Callable,
        tensor_transform: Optional[Callable] = None
    ) -> Callable:
        """Create transform that works with both images and tensors."""
        
        def combined_transform(x):
            if isinstance(x, Image.Image):
                return image_transform(x)
            elif isinstance(x, torch.Tensor):
                if tensor_transform is not None:
                    return tensor_transform(x)
                else:
                    return x
            else:
                raise TypeError(f"Unsupported input type: {type(x)}")
        
        return combined_transform
    
    @staticmethod
    def chain(*transforms: Callable) -> Callable:
        """Chain multiple transforms together."""
        
        def chained_transform(x):
            for transform in transforms:
                x = transform(x)
            return x
        
        return chained_transform


# Presets for common use cases
class TransformPresets:
    """Predefined transform configurations for common scenarios."""
    
    @staticmethod
    def imagenet_style(image_size: int = 224, training: bool = True):
        """ImageNet-style transforms."""
        if training:
            return get_training_transforms(image_size)
        else:
            return get_validation_transforms(image_size)
    
    @staticmethod
    def contrastive_learning(image_size: int = 224):
        """Transforms for contrastive learning."""
        return SimCLRTransform(image_size)
    
    @staticmethod
    def minimal_preprocessing(image_size: int = 224):
        """Minimal preprocessing for testing."""
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])
    
    @staticmethod
    def temporal_augmentation(
        temporal_jitter: bool = True,
        masking: bool = False,
        mask_ratio: float = 0.15
    ):
        """Transforms for temporal data."""
        transform_list = []
        
        if temporal_jitter:
            transform_list.append(TemporalAugmentation())
        
        if masking:
            transform_list.append(MaskingTransform(mask_ratio=mask_ratio))
        
        return transforms.Compose(transform_list) if transform_list else None
