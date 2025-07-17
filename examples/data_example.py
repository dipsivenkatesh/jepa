"""
Comprehensive example showing how to use the JEPA data module.

This example demonstrates:
1. Creating different types of datasets
2. Applying transforms
3. Using data utilities
4. Training with the data
"""

import torch
import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader
import os
import tempfile

# Import JEPA components
from jepa.models import JEPA, Encoder, Predictor
from jepa.data import (
    # Datasets
    SyntheticDataset,
    ImageSequenceDataset,
    TensorDataset,
    create_dataset,
    
    # Transforms
    get_training_transforms,
    get_validation_transforms,
    get_tensor_training_transforms,
    TemporalAugmentation,
    MaskingTransform,
    TransformPresets,
    
    # Utilities
    DataSplitter,
    DataLoader,
    check_data_integrity,
    get_data_stats,
    visualize_batch,
    save_dataset_info,
    compute_dataset_mean_std,
    
    # HuggingFace compatibility (if available)
    is_huggingface_available
)

# Import HF components if available
if is_huggingface_available():
    from data import (
        HuggingFaceJEPADataset,
        load_jepa_hf_dataset,
        PopularHFDatasets,
        get_hf_dataset_info,
        check_hf_availability
    )


def create_sample_images(num_images: int = 20, image_size: int = 64):
    """Create sample images for demonstration."""
    import tempfile
    from PIL import Image
    
    temp_dir = tempfile.mkdtemp()
    print(f"Creating sample images in: {temp_dir}")
    
    for i in range(num_images):
        # Create a simple gradient image with some variation
        img_array = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        
        # Create gradient pattern
        for x in range(image_size):
            for y in range(image_size):
                img_array[x, y] = [
                    (x + i * 10) % 255,
                    (y + i * 5) % 255,
                    (x + y + i) % 255
                ]
        
        img = Image.fromarray(img_array)
        img.save(os.path.join(temp_dir, f"image_{i:03d}.png"))
    
    return temp_dir


def example_synthetic_dataset():
    """Example using synthetic dataset."""
    print("\n=== Synthetic Dataset Example ===")
    
    # Create synthetic dataset
    dataset = SyntheticDataset(
        num_samples=1000,
        data_shape=(3, 64, 64),  # RGB images
        correlation=0.8,
        noise_std=0.1,
        seed=42
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Check a sample
    state_t, state_t1 = dataset[0]
    print(f"Sample shapes: {state_t.shape}, {state_t1.shape}")
    
    # Check data integrity
    integrity = check_data_integrity(dataset, num_samples=100)
    print(f"Data integrity - Error rate: {integrity['error_rate']:.2%}")
    print(f"Shape consistency: {integrity.get('shape_consistency', False)}")
    
    # Get statistics
    stats = get_data_stats(dataset, num_samples=200)
    print(f"Correlation between t and t+1: {stats.get('correlation', 'N/A'):.3f}")
    
    return dataset


def example_image_dataset():
    """Example using image sequence dataset."""
    print("\n=== Image Dataset Example ===")
    
    # Create sample images
    image_dir = create_sample_images(num_images=50, image_size=64)
    
    try:
        # Create transforms
        train_transforms = get_training_transforms(image_size=64)
        val_transforms = get_validation_transforms(image_size=64)
        
        # Create dataset
        dataset = ImageSequenceDataset(
            data_path=image_dir,
            sequence_gap=1,
            transform=train_transforms,
            target_transform=val_transforms
        )
        
        print(f"Image dataset size: {len(dataset)}")
        
        # Check a sample
        state_t, state_t1 = dataset[0]
        print(f"Image sample shapes: {state_t.shape}, {state_t1.shape}")
        
        # Compute dataset statistics for normalization
        mean, std = compute_dataset_mean_std(dataset, num_samples=20)
        print(f"Dataset mean: {mean:.3f}, std: {std:.3f}")
        
        return dataset
        
    finally:
        # Clean up
        import shutil
        shutil.rmtree(image_dir)


def example_tensor_dataset():
    """Example using tensor dataset."""
    print("\n=== Tensor Dataset Example ===")
    
    # Create sample tensor data
    num_samples = 500
    data_dim = 128
    
    # Sequential data where each step is slightly different
    data_sequence = torch.randn(num_samples, data_dim)
    for i in range(1, num_samples):
        data_sequence[i] = 0.9 * data_sequence[i-1] + 0.1 * torch.randn(data_dim)
    
    # Create dataset
    dataset = TensorDataset(
        data_t=data_sequence,
        temporal_offset=1,
        transform=get_tensor_training_transforms()
    )
    
    print(f"Tensor dataset size: {len(dataset)}")
    
    # Check sample
    state_t, state_t1 = dataset[0]
    print(f"Tensor sample shapes: {state_t.shape}, {state_t1.shape}")
    
    return dataset


def example_data_splitting():
    """Example of data splitting strategies."""
    print("\n=== Data Splitting Example ===")
    
    # Create a dataset
    dataset = SyntheticDataset(num_samples=1000, data_shape=(64,))
    
    # Random split
    random_splits = DataSplitter.random_split(
        dataset,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42
    )
    
    print("Random split sizes:")
    for split_name, split_dataset in random_splits.items():
        print(f"  {split_name}: {len(split_dataset)}")
    
    # Temporal split (useful for time series)
    temporal_splits = DataSplitter.temporal_split(
        dataset,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )
    
    print("Temporal split sizes:")
    for split_name, split_dataset in temporal_splits.items():
        print(f"  {split_name}: {len(split_dataset)}")
    
    return random_splits


def example_transforms():
    """Example of different transform configurations."""
    print("\n=== Transforms Example ===")
    
    # Create a simple dataset
    dataset = SyntheticDataset(num_samples=100, data_shape=(3, 32, 32))
    
    # Different transform presets
    transforms_configs = {
        "training": get_training_transforms(image_size=32),
        "validation": get_validation_transforms(image_size=32),
        "contrastive": TransformPresets.contrastive_learning(image_size=32),
        "minimal": TransformPresets.minimal_preprocessing(image_size=32)
    }
    
    print("Transform configurations:")
    for name, transform in transforms_configs.items():
        print(f"  {name}: {len(transform.transforms)} transforms")
    
    # Temporal augmentation example
    temporal_aug = TemporalAugmentation(
        temporal_jitter_prob=0.2,
        temporal_dropout_prob=0.1
    )
    
    # Masking transform example
    masking = MaskingTransform(mask_ratio=0.15, patch_size=8)
    
    # Apply transforms to a sample
    sample_data = torch.randn(3, 32, 32)
    
    # Temporal augmentation
    augmented = temporal_aug(sample_data)
    print(f"Temporal augmentation applied: input {sample_data.shape} -> output {augmented.shape}")
    
    # Masking
    masked_data, mask = masking(sample_data)
    print(f"Masking applied: {mask.sum().item():.0f} elements masked ({mask.mean().item():.1%})")


def example_data_loading():
    """Example of data loading with different configurations."""
    print("\n=== Data Loading Example ===")
    
    # Create dataset
    dataset = SyntheticDataset(num_samples=200, data_shape=(64,))
    
    # Split dataset
    splits = DataSplitter.random_split(dataset, seed=42)
    
    # Create data loaders
    train_loader = DataLoader.create_jepa_dataloader(
        dataset=splits["train"],
        batch_size=32,
        shuffle=True,
        num_workers=0  # Set to 0 for demo
    )
    
    val_loader = DataLoader.create_evaluation_dataloader(
        dataset=splits["val"],
        batch_size=64,
        num_workers=0
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Check a batch
    train_batch = next(iter(train_loader))
    state_t, state_t1 = train_batch
    print(f"Training batch shapes: {state_t.shape}, {state_t1.shape}")
    
    return train_loader, val_loader


def example_full_training_pipeline():
    """Example of complete training pipeline with data."""
    print("\n=== Full Training Pipeline Example ===")
    
    # Create model
    hidden_dim = 64
    encoder = Encoder(hidden_dim)
    predictor = Predictor(hidden_dim)
    model = JEPA(encoder, predictor)
    
    # Create dataset and split
    dataset = SyntheticDataset(
        num_samples=500,
        data_shape=(10, hidden_dim),  # Sequence data
        correlation=0.8
    )
    
    splits = DataSplitter.random_split(dataset, seed=42)
    
    # Create data loaders
    train_loader = DataLoader.create_jepa_dataloader(
        splits["train"],
        batch_size=16,
        num_workers=0
    )
    
    val_loader = DataLoader.create_evaluation_dataloader(
        splits["val"],
        batch_size=32,
        num_workers=0
    )
    
    # Quick training demonstration
    from trainer import create_trainer
    
    trainer = create_trainer(
        model=model,
        learning_rate=1e-3,
        save_dir="./temp_checkpoints"
    )
    
    # Train for a few epochs
    print("Running quick training demo...")
    history = trainer.train(
        train_dataloader=train_loader,
        num_epochs=3,
        val_dataloader=val_loader
    )
    
    print(f"Training completed! Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    
    # Clean up
    import shutil
    if os.path.exists("./temp_checkpoints"):
        shutil.rmtree("./temp_checkpoints")


def example_huggingface_datasets():
    """Example using HuggingFace datasets."""
    print("\n=== HuggingFace Datasets Example ===")
    
    if not is_huggingface_available():
        print("HuggingFace datasets not available. Install with: pip install datasets")
        return None
    
    try:
        # Example 1: Load a popular dataset
        print("Loading CIFAR-10 from HuggingFace...")
        cifar_dataset = PopularHFDatasets.load_cifar10_jepa(
            split="train",
            transform=get_training_transforms(image_size=32)
        )
        
        print(f"CIFAR-10 dataset size: {len(cifar_dataset)}")
        
        # Check a sample
        state_t, state_t1 = cifar_dataset[0]
        print(f"CIFAR-10 sample shapes: {state_t.shape}, {state_t1.shape}")
        
        # Example 2: Load dataset by name
        print("\nLoading Food-101 from HuggingFace...")
        food_dataset = load_jepa_hf_dataset(
            "food101",
            split="train",
            state_t_column="image",
            transform=get_training_transforms(image_size=224),
            temporal_offset=5  # Use images 5 steps apart
        )
        
        print(f"Food-101 dataset size: {len(food_dataset)}")
        
        # Example 3: Get dataset information
        print("\nDataset information:")
        dataset_info = get_hf_dataset_info("cifar10", split="train")
        print(f"Dataset name: {dataset_info['name']}")
        print(f"Number of samples: {dataset_info['num_samples']}")
        print(f"Features: {dataset_info['features']}")
        
        return cifar_dataset
        
    except Exception as e:
        print(f"HuggingFace dataset example failed: {e}")
        print("This might be due to network issues or missing datasets library")
        return None


def example_hf_custom_dataset():
    """Example of creating and using a custom HuggingFace dataset."""
    print("\n=== Custom HuggingFace Dataset Example ===")
    
    if not is_huggingface_available():
        print("HuggingFace datasets not available.")
        return None
    
    try:
        from datasets import Dataset as HFDataset
        from PIL import Image
        import numpy as np
        
        # Create synthetic image data
        num_samples = 100
        image_size = 64
        
        images = []
        labels = []
        
        for i in range(num_samples):
            # Create synthetic RGB image
            img_array = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
            # Add some pattern based on index
            img_array[i % image_size, :, :] = 255  # Horizontal line
            img_array[:, i % image_size, :] = 128   # Vertical line
            
            images.append(Image.fromarray(img_array))
            labels.append(i % 10)  # 10 classes
        
        # Create HuggingFace dataset
        hf_data = HFDataset.from_dict({
            "image": images,
            "label": labels,
            "idx": list(range(num_samples))
        })
        
        print(f"Created custom HF dataset with {len(hf_data)} samples")
        
        # Convert to JEPA format
        jepa_dataset = HuggingFaceJEPADataset(
            hf_dataset=hf_data,
            state_t_column="image",
            temporal_offset=2,
            transform=get_training_transforms(image_size=64)
        )
        
        print(f"JEPA dataset size: {len(jepa_dataset)}")
        
        # Test a sample
        state_t, state_t1 = jepa_dataset[0]
        print(f"Custom dataset sample shapes: {state_t.shape}, {state_t1.shape}")
        
        return jepa_dataset
        
    except Exception as e:
        print(f"Custom HF dataset example failed: {e}")
        return None


def example_hf_dataset_conversion():
    """Example of converting JEPA data to HuggingFace format."""
    print("\n=== HuggingFace Dataset Conversion Example ===")
    
    if not is_huggingface_available():
        print("HuggingFace datasets not available.")
        return None
    
    try:
        from data import JEPAToHuggingFace
        
        # Create some tensor data
        tensor_data = torch.randn(200, 128)  # 200 samples, 128 features
        
        # Convert to HuggingFace dataset
        hf_dataset = JEPAToHuggingFace.from_tensor_data(
            data=tensor_data,
            dataset_name="example_tensor_data"
        )
        
        print(f"Converted tensor data to HF dataset: {len(hf_dataset)} samples")
        print(f"HF dataset features: {list(hf_dataset.features.keys())}")
        
        # Convert back to JEPA dataset
        jepa_dataset = HuggingFaceJEPADataset(
            hf_dataset=hf_dataset,
            state_t_column="data",
            temporal_offset=1
        )
        
        print(f"Converted back to JEPA dataset: {len(jepa_dataset)} samples")
        
        # Test sample
        state_t, state_t1 = jepa_dataset[0]
        print(f"Converted sample shapes: {state_t.shape}, {state_t1.shape}")
        
        return hf_dataset
        
    except Exception as e:
        print(f"HF dataset conversion example failed: {e}")
        return None


def example_dataset_factory():
    """Example using the dataset factory function."""
    print("\n=== Dataset Factory Example ===")
    
    # Create different datasets using factory
    datasets = {}
    
    # Synthetic dataset
    datasets["synthetic"] = create_dataset(
        data_type="synthetic",
        data_path="",  # Not needed for synthetic
        num_samples=200,
        data_shape=(32,)
    )
    
    # Tensor dataset
    sample_data = torch.randn(100, 64)
    datasets["tensor"] = create_dataset(
        data_type="tensor",
        data_path="",  # Not needed for tensor
        data_t=sample_data,
        temporal_offset=1
    )
    
    print("Created datasets:")
    for name, dataset in datasets.items():
        print(f"  {name}: {len(dataset)} samples, type: {type(dataset).__name__}")
        
        # Quick check
        sample = dataset[0]
        print(f"    Sample shapes: {sample[0].shape}, {sample[1].shape}")


def example_hf_training_pipeline():
    """Example of training with HuggingFace datasets."""
    print("\n=== HuggingFace Training Pipeline Example ===")
    
    if not is_huggingface_available():
        print("HuggingFace datasets not available for training pipeline.")
        return
    
    try:
        # Load a dataset from HuggingFace
        dataset = PopularHFDatasets.load_cifar10_jepa(
            split="train",
            transform=get_training_transforms(image_size=32),
            temporal_offset=1
        )
        
        # Take a subset for quick demo
        subset_size = 500
        indices = list(range(min(subset_size, len(dataset))))
        from torch.utils.data import Subset
        subset_dataset = Subset(dataset, indices)
        
        print(f"Using subset of {len(subset_dataset)} samples for training demo")
        
        # Split data
        splits = DataSplitter.random_split(subset_dataset, seed=42)
        
        # Create data loaders
        train_loader = DataLoader.create_jepa_dataloader(
            splits["train"],
            batch_size=16,
            num_workers=0
        )
        
        val_loader = DataLoader.create_evaluation_dataloader(
            splits["val"],
            batch_size=32,
            num_workers=0
        )
        
        # Create model for image data
        # CIFAR-10 images are 32x32x3, so we need to flatten or use CNN encoder
        from models import Encoder, Predictor, JEPA
        
        # Simple model that expects flattened images
        hidden_dim = 256
        encoder = Encoder(hidden_dim)  # This expects sequence input
        predictor = Predictor(hidden_dim)
        model = JEPA(encoder, predictor)
        
        print("Model created for HuggingFace data training")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
        # Quick test of data loading
        batch = next(iter(train_loader))
        state_t, state_t1 = batch
        print(f"Batch shapes: {state_t.shape}, {state_t1.shape}")
        
        print("HuggingFace training pipeline setup complete!")
        
    except Exception as e:
        print(f"HF training pipeline example failed: {e}")
        print("This might be due to network issues or model compatibility")
    """Example using the dataset factory function."""
    print("\n=== Dataset Factory Example ===")
    
    # Create different datasets using factory
    datasets = {}
    
    # Synthetic dataset
    datasets["synthetic"] = create_dataset(
        data_type="synthetic",
        data_path="",  # Not needed for synthetic
        num_samples=200,
        data_shape=(32,)
    )
    
    # Tensor dataset
    sample_data = torch.randn(100, 64)
    datasets["tensor"] = create_dataset(
        data_type="tensor",
        data_path="",  # Not needed for tensor
        data_t=sample_data,
        temporal_offset=1
    )
    
    print("Created datasets:")
    for name, dataset in datasets.items():
        print(f"  {name}: {len(dataset)} samples, type: {type(dataset).__name__}")
        
        # Quick check
        sample = dataset[0]
        print(f"    Sample shapes: {sample[0].shape}, {sample[1].shape}")


def main():
    """Run all data examples."""
    print("JEPA Data Module Examples")
    print("=" * 50)
    
    # Run examples
    try:
        example_synthetic_dataset()
        example_tensor_dataset()
        example_data_splitting()
        example_transforms()
        example_data_loading()
        example_dataset_factory()
        example_full_training_pipeline()
        
        # Try image dataset (may fail if PIL not available)
        try:
            example_image_dataset()
        except Exception as e:
            print(f"\nImage dataset example skipped: {e}")
        
        # Try HuggingFace examples (may fail if datasets not available)
        if is_huggingface_available():
            print(f"\nHuggingFace datasets available! Running HF examples...")
            try:
                example_huggingface_datasets()
                example_hf_custom_dataset()
                example_hf_dataset_conversion()
                example_hf_training_pipeline()
            except Exception as e:
                print(f"\nHuggingFace examples failed: {e}")
        else:
            print(f"\nHuggingFace datasets not available. Install with: pip install datasets")
        
        print("\n" + "=" * 50)
        print("All data examples completed successfully!")
        print("\nTo enable HuggingFace datasets support, install with:")
        print("  pip install datasets")
        
    except Exception as e:
        print(f"Error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
