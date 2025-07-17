"""
HuggingFace datasets integration examples for OpenJEPA.

This example demonstrates how to use HuggingFace datasets with JEPA models,
including loading popular datasets, creating custom datasets, and training.
"""

import torch
import numpy as np
from PIL import Image

# Import JEPA components
from models import JEPA, Encoder, Predictor
from data import (
    is_huggingface_available,
    get_training_transforms,
    get_validation_transforms,
    DataSplitter,
    DataLoader
)

# Import HF components if available
if is_huggingface_available():
    from data import (
        HuggingFaceJEPADataset,
        load_jepa_hf_dataset,
        PopularHFDatasets,
        JEPAToHuggingFace,
        get_hf_dataset_info,
        create_jepa_dataset_card
    )
    from datasets import Dataset as HFDataset


def example_popular_datasets():
    """Load and use popular HuggingFace datasets."""
    print("=== Popular HuggingFace Datasets ===")
    
    if not is_huggingface_available():
        print("HuggingFace datasets not available. Install with: pip install datasets")
        return
    
    # Example 1: CIFAR-10
    print("1. Loading CIFAR-10...")
    try:
        cifar_dataset = PopularHFDatasets.load_cifar10_jepa(
            split="train",
            transform=get_training_transforms(image_size=32)
        )
        print(f"   CIFAR-10 loaded: {len(cifar_dataset)} samples")
        
        # Test sample
        state_t, state_t1 = cifar_dataset[0]
        print(f"   Sample shapes: {state_t.shape}, {state_t1.shape}")
        
    except Exception as e:
        print(f"   CIFAR-10 failed: {e}")
    
    # Example 2: Food-101
    print("\n2. Loading Food-101...")
    try:
        food_dataset = PopularHFDatasets.load_food101_jepa(
            split="train",
            transform=get_training_transforms(image_size=224),
            temporal_offset=3
        )
        print(f"   Food-101 loaded: {len(food_dataset)} samples")
        
    except Exception as e:
        print(f"   Food-101 failed: {e}")
    
    # Example 3: Oxford Pets
    print("\n3. Loading Oxford-IIIT Pet Dataset...")
    try:
        pets_dataset = PopularHFDatasets.load_oxford_pets_jepa(
            split="train", 
            transform=get_validation_transforms(image_size=224)
        )
        print(f"   Oxford Pets loaded: {len(pets_dataset)} samples")
        
    except Exception as e:
        print(f"   Oxford Pets failed: {e}")


def example_custom_hf_dataset():
    """Create and use a custom HuggingFace dataset."""
    print("\n=== Custom HuggingFace Dataset ===")
    
    if not is_huggingface_available():
        print("HuggingFace datasets not available.")
        return
    
    print("Creating custom synthetic image dataset...")
    
    # Generate synthetic image data
    num_samples = 200
    image_size = 64
    
    images = []
    metadata = []
    
    for i in range(num_samples):
        # Create synthetic RGB image with patterns
        img_array = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
        
        # Add geometric patterns based on index
        pattern_type = i % 4
        if pattern_type == 0:  # Horizontal stripes
            img_array[::8, :, :] = 255
        elif pattern_type == 1:  # Vertical stripes
            img_array[:, ::8, :] = 255
        elif pattern_type == 2:  # Diagonal pattern
            for j in range(image_size):
                img_array[j, (j * 2) % image_size, :] = 255
        else:  # Checkerboard
            for x in range(0, image_size, 16):
                for y in range(0, image_size, 16):
                    if (x // 16 + y // 16) % 2:
                        img_array[x:x+16, y:y+16, :] = 255
        
        images.append(Image.fromarray(img_array))
        metadata.append({
            "pattern_type": pattern_type,
            "brightness": float(np.mean(img_array)),
            "sample_id": i
        })
    
    # Create HuggingFace dataset
    hf_data = HFDataset.from_dict({
        "image": images,
        "pattern_type": [m["pattern_type"] for m in metadata],
        "brightness": [m["brightness"] for m in metadata],
        "sample_id": [m["sample_id"] for m in metadata]
    })
    
    print(f"Created HF dataset with {len(hf_data)} samples")
    print(f"Features: {list(hf_data.features.keys())}")
    
    # Convert to JEPA format
    jepa_dataset = HuggingFaceJEPADataset(
        hf_dataset=hf_data,
        state_t_column="image",
        temporal_offset=5,  # Use images 5 steps apart
        transform=get_training_transforms(image_size=64)
    )
    
    print(f"JEPA dataset size: {len(jepa_dataset)}")
    
    # Test sample
    state_t, state_t1 = jepa_dataset[0]
    print(f"Sample shapes: {state_t.shape}, {state_t1.shape}")
    
    return jepa_dataset


def example_dataset_conversion():
    """Convert between JEPA and HuggingFace formats."""
    print("\n=== Dataset Format Conversion ===")
    
    if not is_huggingface_available():
        print("HuggingFace datasets not available.")
        return
    
    # Example 1: Convert tensor data to HF format
    print("1. Converting tensor data to HuggingFace format...")
    
    # Create some feature data
    num_samples = 300
    num_features = 128
    tensor_data = torch.randn(num_samples, num_features)
    
    # Add some structure to the data
    for i in range(num_samples):
        # Add periodic pattern
        tensor_data[i, :] += torch.sin(torch.arange(num_features) * (i / 50.0))
    
    # Convert to HF dataset
    hf_dataset = JEPAToHuggingFace.from_tensor_data(
        data=tensor_data,
        dataset_name="structured_features"
    )
    
    print(f"   Created HF dataset: {len(hf_dataset)} samples")
    print(f"   Features: {list(hf_dataset.features.keys())}")
    
    # Example 2: Convert back to JEPA format
    print("\n2. Converting HF dataset back to JEPA format...")
    
    jepa_dataset = HuggingFaceJEPADataset(
        hf_dataset=hf_dataset,
        state_t_column="data",
        temporal_offset=2
    )
    
    print(f"   JEPA dataset size: {len(jepa_dataset)}")
    
    # Test the conversion
    state_t, state_t1 = jepa_dataset[0]
    print(f"   Sample shapes: {state_t.shape}, {state_t1.shape}")
    
    # Verify data integrity
    original_sample = tensor_data[0]
    converted_sample = state_t
    
    similarity = torch.cosine_similarity(original_sample, converted_sample, dim=0)
    print(f"   Data similarity after conversion: {similarity.item():.4f}")
    
    return hf_dataset, jepa_dataset


def example_dataset_info():
    """Get information about HuggingFace datasets."""
    print("\n=== Dataset Information ===")
    
    if not is_huggingface_available():
        print("HuggingFace datasets not available.")
        return
    
    # Get info for popular datasets
    datasets_to_check = ["cifar10", "food101", "oxford-iiit-pet"]
    
    for dataset_name in datasets_to_check:
        print(f"\n{dataset_name}:")
        try:
            info = get_hf_dataset_info(dataset_name, split="train")
            
            print(f"  Samples: {info['num_samples']}")
            print(f"  Features: {info['features']}")
            print(f"  Feature types: {info['feature_types']}")
            
            if 'sample_types' in info:
                print(f"  Sample types: {info['sample_types']}")
                
        except Exception as e:
            print(f"  Error: {e}")


def example_training_with_hf():
    """Train a JEPA model using HuggingFace dataset."""
    print("\n=== Training with HuggingFace Dataset ===")
    
    if not is_huggingface_available():
        print("HuggingFace datasets not available.")
        return
    
    try:
        # Load a subset of CIFAR-10 for quick training
        print("Loading CIFAR-10 subset...")
        full_dataset = PopularHFDatasets.load_cifar10_jepa(
            split="train",
            transform=get_training_transforms(image_size=32)
        )
        
        # Use subset for demo
        subset_size = 1000
        indices = list(range(min(subset_size, len(full_dataset))))
        from torch.utils.data import Subset
        dataset = Subset(full_dataset, indices)
        
        print(f"Using {len(dataset)} samples for training demo")
        
        # Split data
        splits = DataSplitter.random_split(dataset, seed=42)
        
        # Create data loaders
        train_loader = DataLoader.create_jepa_dataloader(
            splits["train"],
            batch_size=32,
            num_workers=0
        )
        
        val_loader = DataLoader.create_evaluation_dataloader(
            splits["val"],
            batch_size=64,
            num_workers=0
        )
        
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
        # Create model (Note: This is simplified - CIFAR images need proper handling)
        hidden_dim = 512
        encoder = Encoder(hidden_dim)
        predictor = Predictor(hidden_dim)
        model = JEPA(encoder, predictor)
        
        print("Model created for HuggingFace data")
        
        # Quick test of data flow
        batch = next(iter(train_loader))
        state_t, state_t1 = batch
        print(f"Batch shapes: {state_t.shape}, {state_t1.shape}")
        
        # For actual training, you would use:
        # from trainer import create_trainer
        # trainer = create_trainer(model)
        # history = trainer.train(train_loader, num_epochs=10, val_dataloader=val_loader)
        
        print("Training setup complete! (Training step skipped for demo)")
        
    except Exception as e:
        print(f"Training example failed: {e}")
        print("This might be due to network issues or model architecture mismatch")


def example_dataset_card():
    """Create a dataset card for HuggingFace Hub."""
    print("\n=== Dataset Card Creation ===")
    
    if not is_huggingface_available():
        print("HuggingFace datasets not available.")
        return
    
    # Create a sample dataset
    sample_data = torch.randn(100, 64)
    hf_dataset = JEPAToHuggingFace.from_tensor_data(sample_data)
    
    # Create dataset card
    description = """
This is a sample dataset for demonstrating JEPA (Joint-Embedding Predictive Architecture) training.
The dataset contains synthetic feature vectors that can be used for self-supervised learning.
"""
    
    usage_example = """
from openjepa.data import HuggingFaceJEPADataset
from torch.utils.data import DataLoader

# Load and use with JEPA
dataset = HuggingFaceJEPADataset(hf_dataset, state_t_column="data")
loader = DataLoader(dataset, batch_size=32)

for state_t, state_t1 in loader:
    # Train your JEPA model
    predictions, targets = model(state_t, state_t1)
"""
    
    card = create_jepa_dataset_card(
        dataset=hf_dataset,
        description=description.strip(),
        usage_example=usage_example.strip()
    )
    
    print("Dataset card created:")
    print(card)


def main():
    """Run all HuggingFace examples."""
    print("OpenJEPA HuggingFace Integration Examples")
    print("=" * 60)
    
    if not is_huggingface_available():
        print("HuggingFace datasets not available!")
        print("Install with: pip install datasets")
        print("Then run this example again.")
        return
    
    print("HuggingFace datasets available! Running examples...\n")
    
    try:
        example_popular_datasets()
        example_custom_hf_dataset()
        example_dataset_conversion()
        example_dataset_info()
        example_training_with_hf()
        example_dataset_card()
        
        print("\n" + "=" * 60)
        print("All HuggingFace examples completed successfully!")
        print("\nKey benefits of HuggingFace integration:")
        print("• Access to thousands of datasets")
        print("• Automatic caching and optimization")
        print("• Easy sharing via HuggingFace Hub")
        print("• Standardized dataset format")
        print("• Seamless integration with JEPA models")
        
    except Exception as e:
        print(f"Error in HuggingFace examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
