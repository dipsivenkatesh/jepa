# Data Loading Guide

This guide covers how to load and prepare data for JEPA training.

## Data Overview

JEPA works with various data modalities:

- **Images**: Computer vision tasks
- **Text**: Natural language processing
- **Time Series**: Sequential and temporal data
- **Audio**: Speech and sound processing
- **Multimodal**: Combined data types

The key requirement is that data can be split into context and target regions for self-supervised learning.

## Built-in Datasets

JEPA provides built-in support for common datasets:

### Image Datasets

```python
from data.dataset import ImageDataset

# Load ImageNet-style dataset
dataset = ImageDataset(
    root="data/imagenet",
    split="train",
    transform=transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
)
```

### Text Datasets

```python
from data.dataset import TextDataset

# Load text corpus
dataset = TextDataset(
    file_path="data/corpus.txt",
    tokenizer="bert-base-uncased",
    max_length=512,
    masking_strategy="random"
)
```

### Time Series Datasets

```python
from data.dataset import TimeSeriesDataset

# Load time series data
dataset = TimeSeriesDataset(
    file_path="data/timeseries.csv",
    window_size=100,
    stride=50,
    normalize=True
)
```

## Custom Datasets

### Creating Custom Datasets

```python
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.samples = self._load_samples()
    
    def _load_samples(self):
        # Load your data here
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
```

### Hugging Face Integration

```python
from data.hf_compatibility import HuggingFaceDataset

# Load from Hugging Face Hub
dataset = HuggingFaceDataset(
    dataset_name="imagenet-1k",
    split="train",
    streaming=True
)
```

## Data Transforms

### Image Transforms

```python
from data.transforms import ImageTransforms

transforms = ImageTransforms(
    resize=(224, 224),
    normalize=True,
    augmentation={
        'horizontal_flip': 0.5,
        'rotation': 15,
        'color_jitter': 0.1,
        'gaussian_blur': 0.2
    }
)
```

### Text Transforms

```python
from data.transforms import TextTransforms

transforms = TextTransforms(
    tokenizer="bert-base-uncased",
    max_length=512,
    padding=True,
    truncation=True
)
```

### Time Series Transforms

```python
from data.transforms import TimeSeriesTransforms

transforms = TimeSeriesTransforms(
    window_size=100,
    stride=50,
    normalization="z-score",
    noise_level=0.01
)
```

## Context-Target Generation

### Masking Strategies

```python
from data.utils import MaskingStrategy

# Random masking
random_mask = MaskingStrategy(
    strategy="random",
    mask_ratio=0.15,
    min_mask_length=1,
    max_mask_length=10
)

# Block masking
block_mask = MaskingStrategy(
    strategy="block",
    mask_ratio=0.15,
    block_size=16
)

# Structured masking
structured_mask = MaskingStrategy(
    strategy="structured",
    mask_ratio=0.15,
    pattern="grid"  # or "stripes", "patches"
)
```

### Context-Target Splitting

```yaml
data:
  context_target:
    context_ratio: 0.85      # 85% for context
    target_ratio: 0.15       # 15% for target
    overlap_allowed: false   # No overlap between context/target
    strategy: "random"       # How to select regions
```

## Data Loading Configuration

### Basic Configuration

```yaml
data:
  dataset_path: "data/train"
  batch_size: 64
  num_workers: 8
  pin_memory: true
  shuffle: true
  drop_last: true
```

### Advanced Configuration

```yaml
data:
  # Data paths
  train_path: "data/train"
  val_path: "data/val"
  test_path: "data/test"
  
  # Splits
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  
  # Loading
  batch_size: 64
  num_workers: 8
  pin_memory: true
  prefetch_factor: 2
  persistent_workers: true
  
  # Sampling
  sampler: "random"        # or "distributed", "weighted"
  shuffle: true
  drop_last: true
```

## Multi-Modal Data

### Image-Text Pairs

```python
from data.dataset import MultiModalDataset

dataset = MultiModalDataset(
    image_path="data/images",
    text_path="data/captions.json",
    image_transform=image_transforms,
    text_transform=text_transforms
)
```

### Configuration

```yaml
data:
  modalities: ["image", "text"]
  
  image:
    path: "data/images"
    format: "jpg"
    resize: [224, 224]
    
  text:
    path: "data/captions.json"
    tokenizer: "clip"
    max_length: 77
```

## Data Validation

### Automatic Validation

```python
from data.utils import validate_dataset

# Validate dataset structure
is_valid, errors = validate_dataset(dataset)
if not is_valid:
    print("Dataset validation errors:", errors)
```

### Manual Inspection

```python
# Inspect dataset samples
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
sample = next(iter(dataloader))

print(f"Sample shape: {sample.shape}")
print(f"Sample dtype: {sample.dtype}")
print(f"Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
```

## Performance Optimization

### Memory Optimization

```yaml
data:
  # Reduce memory usage
  pin_memory: true
  num_workers: 4           # Don't use too many workers
  prefetch_factor: 2       # Reasonable prefetching
  
  # For large datasets
  streaming: true          # Load data on-demand
  cache_size: 1000         # Cache frequently used samples
```

### Speed Optimization

```python
# Use optimized data loading
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)
```

### GPU Optimization

```python
# Move data to GPU efficiently
for batch in dataloader:
    batch = batch.to(device, non_blocking=True)
    # Training step...
```

## Data Augmentation

### Image Augmentation

```yaml
data:
  augmentation:
    horizontal_flip: 0.5
    vertical_flip: 0.1
    rotation: 15
    scale: [0.8, 1.2]
    color_jitter:
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
      hue: 0.1
    gaussian_blur: 0.2
    noise_level: 0.01
```

### Text Augmentation

```yaml
data:
  augmentation:
    synonym_replacement: 0.1
    random_insertion: 0.1
    random_swap: 0.1
    random_deletion: 0.1
    back_translation: false
```

### Time Series Augmentation

```yaml
data:
  augmentation:
    jittering: 0.01
    scaling: 0.1
    time_warping: 0.1
    window_slicing: true
    permutation: 0.1
```

## Distributed Data Loading

### Multi-GPU Setup

```python
from torch.utils.data.distributed import DistributedSampler

# Distributed sampler
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True
)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=num_workers
)
```

### Configuration

```yaml
data:
  distributed: true
  world_size: 4
  rank: 0               # Set automatically
  
  # Per-GPU batch size
  batch_size: 16        # Effective batch size = 16 * 4 = 64
```

## Data Pipeline Debugging

### Logging

```python
import logging

# Enable data loading logs
logging.getLogger('data').setLevel(logging.DEBUG)

# Log sample information
def log_batch_info(batch):
    logging.info(f"Batch shape: {batch.shape}")
    logging.info(f"Batch dtype: {batch.dtype}")
    logging.info(f"Memory usage: {batch.element_size() * batch.nelement()} bytes")
```

### Profiling

```python
import torch.profiler

# Profile data loading
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    record_shapes=True
) as prof:
    for batch in dataloader:
        # Process batch
        pass

print(prof.key_averages().table(sort_by="cpu_time_total"))
```

## Common Patterns

### Lazy Loading

```python
class LazyDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
    
    def __getitem__(self, idx):
        # Load data only when needed
        return self._load_file(self.file_list[idx])
```

### Cached Loading

```python
from functools import lru_cache

class CachedDataset(Dataset):
    @lru_cache(maxsize=1000)
    def _load_sample(self, idx):
        # Cache frequently accessed samples
        return self._load_data(idx)
```

### Streaming

```python
from torch.utils.data import IterableDataset

class StreamingDataset(IterableDataset):
    def __iter__(self):
        # Stream data continuously
        while True:
            yield self._get_next_sample()
```

## Troubleshooting

### Common Issues

**DataLoader hanging**
- Reduce `num_workers`
- Check for multiprocessing issues
- Disable `pin_memory` temporarily

**Out of memory**
- Reduce batch size
- Use data streaming
- Optimize transforms

**Slow data loading**
- Increase `num_workers`
- Use `pin_memory=True`
- Optimize data format (HDF5, Parquet)

**Inconsistent batch sizes**
- Set `drop_last=True`
- Check dataset length
- Verify sampler configuration

### Debug Mode

```python
# Enable debug mode
dataset.debug = True
dataloader.debug = True

# Minimal batch for testing
test_dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
```

## Best Practices

1. **Preprocess Once**: Prepare data offline when possible
2. **Use Appropriate Formats**: HDF5 for large datasets, Parquet for tabular data
3. **Monitor Memory**: Track memory usage during data loading
4. **Validate Early**: Check data integrity before training
5. **Profile Pipeline**: Identify bottlenecks in data loading
6. **Use Standards**: Follow common data formats and conventions

## Examples

For complete data loading examples, see:
- [Vision Data](../examples/vision.md)
- [NLP Data](../examples/nlp.md)
- [Time Series Data](../examples/timeseries.md)
- [Multi-Modal Data](../examples/multimodal.md)
