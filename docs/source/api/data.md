# Data API

This section documents the data loading and processing modules.

## Core Dataset

```{eval-rst}
.. automodule:: data.dataset
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: data.dataset.JEPADataset
   :members:
   :special-members: __init__

.. autofunction:: data.dataset.get_dataset
```

## Structured Data Support

```{eval-rst}
.. automodule:: data.hf_compatibility
   :members:
   :undoc-members:
   :show-inheritance:
```

### JSONDataset

```{eval-rst}
.. autoclass:: data.dataset.JSONDataset
   :members:
   :special-members: __init__
```

Load data from JSON files with flexible key-based access:

```python
from jepa.data import JSONDataset

# Single array with temporal offset
dataset = JSONDataset(
    json_path="data.json",
    data_key="timeseries",
    time_offset=1
)

# Separate arrays format
dataset = JSONDataset(
    json_path="data.json",
    data_t_key="context_data",
    data_t1_key="target_data"
)
```

### CSVDataset

```{eval-rst}
.. autoclass:: data.dataset.CSVDataset
   :members:
   :special-members: __init__
```

Load data from CSV files with column-based organization:

```python
from jepa.data import CSVDataset

# Single columns with temporal offset
dataset = CSVDataset(
    csv_path="data.csv", 
    data_columns=["feature1", "feature2", "feature3"],
    time_offset=1
)
```

### PickleDataset

```{eval-rst}
.. autoclass:: data.dataset.PickleDataset
   :members:
   :special-members: __init__
```

Load data from Python pickle files:

```python
from jepa.data import PickleDataset

dataset = PickleDataset(
    pickle_path="data.pkl",
    data_key="sequences",
    time_offset=1
)
```

## Transforms

```{eval-rst}
.. automodule:: data.transforms
   :members:
   :undoc-members:
   :show-inheritance:
```

### Base Transform Classes

```{eval-rst}
.. autoclass:: data.transforms.BaseTransform
   :members:
   :special-members: __init__

.. autoclass:: data.transforms.JEPATransform
   :members:
   :special-members: __init__
```

### Image Transforms

```{eval-rst}
.. autoclass:: data.transforms.ImageMaskingTransform
   :members:
   :special-members: __init__

.. autoclass:: data.transforms.ImageAugmentTransform
   :members:
   :special-members: __init__
```

### Text Transforms

```{eval-rst}
.. autoclass:: data.transforms.TextMaskingTransform
   :members:
   :special-members: __init__

.. autoclass:: data.transforms.TokenizationTransform
   :members:
   :special-members: __init__
```

### Time Series Transforms

```{eval-rst}
.. autoclass:: data.transforms.TimeSeriesTransform
   :members:
   :special-members: __init__

.. autoclass:: data.transforms.TemporalMaskingTransform
   :members:
   :special-members: __init__
```

### Structured Data Transforms

```{eval-rst}
.. autoclass:: data.transforms.StructuredDataTransform
   :members:
   :special-members: __init__
```

Transform structured data with preprocessing pipeline:

```python
from jepa.data.transforms import StructuredDataTransform

transform = StructuredDataTransform([
    ("normalize", {"method": "z_score"}),
    ("handle_missing", {"strategy": "interpolate"}),
    ("feature_selection", {"top_k": 20})
])
```

## Data Utilities

```{eval-rst}
.. automodule:: data.utils
   :members:
   :undoc-members:
   :show-inheritance:
```

### Core Utilities

```{eval-rst}
.. autofunction:: data.utils.create_dataloader

.. autofunction:: data.utils.collate_fn

.. autofunction:: data.utils.validate_data_format
```

### Data Loading Helpers

```{eval-rst}
.. autofunction:: data.utils.load_json_data

.. autofunction:: data.utils.load_csv_data

.. autofunction:: data.utils.load_pickle_data
```

### Preprocessing Utilities

```{eval-rst}
.. autofunction:: data.utils.normalize_data

.. autofunction:: data.utils.handle_missing_values

.. autofunction:: data.utils.create_temporal_pairs
```

## Factory Functions

### Dataset Creation

```{eval-rst}
.. autofunction:: data.dataset.create_dataset
```

Create datasets with automatic format detection:

```python
from jepa.data import create_dataset

# Automatic format detection
dataset = create_dataset(
    data_path="data.json",  # Format inferred from extension
    data_key="features",
    temporal_offset=1
)

# Explicit format specification
dataset = create_dataset(
    data_type="csv",
    data_path="measurements.csv",
    data_columns=["sensor_1", "sensor_2"],
    temporal_offset=2
)
```

### DataLoader Creation

```{eval-rst}
.. autofunction:: data.utils.create_jepa_dataloader
```

Create optimized DataLoaders for JEPA training:

```python
from jepa.data.utils import create_jepa_dataloader

dataloader = create_jepa_dataloader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

## Hugging Face Integration

```{eval-rst}
.. automodule:: data.hf_compatibility
   :members:
   :undoc-members:
   :show-inheritance:
```

### HF Dataset Wrapper

```{eval-rst}
.. autoclass:: data.hf_compatibility.HFDatasetWrapper
   :members:
   :special-members: __init__
```

Use Hugging Face datasets with JEPA:

```python
from datasets import load_dataset
from jepa.data.hf_compatibility import HFDatasetWrapper

# Load HF dataset
hf_dataset = load_dataset("imagenet-1k", split="train")

# Wrap for JEPA
dataset = HFDatasetWrapper(
    hf_dataset,
    image_column="image",
    transform=JEPATransform()
)
```

### Custom HF Adapters

```{eval-rst}
.. autoclass:: data.hf_compatibility.BaseHFAdapter
   :members:
   :special-members: __init__

.. autoclass:: data.hf_compatibility.ImageHFAdapter
   :members:
   :special-members: __init__

.. autoclass:: data.hf_compatibility.TextHFAdapter
   :members:
   :special-members: __init__
```

## Configuration

### Data Configuration Classes

```{eval-rst}
.. autoclass:: config.config.DataConfig
   :members:
   :special-members: __init__
```

Configure data loading through YAML or programmatically:

```yaml
data:
  dataset_type: "csv"
  dataset_path: "data/train.csv"
  data_columns: ["feature1", "feature2", "feature3"]
  temporal_offset: 1
  batch_size: 64
  num_workers: 4
  transforms:
    normalize: true
    augment: true
```

## Examples

### Basic Data Loading

```python
from jepa.data import JSONDataset, create_dataloader

# Create dataset
dataset = JSONDataset(
    json_path="timeseries.json",
    data_key="measurements",
    temporal_offset=1
)

# Create dataloader
dataloader = create_dataloader(
    dataset,
    batch_size=32,
    shuffle=True
)

# Use in training
for batch in dataloader:
    context, target = batch
    # Process batch...
```

### Custom Dataset Implementation

```python
from jepa.data.dataset import BaseJEPADataset

class CustomDataset(BaseJEPADataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(**kwargs)
        self.data = self.load_custom_data(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
```

### Advanced Transform Pipeline

```python
from jepa.data.transforms import Compose, Normalize, Augment

# Create transform pipeline
transform = Compose([
    Normalize(method="z_score"),
    Augment(noise_std=0.1),
    JEPATransform(mask_ratio=0.15)
])

# Apply to dataset
dataset = CSVDataset(
    csv_path="data.csv",
    data_columns=["x", "y", "z"],
    temporal_offset=1,
    transform=transform
)
```

For more examples and detailed usage, see the [Data Loading Guide](../guides/data.md) and [Examples](../examples/index.md).
