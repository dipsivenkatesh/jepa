# Structured Data Support in JEPA

JEPA provides comprehensive support for structured data formats, making it easy to work with data stored in various file formats including JSON, CSV, Pickle, and other structured formats with flexible data organization patterns.

## Overview

The structured data module enables JEPA to work seamlessly with:

- **Tabular data** (CSV files, databases)
- **JSON documents** (nested structures, APIs)
- **Pickle files** (complex Python objects)
- **Time series data** (temporal sequences)
- **Mixed data types** (heterogeneous features)

All structured data classes support two main organization patterns:

:::{admonition} Single Array Format
:class: note
One data array with temporal offsets to create (t, t+1) pairs automatically
:::

:::{admonition} Separate Arrays Format
:class: tip
Distinct arrays for data_t and data_t1 when relationships are pre-defined
:::

## Dataset Classes

### JSONDataset

Load data from JSON files with flexible key-based access:

```python
from jepa.data import JSONDataset

# Single array with temporal offset
dataset = JSONDataset(
    json_path="data.json",
    data_key="timeseries",
    time_offset=1  # Use consecutive samples as (t, t+1) pairs
)

# Separate arrays format
dataset = JSONDataset(
    json_path="data.json",
    data_t_key="context_data",
    data_t1_key="target_data"
)
```

### CSVDataset

Load data from CSV files with column-based organization:

```python
from jepa.data import CSVDataset

# Single columns with temporal offset
dataset = CSVDataset(
    csv_path="data.csv", 
    data_columns=["feature1", "feature2", "feature3"],
    time_offset=1
)

# Separate columns for context and target
dataset = CSVDataset(
    csv_path="data.csv",
    data_t_columns=["t_feat1", "t_feat2"], 
    data_t1_columns=["t1_feat1", "t1_feat2"]
)

# Auto-detect numeric columns
dataset = CSVDataset(csv_path="data.csv", time_offset=1)
```

### PickleDataset

Load data from Python pickle files:

```python
from jepa.data import PickleDataset

# Dictionary format with temporal offset
dataset = PickleDataset(
    pickle_path="data.pkl",
    data_key="sequences",
    time_offset=1
)

# Separate arrays format
dataset = PickleDataset(
    pickle_path="data.pkl", 
    data_t_key="context_sequences",
    data_t1_key="target_sequences"
)
```

## Supported Data Formats

### JSON Data Structures

JEPA supports multiple JSON organization patterns:

#### Simple List Format

```json
{
  "data_t": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
  "data_t1": [[2, 3, 4], [5, 6, 7], [8, 9, 10]]
}
```

#### Dictionary of Samples

```json
{
  "samples": [
    {"features": [1, 2, 3], "target": [2, 3, 4], "id": 0},
    {"features": [4, 5, 6], "target": [5, 6, 7], "id": 1}
  ]
}
```

#### Nested Structures

```json
{
  "experiment": {
    "session_1": {
      "measurements": [[1, 2], [3, 4], [5, 6]],
      "labels": ["A", "B", "C"]
    }
  }
}
```

**Usage Examples:**

```python
# Simple list format
dataset = JSONDataset(
    data_path="data.json",
    data_t_key="data_t",
    data_t1_key="data_t1"
)

# Dictionary format
dataset = JSONDataset(
    data_path="samples.json",
    data_t_key="features",
    data_t1_key="target",
    samples_key="samples"  # Navigate to samples array
)

# Nested format with dot notation
dataset = JSONDataset(
    data_path="nested.json",
    data_t_key="experiment.session_1.measurements",
    temporal_offset=1
)
```

### CSV Data Structures

Handle various CSV organizations:

#### Standard Tabular Format

```csv
timestamp,sensor1,sensor2,sensor3,target1,target2
2023-01-01,1.0,2.0,3.0,1.1,2.1
2023-01-02,1.5,2.5,3.5,1.6,2.6
2023-01-03,2.0,3.0,4.0,2.1,3.1
```

#### Time Series Format

```csv
date,value,moving_avg,trend
2023-01-01,100,95,up
2023-01-02,102,98,up
2023-01-03,104,101,up
```

**Usage Examples:**

```python
# Explicit column specification
dataset = CSVDataset(
    csv_path="sensors.csv",
    data_t_columns=["sensor1", "sensor2", "sensor3"],
    data_t1_columns=["target1", "target2"]
)

# Time series with offset
dataset = CSVDataset(
    csv_path="timeseries.csv",
    data_columns=["value", "moving_avg"],
    temporal_offset=1,  # Predict next time step
    index_column="date"  # Use date as index
)

# Column indices (useful for headerless files)
dataset = CSVDataset(
    csv_path="data.csv",
    data_t_columns=[0, 1, 2],  # Use first 3 columns
    data_t1_columns=[3, 4]     # Use next 2 columns
)
```

### Pickle Data Structures

Support for complex Python objects:

```python
import pickle
import numpy as np

# Create complex data structure
data = {
    "sequences": np.random.randn(1000, 50),  # 1000 samples, 50 features
    "metadata": {
        "experiment_id": "exp_001",
        "sampling_rate": 100,
        "features": ["accel_x", "accel_y", "gyro_z", ...]
    },
    "labels": np.random.randint(0, 5, 1000)
}

# Save to pickle
with open("experiment.pkl", "wb") as f:
    pickle.dump(data, f)

# Load with PickleDataset
dataset = PickleDataset(
    pickle_path="experiment.pkl",
    data_key="sequences",
    temporal_offset=5,  # 5-step prediction
    metadata_key="metadata"  # Include metadata
)
```

## Factory Functions

Use factory functions for streamlined dataset creation:

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

# Advanced configuration
dataset = create_dataset(
    data_type="pickle",
    data_path="complex_data.pkl",
    data_t_key="context_embeddings",
    data_t1_key="target_embeddings",
    transform=CustomTransform(),
    cache_data=True  # Cache in memory for faster access
)
```

## Advanced Features

### Temporal Relationship Handling

```python
# Multiple temporal offsets
dataset = JSONDataset(
    json_path="timeseries.json",
    data_key="measurements",
    temporal_offsets=[1, 3, 5],  # Predict 1, 3, and 5 steps ahead
    multi_target=True
)

# Variable-length sequences
dataset = CSVDataset(
    csv_path="variable_length.csv",
    data_columns=["sequence"],
    sequence_column="sequence_id",  # Group by sequence ID
    temporal_offset=1,
    pad_sequences=True  # Pad to maximum length
)

# Sliding windows
dataset = PickleDataset(
    pickle_path="continuous_data.pkl",
    data_key="signal",
    window_size=100,    # 100-sample windows
    stride=50,          # 50% overlap
    temporal_offset=10  # Predict 10 samples ahead
)
```

### Data Preprocessing and Transforms

```python
from jepa.data.transforms import StructuredDataTransform

# Custom preprocessing pipeline
transform = StructuredDataTransform([
    ("normalize", {"method": "z_score"}),
    ("handle_missing", {"strategy": "interpolate"}),
    ("feature_selection", {"top_k": 20}),
    ("temporal_augment", {"jitter": 0.1})
])

dataset = CSVDataset(
    csv_path="raw_data.csv",
    data_columns=["feat_1", "feat_2", "feat_3"],
    temporal_offset=1,
    transform=transform
)
```

### Heterogeneous Data Types

```python
# Mixed data types
class MixedDataset(StructuredDataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = self.load_mixed_data(data_path)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Handle different data types
        numerical = torch.FloatTensor(sample["numerical"])
        categorical = torch.LongTensor(sample["categorical"])
        text = self.tokenize(sample["text"])
        
        return {
            "numerical": numerical,
            "categorical": categorical,
            "text": text
        }

# Usage with custom encoders
dataset = MixedDataset("mixed_data.json")
encoder = MixedDataEncoder(
    numerical_dim=10,
    categorical_vocab=1000,
    text_vocab=30000
)
```

## Integration with JEPA Training

### Basic Training Pipeline

```python
from jepa.data import create_dataset
from jepa.trainer import JEPATrainer
from jepa.models import StructuredDataEncoder, MLPPredictor

# Create dataset
dataset = create_dataset(
    data_type="csv",
    data_path="financial_data.csv",
    data_columns=["price", "volume", "rsi", "macd"],
    temporal_offset=1
)

# Create model components
encoder = StructuredDataEncoder(
    input_dim=4,        # 4 features
    hidden_dims=[64, 128, 64],
    output_dim=32
)

predictor = MLPPredictor(
    input_dim=32,
    hidden_dims=[64, 32],
    output_dim=32
)

# Configure and train
config = {
    'model': {'encoder': encoder, 'predictor': predictor},
    'training': {'epochs': 100, 'batch_size': 64},
    'data': {'dataset': dataset}
}

trainer = JEPATrainer(config)
trainer.train()
```

### Advanced Training with Custom Loss

```python
from jepa.losses import StructuredDataLoss

# Custom loss for structured data
class TimeSeriesLoss(StructuredDataLoss):
    def __init__(self, temporal_weight=0.1):
        super().__init__()
        self.temporal_weight = temporal_weight
    
    def forward(self, pred, target, metadata=None):
        # Standard reconstruction loss
        recon_loss = F.mse_loss(pred, target)
        
        # Temporal consistency loss
        if metadata and "temporal_context" in metadata:
            temporal_loss = self.temporal_consistency_loss(
                pred, metadata["temporal_context"]
            )
            return recon_loss + self.temporal_weight * temporal_loss
        
        return recon_loss

# Use custom loss
trainer = JEPATrainer(config, loss_fn=TimeSeriesLoss())
```

## Best Practices

### Data Organization

1. **Consistent Structure**: Maintain consistent data shapes and types
2. **Meaningful Names**: Use descriptive column/key names
3. **Documentation**: Include metadata about data sources and preprocessing
4. **Version Control**: Track data versions alongside code

### Performance Optimization

```python
# Memory-efficient loading
dataset = CSVDataset(
    csv_path="large_dataset.csv",
    data_columns=["feat1", "feat2"],
    temporal_offset=1,
    chunk_size=10000,    # Process in chunks
    cache_chunks=True,   # Cache frequently accessed chunks
    lazy_loading=True    # Load data on demand
)

# Preprocessing optimization
transform = StructuredDataTransform([
    ("cache_normalize", {"cache_stats": True}),  # Cache normalization stats
    ("vectorized_ops", {"use_numpy": True}),     # Use vectorized operations
    ("parallel_process", {"n_jobs": 4})          # Parallel processing
])
```

### Error Handling and Validation

```python
from jepa.data.validation import validate_structured_data

# Validate data before training
validation_results = validate_structured_data(
    dataset,
    checks=[
        "shape_consistency",
        "missing_values",
        "data_types",
        "temporal_ordering"
    ]
)

if not validation_results.is_valid:
    print("Data validation failed:")
    for error in validation_results.errors:
        print(f"  - {error}")
```

## Real-World Examples

### Financial Time Series

```python
# Stock price prediction
dataset = CSVDataset(
    csv_path="stock_prices.csv",
    data_columns=["open", "high", "low", "close", "volume"],
    temporal_offset=5,  # Predict 5 days ahead
    index_column="date",
    normalize=True
)
```

### IoT Sensor Data

```python
# Sensor anomaly detection
dataset = JSONDataset(
    json_path="sensor_logs.json",
    data_key="readings.sensors",
    temporal_offset=1,
    filter_outliers=True,
    resample_frequency="1min"  # Resample to 1-minute intervals
)
```

### Scientific Measurements

```python
# Laboratory experiment data
dataset = PickleDataset(
    pickle_path="experiment_results.pkl",
    data_t_key="measurements.trial_data",
    data_t1_key="measurements.outcomes",
    metadata_key="experimental_conditions"
)
```

For complete examples and advanced use cases, see the [Structured Data Examples](../examples/structured_data.md) section.
