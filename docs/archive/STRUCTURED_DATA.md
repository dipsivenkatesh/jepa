# Structured Data Formats for JEPA

The JEPA data module supports multiple structured data formats, making it easy to work with data stored in various file formats. This includes JSON, CSV, Pickle, and other formats with flexible data organization.

## New Dataset Classes

### JSONDataset
Load data from JSON files with flexible key-based access.

### CSVDataset  
Load data from CSV files with column-based organization.

### PickleDataset
Load data from Python pickle files.

All these classes support two main data organization patterns:
1. **Single Array Format**: One data array with temporal offsets
2. **Separate Arrays Format**: Separate arrays for data_t and data_t1

## Usage Examples

### JSONDataset - Single Array Format
```python
from jepa.data import JSONDataset

# JSON: {"data": [[1,2,3], [1.1,2.1,3.1], [1.2,2.2,3.2], ...]}
dataset = JSONDataset(
    json_path="data.json",
    data_key="data",
    time_offset=1  # Use consecutive samples as (t, t+1) pairs
)
```

### JSONDataset - Separate Arrays Format  
```python
# JSON: {"data_t": [...], "data_t1": [...]}
dataset = JSONDataset(
    json_path="data.json",
    data_t_key="data_t",
    data_t1_key="data_t1"
)
```

### CSVDataset Examples
```python
from jepa.data import CSVDataset

# Single columns format
dataset = CSVDataset(
    csv_path="data.csv", 
    data_columns=["feature1", "feature2", "feature3"],
    time_offset=1
)

# Separate columns format
dataset = CSVDataset(
    csv_path="data.csv",
    data_t_columns=["t_feat1", "t_feat2"], 
    data_t1_columns=["t1_feat1", "t1_feat2"]
)

# Auto-detect numeric columns
dataset = CSVDataset(csv_path="data.csv", time_offset=1)
```

### PickleDataset Examples
```python
from jepa.data import PickleDataset

# Dictionary format
dataset = PickleDataset(
    pickle_path="data.pkl",
    data_key="timeseries",
    time_offset=1
)

# Separate arrays format
dataset = PickleDataset(
    pickle_path="data.pkl", 
    data_t_key="data_t",
    data_t1_key="data_t1"
)
```

### Factory Function
```python
from jepa.data import create_dataset

# Creates appropriate dataset based on type
dataset = create_dataset(
    data_type="json",  # or "csv", "pickle"
    data_path="data.json",
    data_key="features",
    time_offset=1
)
```

## Supported Formats

### 1. JSON Dataset (`JSONDataset`)

Works with JSON files in multiple formats:

#### List Format
```json
{
  "data_t": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
  "data_t1": [[2, 3, 4], [5, 6, 7], [8, 9, 10]]
}
```

#### Dict Format
```json
{
  "samples": [
    {"data": [1, 2, 3], "target": [2, 3, 4], "id": 0},
    {"data": [4, 5, 6], "target": [5, 6, 7], "id": 1}
  ]
}
```

#### Nested Format
```json
{
  "dataset": {
    "train": {
      "features": [[1, 2, 3], [4, 5, 6]],
      "targets": [[2, 3, 4], [5, 6, 7]]
    }
  }
}
```

**Usage:**
```python
from data import JSONDataset

# List format
dataset = JSONDataset(
    data_path="data.json",
    data_t_key="data_t",
    data_t1_key="data_t1",
    data_format="list"
)

# Dict format
dataset = JSONDataset(
    data_path="data.json",
    data_t_key="data",
    data_t1_key="target", 
    data_format="dict"
)

# Nested format
dataset = JSONDataset(
    data_path="data.json",
    data_t_key="dataset.train.features",
    data_t1_key="dataset.train.targets",
    data_format="nested"
)

# Temporal offset (no separate data_t1)
dataset = JSONDataset(
    data_path="timeseries.json",
    data_t_key="time_series",
    temporal_offset=1,
    data_format="list"
)
```

### 2. CSV Dataset (`CSVDataset`)

Works with CSV files containing tabular data:

**Example CSV:**
```csv
state_t_1,state_t_2,state_t_3,state_t1_1,state_t1_2,state_t1_3
1,2,3,2,3,4
4,5,6,5,6,7
7,8,9,8,9,10
```

**Usage:**
```python
from data import CSVDataset

# With separate state_t and state_t1 columns
dataset = CSVDataset(
    data_path="data.csv",
    data_t_columns=["state_t_1", "state_t_2", "state_t_3"],
    data_t1_columns=["state_t1_1", "state_t1_2", "state_t1_3"]
)

# Time series with temporal offset
dataset = CSVDataset(
    data_path="timeseries.csv",
    data_t_columns=["value1", "value2", "value3"],
    temporal_offset=1
)

# Using column indices
dataset = CSVDataset(
    data_path="data.csv",
    data_t_columns=["0", "1", "2"],  # Column indices
    data_t1_columns=["3", "4", "5"]
)
```

### 3. Pickle Dataset (`PickleDataset`)

Works with Python pickle files containing complex data structures:

**Usage:**
```python
from data import PickleDataset
import pickle
import numpy as np

# Create sample pickle data
data = {
    "data_t": np.array([[1, 2, 3], [4, 5, 6]]),
    "data_t1": np.array([[2, 3, 4], [5, 6, 7]]),
    "metadata": {"source": "experiment_1"}
}
with open("data.pkl", "wb") as f:
    pickle.dump(data, f)

# Load with PickleDataset
dataset = PickleDataset(
    data_path="data.pkl",
    data_t_key="data_t",
    data_t1_key="data_t1"
)

# Direct array with temporal offset
array_data = np.random.randn(100, 10)  # 100 timesteps, 10 features
with open("timeseries.pkl", "wb") as f:
    pickle.dump(array_data, f)

dataset = PickleDataset(
    data_path="timeseries.pkl",
    temporal_offset=1
)
```

## Factory Function

Use the factory function for easy dataset creation:

```python
from data import create_dataset

# JSON dataset
dataset = create_dataset(
    data_type="json",
    data_path="data.json",
    data_t_key="features",
    data_format="list"
)

# CSV dataset
dataset = create_dataset(
    data_type="csv", 
    data_path="data.csv",
    data_t_columns=["col1", "col2"],
    temporal_offset=1
)

# Pickle dataset
dataset = create_dataset(
    data_type="pickle",
    data_path="data.pkl",
    data_t_key="data"
)
```

## Key Features

### Flexible Data Loading
- Support for multiple file formats (JSON, CSV, Pickle)
- Flexible key/column specification
- Nested data structure navigation

### Temporal Relationships
- Direct pairs: `data_t` and `data_t1` specified separately
- Temporal offset: `data_t1 = data_t[t + offset]`
- Customizable temporal gaps

### Transform Integration
- All datasets work with existing JEPA transforms
- Easy integration with data augmentation pipeline

### Error Handling
- Comprehensive error checking and validation
- Clear error messages for debugging
- Graceful fallbacks where possible

## Examples

See `examples/structured_data_example.py` for complete usage examples of all structured data formats.

## Data Preparation Tips

### JSON Files
1. Ensure consistent array shapes across samples
2. Use meaningful key names for clarity
3. Consider nested structures for complex datasets

### CSV Files
1. Include header row for column names
2. Ensure numeric data is properly formatted
3. Consider using pandas for complex preprocessing

### Pickle Files
1. Use consistent data structures
2. Include metadata for dataset documentation
3. Consider version compatibility for long-term storage

## Integration with Training

All structured datasets work seamlessly with the JEPA training pipeline:

```python
from data import create_dataset, DataLoader
from jepa.trainer import create_trainer
from models import JEPA, Encoder, Predictor

# Create dataset
dataset = create_dataset(
    data_type="json",
    data_path="my_data.json",
    data_t_key="sequences",
    temporal_offset=1,
    data_format="list"
)

# Create data loader
loader = DataLoader.create_jepa_dataloader(dataset, batch_size=32)

# Train model
model = JEPA(Encoder(128), Predictor(128))
trainer = create_trainer(model)
history = trainer.train(loader, num_epochs=100)
```
