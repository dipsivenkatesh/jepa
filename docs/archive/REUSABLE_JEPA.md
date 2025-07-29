# Reusable JEPA Model

This implementation provides a flexible JEPA (Joint-Embedding Predictive Architecture) model that can work with any encoder and predictor you design.

## Key Features

- **Modular Design**: Pass any encoder and predictor to the JEPA model
- **Flexible Architecture**: Works with transformers, CNNs, MLPs, or custom architectures
- **Easy to Extend**: Create custom encoders and predictors by inheriting from `nn.Module`

## Usage

### Basic Usage

```python
from models import JEPA, Encoder, Predictor

# Create encoder and predictor
encoder = Encoder(hidden_dim=256)
predictor = Predictor(hidden_dim=256)

# Create JEPA model
jepa_model = JEPA(encoder, predictor)

# Forward pass
pred, target = jepa_model(state_t, state_t1)
loss = jepa_model.loss(pred, target)
```

### Custom Encoders and Predictors

You can create any custom encoder or predictor:

```python
import torch.nn as nn

class CustomCNNEncoder(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, hidden_dim)
        )
    
    def forward(self, x):
        return self.conv_net(x)

class CustomPredictor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.predictor = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        return self.predictor(x)

# Use with JEPA
custom_encoder = CustomCNNEncoder(input_channels=3, hidden_dim=256)
custom_predictor = CustomPredictor(hidden_dim=256)
jepa_model = JEPA(custom_encoder, custom_predictor)
```

### Working with Different Data Types

The JEPA model is agnostic to the data type - it depends on your encoder:

```python
# For images (CNN encoder)
batch_size, channels, height, width = 4, 3, 32, 32
state_t = torch.randn(batch_size, channels, height, width)
state_t1 = torch.randn(batch_size, channels, height, width)

# For sequences (Transformer encoder)
seq_length, batch_size, hidden_dim = 10, 4, 256
state_t = torch.randn(seq_length, batch_size, hidden_dim)
state_t1 = torch.randn(seq_length, batch_size, hidden_dim)

# For any other data type - just design an appropriate encoder
```

## Components

### BaseModel
- Provides common functionality for saving/loading models
- Base class for all models

### JEPA
- Main model class that combines encoder and predictor
- Handles forward pass and loss computation
- Accepts any encoder/predictor pair

### Provided Implementations
- `Encoder`: Basic transformer encoder
- `Predictor`: Simple MLP predictor

## Examples

See `examples/usage_example.py` for complete examples showing:
- Using provided components
- Creating custom encoders and predictors
- Mixed usage scenarios
