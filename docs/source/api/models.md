# Models API

This section documents the model architectures and components.

## Base Model Classes

```{eval-rst}
.. automodule:: models.base
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: models.base.BaseModel
   :members:
   :special-members: __init__
```

Base functionality for all JEPA models:

```python
from jepa.models.base import BaseModel

class CustomModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Define your model architecture
```

## JEPA Core Architecture

```{eval-rst}
.. automodule:: models.jepa
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: models.jepa.JEPA
   :members:
   :special-members: __init__
```

The main JEPA model class:

```python
from jepa.models import JEPA, Encoder, Predictor

# Create encoder and predictor
encoder = Encoder(hidden_dim=256)
predictor = Predictor(hidden_dim=256)

# Create JEPA model
model = JEPA(encoder, predictor)

# Forward pass
pred, target = model(context_data, target_data)
loss = model.compute_loss(pred, target)
```

## Encoder Architectures

```{eval-rst}
.. automodule:: models.encoder
   :members:
   :undoc-members:
   :show-inheritance:
```

### Base Encoder

```{eval-rst}
.. autoclass:: models.encoder.BaseEncoder
   :members:
   :special-members: __init__
```

### Transformer Encoder

```{eval-rst}
.. autoclass:: models.encoder.TransformerEncoder
   :members:
   :special-members: __init__
```

Transformer-based encoder for sequential data:

```python
from jepa.models.encoder import TransformerEncoder

encoder = TransformerEncoder(
    vocab_size=30000,
    hidden_dim=768,
    num_layers=12,
    num_heads=12,
    max_sequence_length=512
)
```

### CNN Encoder

```{eval-rst}
.. autoclass:: models.encoder.CNNEncoder
   :members:
   :special-members: __init__
```

Convolutional encoder for spatial data:

```python
from jepa.models.encoder import CNNEncoder

encoder = CNNEncoder(
    input_channels=3,
    hidden_channels=[64, 128, 256],
    kernel_sizes=[3, 3, 3],
    output_dim=512
)
```

### Vision Transformer (ViT) Encoder

```{eval-rst}
.. autoclass:: models.encoder.ViTEncoder
   :members:
   :special-members: __init__
```

Vision Transformer for image processing:

```python
from jepa.models.encoder import ViTEncoder

encoder = ViTEncoder(
    image_size=224,
    patch_size=16,
    num_layers=12,
    hidden_dim=768,
    num_heads=12
)
```

### RNN Encoder

```{eval-rst}
.. autoclass:: models.encoder.RNNEncoder
   :members:
   :special-members: __init__
```

Recurrent encoder for sequential data:

```python
from jepa.models.encoder import RNNEncoder

encoder = RNNEncoder(
    input_dim=100,
    hidden_dim=256,
    num_layers=3,
    rnn_type="LSTM",  # LSTM, GRU, or RNN
    bidirectional=True
)
```

### MLP Encoder

```{eval-rst}
.. autoclass:: models.encoder.MLPEncoder
   :members:
   :special-members: __init__
```

Multi-layer perceptron encoder:

```python
from jepa.models.encoder import MLPEncoder

encoder = MLPEncoder(
    input_dim=784,
    hidden_dims=[512, 256, 128],
    output_dim=64,
    activation="relu",
    dropout=0.1
)
```

## Predictor Architectures

```{eval-rst}
.. automodule:: models.predictor
   :members:
   :undoc-members:
   :show-inheritance:
```

### Base Predictor

```{eval-rst}
.. autoclass:: models.predictor.BasePredictor
   :members:
   :special-members: __init__
```

### MLP Predictor

```{eval-rst}
.. autoclass:: models.predictor.MLPPredictor
   :members:
   :special-members: __init__
```

Simple multi-layer perceptron predictor:

```python
from jepa.models.predictor import MLPPredictor

predictor = MLPPredictor(
    input_dim=256,
    hidden_dims=[128, 64],
    output_dim=256,
    activation="gelu"
)
```

### Attention Predictor

```{eval-rst}
.. autoclass:: models.predictor.AttentionPredictor
   :members:
   :special-members: __init__
```

Attention-based predictor:

```python
from jepa.models.predictor import AttentionPredictor

predictor = AttentionPredictor(
    input_dim=256,
    num_heads=8,
    num_layers=4,
    feedforward_dim=1024
)
```

### Convolutional Predictor

```{eval-rst}
.. autoclass:: models.predictor.ConvPredictor
   :members:
   :special-members: __init__
```

Convolutional predictor for spatial predictions:

```python
from jepa.models.predictor import ConvPredictor

predictor = ConvPredictor(
    input_channels=256,
    hidden_channels=[128, 64],
    output_channels=256,
    kernel_size=3
)
```

## Specialized Model Variants

### Multimodal JEPA

```{eval-rst}
.. autoclass:: models.multimodal.MultimodalJEPA
   :members:
   :special-members: __init__
```

Handle multiple data modalities:

```python
from jepa.models.multimodal import MultimodalJEPA

model = MultimodalJEPA(
    vision_encoder=ViTEncoder(),
    text_encoder=TransformerEncoder(),
    fusion_dim=512,
    predictor=AttentionPredictor()
)
```

### Hierarchical JEPA

```{eval-rst}
.. autoclass:: models.hierarchical.HierarchicalJEPA
   :members:
   :special-members: __init__
```

Multi-scale hierarchical learning:

```python
from jepa.models.hierarchical import HierarchicalJEPA

model = HierarchicalJEPA(
    scales=[1, 2, 4],
    encoder_configs=[small_config, medium_config, large_config],
    predictor_config=predictor_config
)
```

### Temporal JEPA

```{eval-rst}
.. autoclass:: models.temporal.TemporalJEPA
   :members:
   :special-members: __init__
```

Specialized for time series data:

```python
from jepa.models.temporal import TemporalJEPA

model = TemporalJEPA(
    encoder=RNNEncoder(),
    predictor=AttentionPredictor(),
    temporal_window=100,
    prediction_horizon=10
)
```

## Model Factory Functions

### Create Model from Config

```{eval-rst}
.. autofunction:: models.factory.create_model
```

Create models from configuration:

```python
from jepa.models.factory import create_model

config = {
    'model_type': 'jepa',
    'encoder': {
        'type': 'transformer',
        'hidden_dim': 768,
        'num_layers': 12
    },
    'predictor': {
        'type': 'mlp',
        'hidden_dims': [512, 256]
    }
}

model = create_model(config)
```

### Create Encoder

```{eval-rst}
.. autofunction:: models.factory.create_encoder
```

Create encoders by type:

```python
from jepa.models.factory import create_encoder

encoder = create_encoder(
    encoder_type="cnn",
    input_channels=3,
    output_dim=512
)
```

### Create Predictor

```{eval-rst}
.. autofunction:: models.factory.create_predictor
```

Create predictors by type:

```python
from jepa.models.factory import create_predictor

predictor = create_predictor(
    predictor_type="attention",
    input_dim=512,
    output_dim=512
)
```

## Model Utilities

### Model Loading and Saving

```{eval-rst}
.. autofunction:: models.utils.save_model

.. autofunction:: models.utils.load_model

.. autofunction:: models.utils.load_checkpoint
```

Model persistence utilities:

```python
from jepa.models.utils import save_model, load_model

# Save model
save_model(model, "model.pth", include_config=True)

# Load model
model = load_model("model.pth")

# Load from checkpoint
model, optimizer, epoch = load_checkpoint("checkpoint.pth")
```

### Model Analysis

```{eval-rst}
.. autofunction:: models.utils.count_parameters

.. autofunction:: models.utils.analyze_model

.. autofunction:: models.utils.visualize_architecture
```

Model analysis tools:

```python
from jepa.models.utils import count_parameters, analyze_model

# Count parameters
total_params = count_parameters(model)
print(f"Total parameters: {total_params:,}")

# Analyze model structure
analysis = analyze_model(model, input_shape=(3, 224, 224))
print(analysis)
```

### Model Optimization

```{eval-rst}
.. autofunction:: models.utils.optimize_model

.. autofunction:: models.utils.quantize_model

.. autofunction:: models.utils.prune_model
```

Model optimization utilities:

```python
from jepa.models.utils import optimize_model, quantize_model

# Optimize for inference
optimized_model = optimize_model(model, optimization_level=2)

# Quantize model
quantized_model = quantize_model(model, method="dynamic")
```

## Custom Model Development

### Creating Custom Encoders

```python
from jepa.models.encoder import BaseEncoder
import torch.nn as nn

class CustomEncoder(BaseEncoder):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(output_dim=output_dim, **kwargs)
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

# Register custom encoder
from jepa.models.factory import register_encoder
register_encoder("custom", CustomEncoder)
```

### Creating Custom Predictors

```python
from jepa.models.predictor import BasePredictor

class CustomPredictor(BasePredictor):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim=input_dim, output_dim=output_dim, **kwargs)
        
        self.prediction_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, output_dim)
        )
    
    def forward(self, context_embedding):
        return self.prediction_head(context_embedding)

# Register custom predictor
from jepa.models.factory import register_predictor
register_predictor("custom", CustomPredictor)
```

### Creating Custom JEPA Variants

```python
from jepa.models.jepa import JEPA

class CustomJEPA(JEPA):
    def __init__(self, encoder, predictor, **kwargs):
        super().__init__(encoder, predictor, **kwargs)
        
        # Add custom components
        self.auxiliary_head = nn.Linear(encoder.output_dim, 10)
    
    def forward(self, context, target):
        # Standard JEPA forward pass
        pred, target_emb = super().forward(context, target)
        
        # Additional auxiliary prediction
        aux_pred = self.auxiliary_head(target_emb)
        
        return pred, target_emb, aux_pred
    
    def compute_loss(self, pred, target, aux_pred=None, aux_target=None):
        # Standard JEPA loss
        main_loss = super().compute_loss(pred, target)
        
        # Additional auxiliary loss
        if aux_pred is not None and aux_target is not None:
            aux_loss = F.cross_entropy(aux_pred, aux_target)
            return main_loss + 0.1 * aux_loss
        
        return main_loss
```

## Configuration

### Model Configuration

```{eval-rst}
.. autoclass:: config.config.ModelConfig
   :members:
   :special-members: __init__
```

Configure models through YAML:

```yaml
model:
  type: "jepa"
  encoder:
    type: "transformer"
    hidden_dim: 768
    num_layers: 12
    num_heads: 12
    dropout: 0.1
  predictor:
    type: "mlp"
    hidden_dims: [512, 256]
    activation: "gelu"
    dropout: 0.1
  loss_function: "contrastive"
  temperature: 0.1
```

## Examples

### Basic Model Usage

```python
from jepa.models import JEPA, TransformerEncoder, MLPPredictor

# Create components
encoder = TransformerEncoder(vocab_size=30000, hidden_dim=768)
predictor = MLPPredictor(input_dim=768, output_dim=768)

# Create JEPA model
model = JEPA(encoder, predictor)

# Training mode
model.train()
context, target = get_batch()
pred, target_emb = model(context, target)
loss = model.compute_loss(pred, target_emb)

# Inference mode
model.eval()
with torch.no_grad():
    embeddings = model.encode(data)
```

### Advanced Model Configuration

```python
from jepa.models.factory import create_model
from jepa.config import ModelConfig

config = ModelConfig(
    model_type="multimodal_jepa",
    encoders={
        "vision": {
            "type": "vit",
            "image_size": 224,
            "patch_size": 16,
            "hidden_dim": 768
        },
        "text": {
            "type": "transformer",
            "vocab_size": 30000,
            "hidden_dim": 768
        }
    },
    fusion_config={
        "fusion_dim": 512,
        "fusion_type": "cross_attention"
    },
    predictor={
        "type": "attention",
        "num_heads": 8,
        "num_layers": 4
    }
)

model = create_model(config)
```

For more examples and detailed usage, see the [Training Guide](../guides/training.md) and [Examples](../examples/index.md).
