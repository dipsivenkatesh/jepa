# JEPA Architecture & Design

The JEPA (Joint-Embedding Predictive Architecture) framework provides a flexible, modular approach to self-supervised learning that can work with any encoder and predictor you design.

## Core Philosophy

JEPA is built on the principle of **architectural flexibility**. Instead of forcing you into specific model designs, JEPA provides a framework that can accommodate:

- **Any encoder architecture** (CNNs, Transformers, MLPs, custom designs)
- **Any predictor design** (simple MLPs, complex networks, attention mechanisms)
- **Any data modality** (images, text, time series, audio, multimodal)

## Key Features

:::{admonition} Modular Design
:class: tip
Pass any encoder and predictor to the JEPA model - no architectural constraints
:::

:::{admonition} Flexible Architecture
:class: note
Works seamlessly with transformers, CNNs, MLPs, or your custom architectures
:::

:::{admonition} Easy to Extend
:class: important
Create custom encoders and predictors by simply inheriting from `nn.Module`
:::

## Basic Usage

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

## Custom Encoders and Predictors

The real power of JEPA comes from creating custom components tailored to your specific use case:

### Custom CNN Encoder

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
```

### Custom Predictor

```python
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

## Multi-Modal Flexibility

JEPA is completely agnostic to data type - it depends entirely on your encoder design:

### Computer Vision

```python
# For images (CNN encoder)
batch_size, channels, height, width = 4, 3, 224, 224
state_t = torch.randn(batch_size, channels, height, width)
state_t1 = torch.randn(batch_size, channels, height, width)

# CNN encoder processes spatial information
cnn_encoder = CustomCNNEncoder(input_channels=3, hidden_dim=512)
```

### Natural Language Processing

```python
# For sequences (Transformer encoder)
seq_length, batch_size, hidden_dim = 512, 32, 768
state_t = torch.randn(seq_length, batch_size, hidden_dim)
state_t1 = torch.randn(seq_length, batch_size, hidden_dim)

# Transformer encoder processes sequential information
transformer_encoder = TransformerEncoder(vocab_size=30000, hidden_dim=768)
```

### Time Series

```python
# For temporal data (RNN/CNN encoder)
time_steps, batch_size, features = 100, 64, 10
state_t = torch.randn(time_steps, batch_size, features)
state_t1 = torch.randn(time_steps, batch_size, features)

# Temporal encoder captures time dependencies
temporal_encoder = TemporalEncoder(input_dim=10, hidden_dim=256)
```

## Architecture Components

### BaseModel

Provides common functionality for all models:

- **Model persistence**: Save and load trained models
- **Configuration management**: Handle model hyperparameters
- **Device management**: Automatic CPU/GPU handling
- **Utility methods**: Common operations across models

### JEPA Core

The main model class that orchestrates training:

- **Encoder-Predictor coordination**: Manages the interaction between components
- **Loss computation**: Computes contrastive or reconstruction losses
- **Forward pass handling**: Manages data flow through the architecture
- **Gradient management**: Handles backpropagation and optimization

### Provided Implementations

JEPA comes with several ready-to-use components:

**Encoders:**
- `Encoder`: Basic transformer encoder for sequential data
- `CNNEncoder`: Convolutional encoder for spatial data
- `MLPEncoder`: Simple feedforward encoder

**Predictors:**
- `Predictor`: Simple MLP predictor
- `AttentionPredictor`: Attention-based predictor
- `RecurrentPredictor`: RNN-based predictor

## Design Patterns

### Encoder Design

Your encoder should:

1. **Accept raw input data** in your target modality
2. **Output fixed-size embeddings** that capture meaningful representations
3. **Be differentiable** throughout (standard PyTorch requirement)
4. **Handle batching** appropriately for your data type

```python
class MyEncoder(nn.Module):
    def __init__(self, input_spec, hidden_dim):
        super().__init__()
        # Define your architecture here
        
    def forward(self, x):
        # x: input data in your format
        # returns: tensor of shape (batch_size, hidden_dim)
        return embeddings
```

### Predictor Design

Your predictor should:

1. **Accept context embeddings** from the encoder
2. **Predict target embeddings** in the same space
3. **Learn meaningful transformations** that capture temporal/spatial relationships
4. **Output embeddings** of the same dimensionality as the encoder

```python
class MyPredictor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Define prediction architecture
        
    def forward(self, context_embedding):
        # context_embedding: encoder output for context data
        # returns: predicted embedding for target data
        return predicted_embedding
```

## Advanced Usage Patterns

### Multi-Scale Encoders

```python
class MultiScaleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.local_encoder = LocalCNNEncoder()
        self.global_encoder = GlobalTransformerEncoder()
        self.fusion = FusionLayer()
        
    def forward(self, x):
        local_features = self.local_encoder(x)
        global_features = self.global_encoder(x)
        return self.fusion(local_features, global_features)
```

### Hierarchical Predictors

```python
class HierarchicalPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.coarse_predictor = CoarsePredictor()
        self.fine_predictor = FinePredictor()
        
    def forward(self, context):
        coarse_pred = self.coarse_predictor(context)
        fine_pred = self.fine_predictor(context, coarse_pred)
        return fine_pred
```

### Cross-Modal Encoders

```python
class CrossModalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = VisionEncoder()
        self.text_encoder = TextEncoder()
        self.cross_attention = CrossAttention()
        
    def forward(self, vision_input, text_input):
        v_features = self.vision_encoder(vision_input)
        t_features = self.text_encoder(text_input)
        return self.cross_attention(v_features, t_features)
```

## Best Practices

### Model Design

1. **Start simple**: Begin with basic architectures and gradually add complexity
2. **Match architecture to data**: Use CNNs for spatial data, RNNs for sequential data
3. **Consider computational constraints**: Balance model size with available resources
4. **Use pretrained components**: Leverage existing pretrained encoders when possible

### Training Strategies

1. **Progressive training**: Start with smaller models and scale up
2. **Curriculum learning**: Begin with easier examples and increase difficulty
3. **Regularization**: Use dropout, weight decay, and other regularization techniques
4. **Monitoring**: Track both encoder and predictor performance separately

### Performance Optimization

1. **Batch size tuning**: Find the optimal batch size for your hardware
2. **Mixed precision**: Use automatic mixed precision for faster training
3. **Gradient accumulation**: Simulate larger batch sizes when memory constrained
4. **Efficient data loading**: Optimize your data pipeline for maximum throughput

## Examples and Use Cases

For complete examples showing these concepts in action, see:

- [Vision Examples](../examples/vision.md) - CNN encoders for image processing
- [NLP Examples](../examples/nlp.md) - Transformer encoders for text
- [Time Series Examples](../examples/timeseries.md) - Temporal encoders for sequential data
- [Multimodal Examples](../examples/multimodal.md) - Cross-modal architectures

The flexibility of JEPA means you can adapt it to virtually any domain by designing appropriate encoders and predictors for your specific use case.
