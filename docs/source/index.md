# JEPA Framework Documentation

Welcome to the **JEPA (Joint-Embedding Predictive Architecture)** framework documentation. JEPA is a powerful self-supervised learning framework that learns representations by predicting parts of the input from other parts.

## 🚀 Quick Start

```python
from jepa import JEPATrainer
from jepa.config import load_config

# Load configuration and start training
config = load_config("config/default_config.yaml")
trainer = JEPATrainer(config)
trainer.train()
```

Or use the CLI:

```bash
jepa-train --config config/default_config.yaml
```

### Action-Conditioned JEPA

If actions influence transitions, use `JEPAAction` to condition predictions on actions:

```python
from jepa import JEPAAction
import torch.nn as nn

state_encoder = ...  # outputs state_dim
action_encoder = ... # outputs action_dim
predictor = nn.Sequential(nn.Linear(state_dim + action_dim, state_dim))
model = JEPAAction(state_encoder, action_encoder, predictor)
```

## 📖 Documentation Sections

```{toctree}
:maxdepth: 2
:caption: User Guide

guides/installation.md
guides/quickstart.md
guides/configuration.md
guides/training.md
guides/data.md
```

```{toctree}
:maxdepth: 2
:caption: Core Concepts

concepts/architecture.md
concepts/self_supervised_learning.md
concepts/structured_data.md
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/models.md
api/training.md
api/data.md
api/logging.md
api/config.md
api/cli.md
```

```{toctree}
:maxdepth: 2
:caption: Examples

examples/index.md
```

## 🎯 Key Features

**🔧 Modular Design**
- Flexible encoder-predictor architecture
- Support for any PyTorch model as encoder/predictor
- Easy to extend and customize

**🌍 Multi-Modal Support**
- Computer Vision (images, videos)
- Natural Language Processing (text)
- Time Series (sequential data)
- Audio processing
- Multimodal learning

**⚡ High Performance**
- Mixed precision training
- Distributed training support
- Triton kernel optimization
- Memory-efficient implementations

**📊 Comprehensive Logging**
- Weights & Biases integration
- TensorBoard support
- Console logging
- Multi-backend logging system

**🎛️ Production Ready**
- CLI interface for easy deployment
- Flexible configuration system
- Comprehensive testing
- Docker support

## 🏗️ Architecture Overview

JEPA follows a simple yet powerful architecture:

```
Input Data → [Context/Target Split] → Encoder → Joint Embedding Space
                                         ↓
Target Embedding ← Predictor ← Context Embedding
```

The model learns by:
1. **Splitting** input into context and target regions
2. **Encoding** both context and target separately
3. **Predicting** target embeddings from context embeddings
4. **Learning** representations that capture meaningful relationships

## 🎨 Use Cases

**Computer Vision**
- Image classification pretraining
- Object detection backbone
- Medical image analysis
- Satellite imagery processing

**Natural Language Processing**
- Language model pretraining
- Document understanding
- Code representation learning
- Cross-lingual embeddings

**Time Series**
- Forecasting model pretraining
- Anomaly detection
- Financial data analysis
- IoT sensor data processing

**Multimodal Learning**
- Vision-language models
- Audio-visual learning
- Cross-modal retrieval
- Multimodal reasoning

## 🔗 Quick Links

- **[Installation Guide](guides/installation.md)** - Get started in 5 minutes
- **[Quick Start](guides/quickstart.md)** - Your first JEPA model
- **[API Reference](api/models.md)** - Complete API documentation
- **[Examples](examples/index.md)** - Real-world use cases
- **[GitHub Repository](https://github.com/dipsivenkatesh/jepa)** - Source code and issues

## 📄 Citation

If you use JEPA in your research, please cite:

```bibtex
@article{jepa2024,
  title={Joint-Embedding Predictive Architecture for Self-Supervised Learning},
  author={Dilip Venkatesh},
  year={2025}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/dipsivenkatesh/jepa/blob/main/LICENSE) file for details.

---

*Built with ❤️ by the [Dilip Venkatesh](https://dipsivenkatesh.github.io/)*
