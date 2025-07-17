"""
Example demonstrating how to use the reusable JEPA model with different encoders and predictors.
"""

import torch
import torch.nn as nn
from jepa.models.jepa import JEPA
from jepa.models.encoder import Encoder
from jepa.models.predictor import Predictor


class CNNEncoder(nn.Module):
    """Example custom CNN encoder that can be used with JEPA."""
    
    def __init__(self, input_channels=3, hidden_dim=256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, hidden_dim)
        )
    
    def forward(self, x):
        return self.conv_layers(x)


class MLPPredictor(nn.Module):
    """Example custom MLP predictor that can be used with JEPA."""
    
    def __init__(self, hidden_dim=256, num_layers=3):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def example_usage():
    """Demonstrate different ways to use the JEPA model."""
    
    hidden_dim = 256
    
    # Example 1: Using the provided transformer encoder and predictor
    transformer_encoder = Encoder(hidden_dim)
    predictor = Predictor(hidden_dim)
    jepa_transformer = JEPA(transformer_encoder, predictor)
    
    # Example 2: Using custom CNN encoder with MLP predictor
    cnn_encoder = CNNEncoder(input_channels=3, hidden_dim=hidden_dim)
    mlp_predictor = MLPPredictor(hidden_dim)
    jepa_cnn = JEPA(cnn_encoder, mlp_predictor)
    
    # Example 3: Mix and match - CNN encoder with predictor
    jepa_mixed = JEPA(cnn_encoder, predictor)
    
    print("JEPA models created successfully!")
    print(f"Transformer JEPA: {jepa_transformer}")
    print(f"CNN JEPA: {jepa_cnn}")
    print(f"Mixed JEPA: {jepa_mixed}")
    
    # Example forward pass (you would replace this with actual data)
    # For transformer (expects sequence data)
    seq_length, batch_size = 10, 4
    dummy_seq_t = torch.randn(seq_length, batch_size, hidden_dim)
    dummy_seq_t1 = torch.randn(seq_length, batch_size, hidden_dim)
    
    pred_transformer, target_transformer = jepa_transformer(dummy_seq_t, dummy_seq_t1)
    loss_transformer = jepa_transformer.loss(pred_transformer, target_transformer)
    print(f"Transformer JEPA loss: {loss_transformer.item():.4f}")
    
    # For CNN (expects image data)
    batch_size, channels, height, width = 4, 3, 32, 32
    dummy_img_t = torch.randn(batch_size, channels, height, width)
    dummy_img_t1 = torch.randn(batch_size, channels, height, width)
    
    pred_cnn, target_cnn = jepa_cnn(dummy_img_t, dummy_img_t1)
    loss_cnn = jepa_cnn.loss(pred_cnn, target_cnn)
    print(f"CNN JEPA loss: {loss_cnn.item():.4f}")


if __name__ == "__main__":
    example_usage()
