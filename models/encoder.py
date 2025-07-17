"""
Need to update this to use any encoder we want, such as a transformer like GPT.
"""


import torch.nn as nn

from models.base import BaseModel

class SimpleTransformerEncoder(BaseModel):
    def __init__(self, hidden_dim):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4),
            num_layers=6
        )

    def forward(self, x):
        return self.encoder(x)
