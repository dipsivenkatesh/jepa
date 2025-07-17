import torch.nn as nn
from .base import BaseModel

class JEPA(BaseModel):
    def __init__(self, encoder, predictor):
        """
        Initialize JEPA model with any encoder and predictor.
        
        Args:
            encoder: Any encoder model (e.g., transformer, CNN, etc.)
            predictor: Any predictor model for latent space prediction
        """
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.loss_fn = nn.MSELoss()

    def forward(self, state_t, state_t1):
        z_t = self.encoder(state_t)
        z_t1 = self.encoder(state_t1).detach()
        pred = self.predictor(z_t)
        return pred, z_t1

    def loss(self, prediction, target):
        return self.loss_fn(prediction, target)
