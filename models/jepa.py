import torch.nn as nn
from .base import BaseModel
from .encoder import SimpleTransformerEncoder
from .predictor import LatentPredictor

class JEPA(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = SimpleTransformerEncoder(config.hidden_dim)
        self.predictor = LatentPredictor(config.hidden_dim)
        self.loss_fn = nn.MSELoss()

    def forward(self, state_t, state_t1):
        z_t = self.encoder(state_t)
        z_t1 = self.encoder(state_t1).detach()
        pred = self.predictor(z_t)
        return pred, z_t1

    def loss(self, prediction, target):
        return self.loss_fn(prediction, target)
