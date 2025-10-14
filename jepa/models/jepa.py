import torch
import torch.nn as nn
from typing import Tuple

from .base import BaseModel


class JEPA(BaseModel):
    def __init__(self, encoder: nn.Module, predictor: nn.Module) -> None:
        """Initialize JEPA model with any encoder and predictor."""
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

    def forward(self, state_t: torch.Tensor, state_t1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_t = self.encoder(state_t)
        z_t1 = self.encoder(state_t1).detach()
        pred = self.predictor(z_t)
        return pred, z_t1


class JEPAAction(JEPA):
    """
    JEPA variant that conditions the next-state prediction on an action.

    Uses the same state encoder for ``state_t`` and ``state_t1`` as the base JEPA,
    plus an ``action_encoder`` for ``action_t``. The predictor receives the
    concatenation of ``[z_t, a_t]`` and outputs a prediction that matches the
    target next-state embedding ``z_{t+1}``.
    """

    def __init__(
        self,
        state_encoder: nn.Module,
        action_encoder: nn.Module,
        predictor: nn.Module,
    ) -> None:
        super().__init__(encoder=state_encoder, predictor=predictor)
        self.action_encoder = action_encoder

    def forward(
        self,
        state_t: torch.Tensor,
        action_t: torch.Tensor,
        state_t1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode states
        z_t = self.encoder(state_t)
        z_t1 = self.encoder(state_t1).detach()

        # Encode action
        a_t = self.action_encoder(action_t)

        # Flatten sequence dims if needed to form feature vectors
        if z_t.ndim > 2:
            z_t = z_t.reshape(z_t.size(0), -1)
        if a_t.ndim > 2:
            a_t = a_t.reshape(a_t.size(0), -1)

        fused = torch.cat([z_t, a_t], dim=-1)
        pred = self.predictor(fused)
        return pred, z_t1
