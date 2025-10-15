import torch
import torch.nn as nn
from typing import Tuple, Any
from collections.abc import Mapping

try:  # Optional dependency for HuggingFace compatibility
    from transformers import PreTrainedModel
except ImportError:  # pragma: no cover - transformers is optional
    PreTrainedModel = None  # type: ignore

from .base import BaseModel


class JEPA(BaseModel):
    def __init__(self, encoder: nn.Module, predictor: nn.Module) -> None:
        """Initialize JEPA model with any encoder and predictor."""
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self._pool_strategy = "cls"
        self._is_hf_encoder = self._is_hf_module(encoder)

    def _normalize_encoder_output(self, output: Any) -> torch.Tensor:
        """Convert encoder outputs into tensors compatible with JEPA predictors."""
        if torch.is_tensor(output):
            return output

        # HuggingFace models typically return ``ModelOutput`` objects that behave like mappings
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            return output.last_hidden_state
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output

        if isinstance(output, Mapping):
            for key in ("last_hidden_state", "pooler_output", "hidden_states", "logits"):
                value = output.get(key)
                if value is None:
                    continue
                if key == "hidden_states" and isinstance(value, (list, tuple)):
                    return value[-1]
                return value
            first_value = next(iter(output.values()))
            return self._normalize_encoder_output(first_value)

        if isinstance(output, (list, tuple)) and output:
            return self._normalize_encoder_output(output[0])

        raise TypeError(
            "Unsupported encoder output type. Ensure the encoder returns tensors or a HuggingFace "
            "ModelOutput containing `last_hidden_state`, `pooler_output`, or `logits`."
        )

    def _is_hf_module(self, module: nn.Module) -> bool:
        return bool(PreTrainedModel and isinstance(module, PreTrainedModel))

    def _encode_state(self, *inputs: Any, **kwargs: Any) -> torch.Tensor:
        encoded = self.encoder(*inputs, **kwargs)
        normalized = self._normalize_encoder_output(encoded)
        return self._prepare_representation(normalized, self._is_hf_encoder)

    def _prepare_representation(self, embedding: torch.Tensor, is_hf: bool) -> torch.Tensor:
        """Shape encoder outputs for downstream predictors."""
        if embedding.ndim == 3 and is_hf:
            # Default to CLS token pooling for Transformer encoders
            if self._pool_strategy == "cls":
                return embedding[:, 0]
            if self._pool_strategy == "mean":
                return embedding.mean(dim=1)
        if embedding.ndim > 2:
            return embedding.reshape(embedding.size(0), -1)
        return embedding

    def forward(self, state_t: torch.Tensor, state_t1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_t = self._encode_state(state_t)
        z_t1 = self._encode_state(state_t1).detach()
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
        self._is_hf_action_encoder = self._is_hf_module(action_encoder)

    def forward(
        self,
        state_t: torch.Tensor,
        action_t: torch.Tensor,
        state_t1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode states
        z_t = self._encode_state(state_t)
        z_t1 = self._encode_state(state_t1).detach()

        # Encode action
        a_t_raw = self.action_encoder(action_t)
        a_t = self._normalize_encoder_output(a_t_raw)
        a_t = self._prepare_representation(a_t, self._is_hf_action_encoder)

        # Flatten sequence dims if needed to form feature vectors
        if z_t.ndim > 2:
            z_t = z_t.reshape(z_t.size(0), -1)

        fused = torch.cat([z_t, a_t], dim=-1)
        pred = self.predictor(fused)
        return pred, z_t1
