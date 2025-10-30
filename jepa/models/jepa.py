import torch
import torch.nn as nn
from typing import Optional, Tuple, Any
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

class JEPAInterleavedTemporal(JEPA):
    """
    Interleaved (z, u) temporal JEPA that reuses JEPA's normalization & pooling.

    Expects:
      - encoder: per-step state encoder (same as JEPA.encoder)
      - predictor: temporal causal module mapping (B, L, D) -> (B, L, D)
      - action_embedder: optional, maps (B, T) -> (B, T, D_a)
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        action_embedder: Optional[nn.Module] = None,
        d_model: Optional[int] = None,
        use_type_embeddings: bool = True,
    ) -> None:
        super().__init__(encoder=encoder, predictor=predictor)
        self.action_embedder = action_embedder
        self._is_hf_action = self._is_hf_module(action_embedder) if action_embedder else False

        self.d_model = d_model  # may be None; we’ll infer lazily
        self._enc_proj: Optional[nn.Linear] = None
        self._act_proj: Optional[nn.Linear] = None

        self.use_type_embeddings = use_type_embeddings
        self.state_type: Optional[nn.Parameter] = None
        self.action_type: Optional[nn.Parameter] = None

        self.pred_head: Optional[nn.Linear] = None

    # ---- utilities ----

    def _maybe_init_dims(self, z_like: torch.Tensor) -> None:
        if self.d_model is None:
            self.d_model = int(z_like.shape[-1])
        if self.pred_head is None:
            self.pred_head = nn.Linear(self.d_model, self.d_model)
        if self.use_type_embeddings and self.state_type is None:
            self.state_type = nn.Parameter(torch.zeros(1, 1, self.d_model))
            self.action_type = nn.Parameter(torch.zeros(1, 1, self.d_model))

    def _ensure_proj(self, in_dim: int, kind: str) -> Optional[nn.Linear]:
        if self.d_model is None or in_dim == self.d_model:
            return None
        proj = nn.Linear(in_dim, self.d_model)
        if kind == "enc":
            self._enc_proj = proj
        else:
            self._act_proj = proj
        return proj

    def _encode_steps(self, input_ids: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        """Encode a batch of T steps: (B,T,S)->(B,T,D). Reuses JEPA’s normalization & pooling."""
        B, T, S = input_ids.shape
        flat_ids = input_ids.reshape(B * T, S)
        flat_att = attention.reshape(B * T, S)

        out = self.encoder(input_ids=flat_ids, attention_mask=flat_att)
        feats = self._normalize_encoder_output(out)  # (B*T,S,D?) or (B*T,D?)
        # Pool/flatten with JEPA logic:
        z_flat = self._prepare_representation(feats, self._is_hf_encoder)  # (B*T,D_enc)

        self._maybe_init_dims(z_flat)
        if self._enc_proj is None:
            self._ensure_proj(z_flat.shape[-1], "enc")
        if self._enc_proj is not None:
            z_flat = self._enc_proj(z_flat)
        return z_flat.view(B, T, self.d_model)  # (B,T,D)

    def _encode_next(
        self,
        next_ids: torch.Tensor,          # (B,S_next) or (B,T,S_next)
        next_att: torch.Tensor,          # same shape
        T: int,
    ) -> torch.Tensor:
        if next_ids.ndim == 2:
            next_ids = next_ids.unsqueeze(1).expand(-1, T, -1)
            next_att = next_att.unsqueeze(1).expand(-1, T, -1)
        z_next = self._encode_steps(next_ids, next_att)  # (B,T,D)
        return z_next.detach()

    def _interleave(self, z: torch.Tensor, u: Optional[torch.Tensor]) -> torch.Tensor:
        """[z0,u0,z1,u1,...] if u provided, else [z0,z1,...]."""
        if u is None:
            return z
        B, T, D = z.shape
        seq = torch.empty(B, 2 * T, D, device=z.device, dtype=z.dtype)
        seq[:, 0::2, :] = z
        seq[:, 1::2, :] = u
        return seq

    # ---- forward ----

    def forward(
        self,
        history_input_ids: torch.Tensor,    # (B,T,S)
        history_attention: torch.Tensor,    # (B,T,S)
        next_input_ids: torch.Tensor,       # (B,S_next) or (B,T,S_next)
        next_attention: torch.Tensor,       # same
        action_ids: Optional[torch.Tensor] = None,  # (B,T)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = history_input_ids.shape

        # encode states (each step is state⊕action tokens from your dataloader)
        z_hist = self._encode_steps(history_input_ids, history_attention)  # (B,T,D)

        # optional separate action token stream
        u = None
        if self.action_embedder is not None and action_ids is not None:
            u_raw = self.action_embedder(action_ids)  # expect (B,T,D_a)
            if not torch.is_tensor(u_raw):
                u_raw = self._normalize_encoder_output(u_raw)
            if u_raw.ndim != 3:
                raise ValueError("action_embedder must return (B,T,D_a)")
            self._maybe_init_dims(z_hist.reshape(B * T, -1))
            if self._act_proj is None:
                self._ensure_proj(u_raw.shape[-1], "act")
            u = self._act_proj(u_raw) if self._act_proj is not None else u_raw  # (B,T,D)

        # interleave and add type embeddings (optional)
        seq = self._interleave(z_hist, u)  # (B,L,D), L=T or 2T
        if self.use_type_embeddings and self.state_type is not None:
            if u is None:
                seq = seq + self.state_type
            else:
                seq = seq + torch.where(
                    (torch.arange(seq.size(1), device=seq.device) % 2 == 0).view(1, -1, 1),
                    self.state_type, self.action_type
                )

        # temporal predictor returns per-position hidden states
        h = self.predictor(seq)  # (B,L,D)

        # gather hidden states at state positions
        h_states = h if u is None else h[:, 0::2, :]  # (B,T,D)

        # predict z_{t+1} from h_t
        if self.pred_head is None:
            self._maybe_init_dims(h_states.reshape(B * T, -1))
        z_pred_next = self.pred_head(h_states)  # (B,T,D)

        # encode targets for s_{t+1}
        z_target_next = self._encode_next(next_input_ids, next_attention, T)  # (B,T,D)

        return z_pred_next, z_target_next
