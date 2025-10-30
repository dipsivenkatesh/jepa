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

class JEPAInterleaved(JEPA):
    """
    JEPA variant for interleaved (state, action, state, action, ...) prefixes.

    This class:
      - Encodes each *state* event with `state_encoder` (shared with base JEPA).
      - Encodes each *action* event with `action_encoder`.
      - Normalizes both into a shared model space and interleaves them along time.
      - Feeds the (B, E, D) sequence into `predictor`, which should output a single
        next-state embedding prediction per example (B, D) for the *next* state
        specified by `labels`.

    Expected `forward` inputs (aligned with your collate function):
      inputs:           LongTensor (B, E_max, S_max)
      inputs_attention: BoolTensor (B, E_max, S_max)
      event_type:       LongTensor (B, E_max)  with 0 = state, 1 = action
      labels:           LongTensor (B, N_max)
      labels_attention: BoolTensor (B, N_max)

    The `predictor` you provide should accept at least:
        predictor(seq_emb: Tensor) -> Tensor
    and may optionally accept keyword arguments:
        event_type=..., event_mask=...
    If provided, they will be forwarded.
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

    # ---------- helpers ----------

    def _encode_state_events(
        self,
        tokens: torch.Tensor,           # (Ns, S_max)
        attention: torch.Tensor,        # (Ns, S_max)
    ) -> torch.Tensor:
        """
        Encode a packed set of STATE events with the state encoder.
        Returns pooled embeddings (Ns, D).
        """
        if tokens.numel() == 0:
            # no state rows in this batch slice
            return tokens.new_zeros((0, self._infer_embed_dim()))
        z = self._encode_state(input_ids=tokens, attention_mask=attention)  # (Ns, D)
        if z.ndim > 2:
            # ensure pooled representation per event
            z = self._prepare_representation(z, self._is_hf_encoder)
        return z

    def _encode_action_events(
        self,
        tokens: torch.Tensor,           # (Na, S_max)
        attention: torch.Tensor,        # (Na, S_max)
    ) -> torch.Tensor:
        """
        Encode a packed set of ACTION events with the action encoder.
        Returns pooled embeddings (Na, D).
        """
        if tokens.numel() == 0:
            return tokens.new_zeros((0, self._infer_embed_dim()))

        # Try named args first (HF-style); fallback to positional if needed
        try:
            out = self.action_encoder(input_ids=tokens, attention_mask=attention)
        except TypeError:
            out = self.action_encoder(tokens, attention)

        a = self._normalize_encoder_output(out)
        a = self._prepare_representation(a, self._is_hf_action_encoder)
        return a

    def _infer_embed_dim(self) -> int:
        """
        Best-effort inference of embedding dim from encoders or predictor.
        Used only for zero-size allocations in corner cases.
        """
        # Try state encoder hidden size
        if hasattr(self.encoder, "config") and hasattr(self.encoder.config, "n_embd"):
            return int(self.encoder.config.n_embd)
        if hasattr(self.encoder, "config") and hasattr(self.encoder.config, "hidden_size"):
            return int(self.encoder.config.hidden_size)
        # Try a projection on predictor
        if hasattr(self.predictor, "out") and isinstance(self.predictor.out, nn.Linear):
            return int(self.predictor.out.out_features)
        if hasattr(self.predictor, "proj") and isinstance(self.predictor.proj, nn.Linear):
            return int(self.predictor.proj.out_features)
        # Fallback
        return 768

    # ---------- main forward ----------

    def forward(
        self,
        inputs: torch.Tensor,                  # (B, E_max, S_max)
        inputs_attention: torch.Tensor,        # (B, E_max, S_max) bool
        event_type: torch.Tensor,              # (B, E_max) 0=state, 1=action
        labels: torch.Tensor,                  # (B, N_max)
        labels_attention: torch.Tensor,        # (B, N_max) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          pred:    (B, D)    predicted next-state embedding from the prefix
          z_next:  (B, D)    ground-truth next-state embedding (encoded from labels)
        """
        B, E, S = inputs.shape

        # Flatten events: (B*E, S)
        flat_tokens = inputs.view(B * E, S)
        flat_attn   = inputs_attention.view(B * E, S)
        flat_types  = event_type.view(B * E)

        # Build selectors
        is_state  = flat_types == 0
        is_action = flat_types == 1

        # Encode STATE events → z
        z_flat = self._encode_state_events(
            tokens=flat_tokens[is_state],
            attention=flat_attn[is_state],
        )  # (Ns, D)

        # Encode ACTION events → u
        u_flat = self._encode_action_events(
            tokens=flat_tokens[is_action],
            attention=flat_attn[is_action],
        )  # (Na, D)

        # Allocate full (B*E, D) and scatter back
        D = z_flat.size(-1) if z_flat.numel() else (u_flat.size(-1) if u_flat.numel() else self._infer_embed_dim())
        seq_flat = flat_tokens.new_zeros((B * E, D), dtype=torch.float32)

        if z_flat.numel():
            seq_flat[is_state] = z_flat
        if u_flat.numel():
            seq_flat[is_action] = u_flat

        # Reshape to (B, E, D)
        seq_emb = seq_flat.view(B, E, D)

        # Event mask: True if the event row has at least 1 valid token
        event_mask = (inputs_attention.any(dim=-1))  # (B, E) bool

        # ---- Predictor: consume sequence and output next-state embedding prediction ----
        # We forward optional hints if the predictor accepts them.
        try:
            pred = self.predictor(seq_emb, event_type=event_type, event_mask=event_mask)  # (B, D)
        except TypeError:
            pred = self.predictor(seq_emb)  # (B, D)

        # ---- Encode ground-truth next state from labels ----
        z_next = self._encode_state(input_ids=labels, attention_mask=labels_attention)  # (B, D) or (B, *, D)
        if z_next.ndim > 2:
            z_next = self._prepare_representation(z_next, self._is_hf_encoder)  # (B, D)

        # Detach targets (JEPA-style)
        z_next = z_next.detach()

        return pred, z_next
