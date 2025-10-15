import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Base class for JEPA models with lightweight save/load utilities."""

    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):  # pragma: no cover - interface definition
        raise NotImplementedError("Forward method must be implemented by subclasses.")

    # ------------------------------------------------------------------
    # Legacy helpers
    # ------------------------------------------------------------------
    def save(self, path: str):
        """Save only the state dict to ``path`` (legacy helper)."""
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: Optional[str] = None):
        """Load weights from ``path`` and switch to eval mode (legacy helper)."""
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)
        self.eval()

    # ------------------------------------------------------------------
    # HuggingFace-style persistence API
    # ------------------------------------------------------------------
    def save_pretrained(
        self,
        save_directory: str,
        *,
        config: Optional[Dict[str, Any]] = None,
        safe_serialization: bool = False,
        filename: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        Persist model weights (and optional config) in a directory, mimicking
        ``transformers.PreTrainedModel.save_pretrained``.

        Args:
            save_directory: Target directory (will be created if missing).
            config: Optional configuration dictionary to write as ``config.json``.
                If omitted, ``self.config`` is used when available.
            safe_serialization: If ``True`` and ``safetensors`` is installed, store
                weights as ``model.safetensors``.
            filename: Optional custom filename for the weights (defaults to
                ``model.safetensors`` or ``pytorch_model.bin`` depending on
                ``safe_serialization``).

        Returns:
            Tuple with paths ``(weights_path, config_path_or_none)``.
        """
        os.makedirs(save_directory, exist_ok=True)

        weights_path: str
        if safe_serialization:
            try:
                from safetensors.torch import save_file  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "Safe serialization requires the `safetensors` package."
                ) from exc
            weights_name = filename or "model.safetensors"
            weights_path = os.path.join(save_directory, weights_name)
            save_file(self.state_dict(), weights_path)
        else:
            weights_name = filename or "pytorch_model.bin"
            weights_path = os.path.join(save_directory, weights_name)
            torch.save(self.state_dict(), weights_path)

        config_dict = config if config is not None else getattr(self, "config", None)
        config_path: Optional[str] = None
        if config_dict is not None:
            config_path = os.path.join(save_directory, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(_make_json_serializable(config_dict), f, indent=2)

        return weights_path, config_path

    @classmethod
    def from_pretrained(
        cls,
        save_directory: str,
        *,
        map_location: Optional[str] = "auto",
        strict: bool = True,
        config: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
        **model_kwargs: Any,
    ) -> "BaseModel":
        """
        Instantiate a model from weights written by :meth:`save_pretrained`.

        Args:
            save_directory: Directory containing saved weights.
            map_location: Passed to ``torch.load`` (``"auto"`` chooses CUDA when
                available).
            strict: Passed to ``load_state_dict``.
            config: Optional config dictionary. If omitted, ``config.json`` is
                loaded when present.
            filename: Override weight filename. Defaults to
                ``model.safetensors`` (when present) else ``pytorch_model.bin``.
            **model_kwargs: Additional keyword arguments forwarded to the class
                constructor.

        Returns:
            Instantiated model in evaluation mode.
        """

        config_dict = config
        config_path = os.path.join(save_directory, "config.json")
        if config_dict is None and os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)

        if "config" not in model_kwargs and config_dict is not None:
            model_kwargs = {**model_kwargs, "config": config_dict}

        model = cls(**model_kwargs)

        if map_location == "auto":
            map_location = "cuda" if torch.cuda.is_available() else "cpu"

        weights_path = _resolve_weight_file(save_directory, filename)

        state_dict = _load_state_dict(weights_path, map_location=map_location)
        model.load_state_dict(state_dict, strict=strict)

        if config_dict is not None:
            setattr(model, "config", config_dict)

        model.eval()
        return model


def _make_json_serializable(data: Any) -> Any:
    """Convert objects that are not JSON serializable into strings."""
    if isinstance(data, dict):
        return {key: _make_json_serializable(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_make_json_serializable(item) for item in data]
    if isinstance(data, (str, int, float, bool)) or data is None:
        return data
    return str(data)


def _resolve_weight_file(save_directory: str, filename: Optional[str]) -> str:
    """Locate the appropriate weight file inside ``save_directory``."""
    if filename is not None:
        candidate = os.path.join(save_directory, filename)
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"No weight file found at {candidate}")
        return candidate

    safetensors_path = os.path.join(save_directory, "model.safetensors")
    if os.path.isfile(safetensors_path):
        return safetensors_path

    torch_path = os.path.join(save_directory, "pytorch_model.bin")
    if os.path.isfile(torch_path):
        return torch_path

    raise FileNotFoundError(
        "Expected weight file `model.safetensors` or `pytorch_model.bin` in "
        f"{save_directory}"
    )


def _load_state_dict(path: str, map_location: Optional[str]):
    """Load a state dict from ``path`` handling safetensors when present."""
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Encountered a safetensors file but the `safetensors` package "
                "is not installed."
            ) from exc
        return load_file(path, device=map_location)

    return torch.load(path, map_location=map_location)
