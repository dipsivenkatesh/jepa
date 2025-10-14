"""Loss function definitions for JEPA training."""

from __future__ import annotations

from functools import partial
from typing import Callable, Dict

import torch
import torch.nn.functional as F


LossFn = Callable[..., torch.Tensor]


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Return a flat view of the off-diagonal elements of a square matrix."""
    n, m = x.shape
    if n != m:
        raise ValueError("Input must be a square matrix")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    *,
    lambda_inv: float = 25.0,
    mu_var: float = 25.0,
    nu_cov: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
    center: bool = True,
) -> torch.Tensor:
    """Variance-Invariance-Covariance Regularization loss."""
    if z1.shape != z2.shape or z1.dim() != 2:
        raise ValueError("Inputs must both be shaped (batch, dim)")

    batch_size, dim = z1.shape

    inv = F.mse_loss(z1, z2)

    std_z1 = torch.sqrt(z1.var(dim=0, unbiased=False) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0, unbiased=False) + eps)
    var_z1 = torch.mean(F.relu(gamma - std_z1))
    var_z2 = torch.mean(F.relu(gamma - std_z2))
    var = var_z1 + var_z2

    if center:
        z1_c = z1 - z1.mean(dim=0, keepdim=True)
        z2_c = z2 - z2.mean(dim=0, keepdim=True)
    else:
        z1_c, z2_c = z1, z2

    denom = max(batch_size - 1, 1)
    cov_z1 = (z1_c.T @ z1_c) / denom
    cov_z2 = (z2_c.T @ z2_c) / denom
    cov = _off_diagonal(cov_z1).pow(2).sum() / dim
    cov += _off_diagonal(cov_z2).pow(2).sum() / dim

    return lambda_inv * inv + mu_var * var + nu_cov * cov


def vcreg_loss(
    z: torch.Tensor,
    *,
    mu_var: float = 25.0,
    nu_cov: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
    center: bool = True,
) -> torch.Tensor:
    """Variance + covariance regularization (single-view)."""
    if z.dim() != 2:
        raise ValueError("Input must be shaped (batch, dim)")

    batch_size, dim = z.shape

    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    var_term = torch.mean(F.relu(gamma - std))

    z_centered = z - z.mean(dim=0, keepdim=True) if center else z
    cov = (z_centered.T @ z_centered) / max(batch_size - 1, 1)
    cov_term = _off_diagonal(cov).pow(2).sum() / dim

    return mu_var * (2.0 * var_term) + nu_cov * cov_term


def mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Mean squared error loss."""
    return F.mse_loss(prediction, target, reduction=reduction)


LOSS_FUNCTIONS: Dict[str, LossFn] = {
    "vicreg": vicreg_loss,
    "vcreg": vcreg_loss,
    "mse": mse_loss,
    "l2": mse_loss,
    "mean_squared_error": mse_loss,
}


def get_loss(name: str, **kwargs) -> LossFn:
    """Return a configured loss function by name."""
    key = name.lower()
    if key not in LOSS_FUNCTIONS:
        available = ", ".join(sorted(LOSS_FUNCTIONS))
        raise KeyError(f"Unknown loss '{name}'. Available losses: {available}")
    fn = LOSS_FUNCTIONS[key]
    return partial(fn, **kwargs) if kwargs else fn


__all__ = [
    "vicreg_loss",
    "vcreg_loss",
    "mse_loss",
    "LOSS_FUNCTIONS",
    "get_loss",
]
