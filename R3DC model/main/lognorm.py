"""Log-space depth normalisation (Eq. 1 in the paper).

Compresses wide outdoor depth ranges (KITTI ~80 m) and indoor ranges
(NYU ~10 m) onto the same [0, 1] interval while preserving relative ordering.
Pairs naturally with the scale-invariant log (SILog) loss.
"""
from __future__ import annotations

import torch

EPS = 1e-3


def log_normalize(depth: torch.Tensor, d_min: float, d_max: float, eps: float = EPS) -> torch.Tensor:
    """Map a raw metric depth `d` to the unit interval via log scaling.

    .. math::

        \\tilde{d} =
        \\frac{\\ln(d + \\varepsilon) - \\ln(d_{\\min} + \\varepsilon)}
             {\\ln(d_{\\max} + \\varepsilon) - \\ln(d_{\\min} + \\varepsilon)}

    Args:
        depth: depth tensor of any shape, in metres.
        d_min: lower bound of the dataset's depth range.
        d_max: upper bound of the dataset's depth range.
        eps: numerical offset to avoid ``log(0)``.

    Returns:
        Tensor with the same shape, values approximately in ``[0, 1]``.
    """
    if d_max <= d_min:
        raise ValueError(f"d_max ({d_max}) must be greater than d_min ({d_min}).")
    lo = torch.log(torch.as_tensor(d_min + eps, dtype=depth.dtype, device=depth.device))
    hi = torch.log(torch.as_tensor(d_max + eps, dtype=depth.dtype, device=depth.device))
    return (torch.log(depth.clamp_min(0) + eps) - lo) / (hi - lo)


def log_denormalize(normed: torch.Tensor, d_min: float, d_max: float, eps: float = EPS) -> torch.Tensor:
    """Inverse of :func:`log_normalize`: map ``[0, 1]`` back to metric depth."""
    if d_max <= d_min:
        raise ValueError(f"d_max ({d_max}) must be greater than d_min ({d_min}).")
    lo = torch.log(torch.as_tensor(d_min + eps, dtype=normed.dtype, device=normed.device))
    hi = torch.log(torch.as_tensor(d_max + eps, dtype=normed.dtype, device=normed.device))
    return torch.exp(normed * (hi - lo) + lo) - eps
