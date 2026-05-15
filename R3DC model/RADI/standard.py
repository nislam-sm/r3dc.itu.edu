"""Standard depth-estimation metrics.

All metrics expect tensors in *metric* (linear) depth space, not log-normalised
space. Pass through :func:`r3dc.utils.lognorm.log_denormalize` before calling
these if you have model outputs in normalised space.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


def _valid(pred: torch.Tensor, target: torch.Tensor,
           mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flattened predicted and target values where ``mask`` is true."""
    m = (mask > 0.5) & (target > 0)
    return pred[m], target[m]


def rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    p, t = _valid(pred, target, mask)
    if p.numel() == 0:
        return float("nan")
    return torch.sqrt(((p - t) ** 2).mean()).item()


def mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    p, t = _valid(pred, target, mask)
    if p.numel() == 0:
        return float("nan")
    return (p - t).abs().mean().item()


def abs_rel(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    p, t = _valid(pred, target, mask)
    if p.numel() == 0:
        return float("nan")
    return ((p - t).abs() / t.clamp_min(1e-6)).mean().item()


def silog(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
          alpha: float = 0.85, eps: float = 1e-3) -> float:
    p, t = _valid(pred, target, mask)
    if p.numel() == 0:
        return float("nan")
    delta = torch.log(p.clamp_min(eps)) - torch.log(t.clamp_min(eps))
    return torch.sqrt(delta.pow(2).mean() - alpha * delta.mean().pow(2)).item()


def delta_n(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
            n: int = 1) -> float:
    """Fraction of pixels with ``max(p/t, t/p) < 1.25^n``."""
    p, t = _valid(pred, target, mask)
    if p.numel() == 0:
        return float("nan")
    ratio = torch.maximum(p / t.clamp_min(1e-6), t / p.clamp_min(1e-6))
    return (ratio < (1.25 ** n)).float().mean().item()


@dataclass
class StandardMetrics:
    """All standard metrics in one dataclass for easy logging."""

    rmse: float
    mae: float
    abs_rel: float
    silog: float
    delta1: float
    delta2: float
    delta3: float

    def to_dict(self) -> Dict[str, float]:
        return self.__dict__.copy()


def compute_standard_metrics(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> StandardMetrics:
    return StandardMetrics(
        rmse=rmse(pred, target, mask),
        mae=mae(pred, target, mask),
        abs_rel=abs_rel(pred, target, mask),
        silog=silog(pred, target, mask),
        delta1=delta_n(pred, target, mask, 1),
        delta2=delta_n(pred, target, mask, 2),
        delta3=delta_n(pred, target, mask, 3),
    )
