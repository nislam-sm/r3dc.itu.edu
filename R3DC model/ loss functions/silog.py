"""Scale-Invariant Log (SILog) loss (Eigen et al., 2014).

.. math::

    \\mathcal{L}_{SI} = \\frac{1}{|V|} \\sum_{p\\in V} \\Delta_\\ell^2
                    - \\frac{0.85}{|V|^2}\\Bigl(\\sum_{p\\in V}\\Delta_\\ell\\Bigr)^2,
    \\quad \\Delta_\\ell = \\ln \\hat{d} - \\ln d_{gt}.

Operates in *log-normalised* depth space where inputs are in ``(0, 1]`` and a
small ``eps`` is added before taking ``log`` to avoid NaNs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SILogLoss(nn.Module):
    def __init__(self, alpha: float = 0.85, eps: float = 1e-3):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        self.alpha = alpha
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = mask > 0.5
        if valid.sum() == 0:
            return pred.sum() * 0.0  # keeps the graph but contributes zero

        # ``pred`` and ``target`` live in [0, 1]; clamp to avoid log(0).
        p = pred[valid].clamp_min(self.eps)
        t = target[valid].clamp_min(self.eps)
        delta = torch.log(p) - torch.log(t)
        n = delta.numel()
        return (delta.pow(2).mean()
                - self.alpha * delta.mean().pow(2)) * (1.0 if n > 0 else 0.0)
