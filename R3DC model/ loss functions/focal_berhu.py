"""Focal-BerHu loss with hard-example mining.

A standard BerHu (reversed Huber) regression loss wrapped in a focal weight
``(1 - exp(-|e|))^gamma`` that smoothly up-weights pixels with large errors
without a hard threshold. See Eq. (12) in the paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FocalBerHuLoss(nn.Module):
    """Focal-BerHu hybrid loss.

    Args:
        gamma: focal exponent (paper uses 2).
        c_ratio: BerHu threshold as a fraction of the max error in the batch.
        eps: numerical floor.
    """

    def __init__(self, gamma: float = 2.0, c_ratio: float = 0.2, eps: float = 1e-6):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        if not 0.0 < c_ratio <= 1.0:
            raise ValueError("c_ratio must be in (0, 1].")
        self.gamma = gamma
        self.c_ratio = c_ratio
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = mask > 0.5
        if valid.sum() == 0:
            return pred.sum() * 0.0

        diff = (pred - target)[valid]
        err = diff.abs()
        c = (self.c_ratio * err.max()).clamp_min(self.eps)

        # BerHu term
        l1 = err
        l2 = (diff.pow(2) + c.pow(2)) / (2.0 * c)
        berhu = torch.where(err <= c, l1, l2)

        # Focal weight
        focal = (1.0 - torch.exp(-err)).pow(self.gamma)
        return (focal * berhu).mean()
