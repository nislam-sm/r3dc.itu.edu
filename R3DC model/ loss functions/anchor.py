"""Sparse Anchor loss.

L1 loss between the predicted dense depth and the sparse input at the
positions where sparse measurements exist. Prevents metric-scale drift,
especially at extreme sparsity.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SparseAnchorLoss(nn.Module):
    """L1 loss restricted to sparse-anchor pixels."""

    def forward(
        self,
        pred: torch.Tensor,
        sparse_target: torch.Tensor,
        sparse_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = sparse_mask > 0.5
        if valid.sum() == 0:
            return pred.sum() * 0.0
        return (pred[valid] - sparse_target[valid]).abs().mean()
