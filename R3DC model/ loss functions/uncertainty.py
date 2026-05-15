"""Laplace negative log-likelihood for self-supervised uncertainty.

.. math::

    \\mathcal{L}_{UNC} = \\frac{1}{|V|} \\sum_{p\\in V}
        \\frac{|\\hat{d} - d_{gt}|}{\\hat{\\sigma}} + \\ln \\hat{\\sigma}.

Under a Laplace likelihood, this is the maximum-likelihood objective for
jointly learning depth and its scale. The first term penalises confident-but-
wrong predictions; the ``ln σ`` term prevents trivial collapse to σ → 0.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LaplaceUncertaintyLoss(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred_depth: torch.Tensor,
        target_depth: torch.Tensor,
        uncertainty: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = mask > 0.5
        if valid.sum() == 0:
            return pred_depth.sum() * 0.0
        sigma = uncertainty[valid].clamp_min(self.eps)
        diff = (pred_depth - target_depth)[valid].abs()
        return (diff / sigma + torch.log(sigma)).mean()


class GradientConsistencyLoss(nn.Module):
    """L1 loss on first-order spatial gradients of depth (Sobel-style)."""

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        p_dx = pred[..., :, 1:] - pred[..., :, :-1]
        p_dy = pred[..., 1:, :] - pred[..., :-1, :]
        t_dx = target[..., :, 1:] - target[..., :, :-1]
        t_dy = target[..., 1:, :] - target[..., :-1, :]
        if mask is None:
            return ((p_dx - t_dx).abs().mean() + (p_dy - t_dy).abs().mean())
        m_dx = mask[..., :, 1:] * mask[..., :, :-1]
        m_dy = mask[..., 1:, :] * mask[..., :-1, :]
        loss_x = ((p_dx - t_dx).abs() * m_dx).sum() / m_dx.sum().clamp_min(1.0)
        loss_y = ((p_dy - t_dy).abs() * m_dy).sum() / m_dy.sum().clamp_min(1.0)
        return loss_x + loss_y
