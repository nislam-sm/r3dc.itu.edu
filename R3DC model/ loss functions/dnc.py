"""Depth-Normal Consistency (DNC) loss.

Computes surface normals from Sobel-filtered depth gradients and rewards
agreement between horizontally and vertically neighbouring normals. See
Eq. (16) in the paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0],
                         [-2.0, 0.0, 2.0],
                         [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1.0, -2.0, -1.0],
                         [ 0.0,  0.0,  0.0],
                         [ 1.0,  2.0,  1.0]]).view(1, 1, 3, 3)


class DepthNormalConsistencyLoss(nn.Module):
    """Encourages locally consistent surface normals derived from depth."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer("sobel_x", _SOBEL_X)
        self.register_buffer("sobel_y", _SOBEL_Y)

    def _normals(self, depth: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(depth, self.sobel_x, padding=1)
        gy = F.conv2d(depth, self.sobel_y, padding=1)
        ones = torch.ones_like(depth)
        n = torch.cat([-gx, -gy, ones], dim=1)  # (B, 3, H, W)
        return n / n.norm(dim=1, keepdim=True).clamp_min(self.eps)

    def forward(self, pred_depth: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        normals = self._normals(pred_depth)
        # Horizontal and vertical neighbour dot-products.
        h_dot = (normals[..., :, :-1] * normals[..., :, 1:]).sum(dim=1, keepdim=True)
        v_dot = (normals[..., :-1, :] * normals[..., 1:, :]).sum(dim=1, keepdim=True)
        if mask is not None:
            h_mask = mask[..., :, :-1] * mask[..., :, 1:]
            v_mask = mask[..., :-1, :] * mask[..., 1:, :]
            h_dot = h_dot * h_mask
            v_dot = v_dot * v_mask
            denom = (h_mask.sum() + v_mask.sum()).clamp_min(1.0)
            return 1.0 - (h_dot.sum() + v_dot.sum()) / (denom * 2.0)
        return 1.0 - 0.5 * (h_dot.mean() + v_dot.mean())
