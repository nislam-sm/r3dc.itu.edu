"""SSIM loss for depth maps.

Uses a Gaussian window of size ``ws`` (default 7). Reduces to ``1 - SSIM``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    half = (window_size - 1) / 2.0
    x = torch.arange(window_size).float() - half
    g = torch.exp(-(x.pow(2)) / (2.0 * sigma * sigma))
    g = g / g.sum()
    return g.outer(g)


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 7, sigma: float = 1.5,
                 c1: float = 0.01 ** 2, c2: float = 0.03 ** 2):
        super().__init__()
        self.window_size = window_size
        self.c1, self.c2 = c1, c2
        window = _gaussian_window(window_size, sigma).view(1, 1, window_size, window_size)
        self.register_buffer("window", window)

    def _filter(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.window, padding=self.window_size // 2)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        mu_x = self._filter(pred)
        mu_y = self._filter(target)
        mu_xy = mu_x * mu_y
        mu_xx = mu_x * mu_x
        mu_yy = mu_y * mu_y

        sigma_x = self._filter(pred * pred) - mu_xx
        sigma_y = self._filter(target * target) - mu_yy
        sigma_xy = self._filter(pred * target) - mu_xy

        ssim_num = (2 * mu_xy + self.c1) * (2 * sigma_xy + self.c2)
        ssim_den = (mu_xx + mu_yy + self.c1) * (sigma_x + sigma_y + self.c2)
        ssim_map = ssim_num / ssim_den.clamp_min(1e-8)

        if mask is not None:
            m = mask.clamp(0, 1)
            denom = m.sum().clamp_min(1.0)
            return 1.0 - (ssim_map * m).sum() / denom
        return 1.0 - ssim_map.mean()
