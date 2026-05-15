"""Region masks for RADI evaluation.

The four region types are:

* ``all``         — full valid set.
* ``edge``        — Sobel-magnitude on the RGB exceeds a threshold.
* ``textureless`` — local luminance standard deviation below a threshold.
* ``far``         — depth exceeds a fraction of the maximum.

All boolean masks share the same spatial layout as the inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class RegionConfig:
    """Hyperparameters from the paper (Sec. 4)."""

    sobel_threshold: float = 0.05      # for the "edge" mask
    luma_std_threshold: float = 8.0    # for the "textureless" mask (on 0-255 luma)
    far_fraction: float = 0.75         # for the "far" mask
    window_size: int = 7               # local std window


def _rgb_to_luma(rgb: torch.Tensor) -> torch.Tensor:
    # Rec. 601 weights; rgb assumed in [0, 1].
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _sobel_magnitude(luma: torch.Tensor) -> torch.Tensor:
    device = luma.device
    sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                            [-2.0, 0.0, 2.0],
                            [-1.0, 0.0, 1.0]], device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                            [ 0.0,  0.0,  0.0],
                            [ 1.0,  2.0,  1.0]], device=device).view(1, 1, 3, 3)
    gx = F.conv2d(luma, sobel_x, padding=1)
    gy = F.conv2d(luma, sobel_y, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def _local_std(luma: torch.Tensor, ws: int) -> torch.Tensor:
    pad = ws // 2
    mean = F.avg_pool2d(luma, kernel_size=ws, stride=1, padding=pad)
    mean_sq = F.avg_pool2d(luma * luma, kernel_size=ws, stride=1, padding=pad)
    return torch.sqrt((mean_sq - mean * mean).clamp_min(0.0))


def build_region_masks(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    valid_mask: torch.Tensor,
    config: RegionConfig | None = None,
) -> Dict[str, torch.Tensor]:
    """Build per-region boolean masks.

    Args:
        rgb: ``(B, 3, H, W)`` in ``[0, 1]``.
        depth: ``(B, 1, H, W)``, metric depth.
        valid_mask: ``(B, 1, H, W)``, 1 where ground truth is valid.
        config: optional :class:`RegionConfig`.

    Returns:
        Dict mapping ``"all"``, ``"edge"``, ``"textureless"``, ``"far"`` to
        boolean tensors of shape ``(B, 1, H, W)``.
    """
    cfg = config or RegionConfig()
    luma = _rgb_to_luma(rgb)
    sobel = _sobel_magnitude(luma)
    std = _local_std(luma * 255.0, cfg.window_size)  # paper uses 0-255 luma scale
    far_thresh = cfg.far_fraction * depth.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)

    valid_bool = valid_mask > 0.5
    return {
        "all": valid_bool,
        "edge": valid_bool & (sobel > cfg.sobel_threshold),
        "textureless": valid_bool & (std < cfg.luma_std_threshold),
        "far": valid_bool & (depth > far_thresh),
    }
