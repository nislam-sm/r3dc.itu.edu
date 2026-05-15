"""Dual-stream encoder (RGB + sparse depth) with optional CMA at each stage.

* The RGB stream is a stem + three stride-2 residual stages with widths
  ``{B/2, B, 2B, 4B}`` at scales ``{1, 1/2, 1/4, 1/8}``.
* The sparse-depth stream mirrors the RGB stream, but uses **Deformable
  Convolution v2** in the stride-2 stages so the receptive field can adapt
  to irregular sparse points.
* After each encoder stage the depth features can be enhanced with
  :class:`CrossModalAttention` queries against the corresponding RGB features.

The encoder returns the lateral skips for the FPN decoder *and* the deepest
features for the bottleneck. See Section 3.3 of the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from r3dc.models.cma import CrossModalAttention
from r3dc.models.common import DeformConv2d, DropPath, ResBlock


@dataclass
class EncoderFeatures:
    """Multi-scale skips returned by :class:`DualStreamEncoder`."""
    rgb: List[torch.Tensor]    # [stem, s2, s4, s8]
    depth: List[torch.Tensor]  # [stem, s2, s4, s8] (post-CMA fusion)


# ---------------------------------------------------------------------------
def _stem(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


def _rgb_down(in_ch: int, out_ch: int, drop_path: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        ResBlock(out_ch, drop_path=drop_path),
    )


def _depth_down(in_ch: int, out_ch: int, drop_path: float) -> nn.Sequential:
    """Stride-2 downsample using DCNv2."""
    return nn.Sequential(
        DeformConv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        ResBlock(out_ch, drop_path=drop_path),
    )


# ---------------------------------------------------------------------------
class DualStreamEncoder(nn.Module):
    """RGB + sparse-depth dual-stream encoder.

    Args:
        base_channels: width ``B`` of the RGB stream. Channels at stages
            ``{stem, s2, s4, s8}`` are ``{B/2, B, 2B, 4B}``.
        depth_in_ch: input channels of the sparse-depth stream
            (typically 2 = [normalised depth, mask]).
        use_cma: enable cross-modal attention at stages s2/s4/s8.
        n_max: token cap for CMA (see :class:`CrossModalAttention`).
        drop_path: DropPath probability for residual blocks.
    """

    def __init__(
        self,
        base_channels: int = 64,
        depth_in_ch: int = 2,
        use_cma: bool = True,
        n_max: int = 512,
        drop_path: float = 0.1,
    ):
        super().__init__()
        B = base_channels
        chans = [B // 2, B, 2 * B, 4 * B]
        self.channels = chans

        # ------- RGB stream -------
        self.rgb_stem = _stem(3, chans[0])
        self.rgb_s2 = _rgb_down(chans[0], chans[1], drop_path=drop_path)
        self.rgb_s4 = _rgb_down(chans[1], chans[2], drop_path=drop_path)
        self.rgb_s8 = _rgb_down(chans[2], chans[3], drop_path=drop_path)

        # ------- Depth stream -------
        self.dep_stem = _stem(depth_in_ch, chans[0])
        self.dep_s2 = _depth_down(chans[0], chans[1], drop_path=drop_path)
        self.dep_s4 = _depth_down(chans[1], chans[2], drop_path=drop_path)
        self.dep_s8 = _depth_down(chans[2], chans[3], drop_path=drop_path)

        # ------- CMA at each scale -------
        self.use_cma = use_cma
        if use_cma:
            self.cma_s2 = CrossModalAttention(chans[1], chans[1], num_heads=4, n_max=n_max)
            self.cma_s4 = CrossModalAttention(chans[2], chans[2], num_heads=4, n_max=n_max)
            self.cma_s8 = CrossModalAttention(chans[3], chans[3], num_heads=8, n_max=n_max)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> EncoderFeatures:
        # RGB tower
        r0 = self.rgb_stem(rgb)
        r1 = self.rgb_s2(r0)
        r2 = self.rgb_s4(r1)
        r3 = self.rgb_s8(r2)

        # Depth tower with CMA fusion
        d0 = self.dep_stem(depth)
        d1 = self.dep_s2(d0)
        if self.use_cma:
            d1 = self.cma_s2(d1, r1)
        d2 = self.dep_s4(d1)
        if self.use_cma:
            d2 = self.cma_s4(d2, r2)
        d3 = self.dep_s8(d2)
        if self.use_cma:
            d3 = self.cma_s8(d3, r3)

        return EncoderFeatures(rgb=[r0, r1, r2, r3], depth=[d0, d1, d2, d3])
