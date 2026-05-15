"""FPN-style decoder with EfficientUpBlocks.

Each ``EfficientUpBlock`` performs:
  1. Transposed-conv 4x4/stride-2 upsample.
  2. DCN-based fusion of the depth-stream lateral skip.
  3. Pre-activation ResBlock + DropPath.
  4. CBAM channel/spatial attention.
  5. Cross-Modal Attention against the RGB-stream skip at the same scale.

See Section 3.5 of the paper.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from r3dc.models.cma import CrossModalAttention
from r3dc.models.common import CBAM, DeformConv2d, DropPath, ResBlock


class EfficientUpBlock(nn.Module):
    """A single decoder stage."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        rgb_skip_ch: int,
        n_max: int = 512,
        drop_path: float = 0.1,
        use_cma: bool = True,
    ):
        super().__init__()
        # 1) Upsample
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)

        # 2) DCN fusion of lateral skip
        self.skip_align = nn.Conv2d(skip_ch, out_ch, kernel_size=1, bias=False)
        self.fuse = nn.Sequential(
            DeformConv2d(2 * out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # 3) Residual refinement
        self.res = ResBlock(out_ch, drop_path=drop_path)
        # 4) CBAM
        self.cbam = CBAM(out_ch)
        # 5) CMA against RGB skip
        self.use_cma = use_cma
        if use_cma:
            self.cma = CrossModalAttention(out_ch, rgb_skip_ch, num_heads=max(1, out_ch // 16),
                                           n_max=n_max)

    def forward(self, x: torch.Tensor, depth_skip: torch.Tensor, rgb_skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != depth_skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=depth_skip.shape[-2:], mode="bilinear",
                                          align_corners=False)
        skip = self.skip_align(depth_skip)
        x = self.fuse(torch.cat([x, skip], dim=1))
        x = self.res(x)
        x = self.cbam(x)
        if self.use_cma:
            x = self.cma(x, rgb_skip)
        return x


class FPNDecoder(nn.Module):
    """Top-down FPN decoder consuming bottleneck + four lateral skips."""

    def __init__(
        self,
        bottleneck_ch: int,
        depth_skip_ch: List[int],
        rgb_skip_ch: List[int],
        out_channels: List[int],
        n_max: int = 512,
        drop_path: float = 0.1,
    ):
        super().__init__()
        if len(depth_skip_ch) != 4 or len(rgb_skip_ch) != 4 or len(out_channels) != 4:
            raise ValueError("Expecting 4 lateral skips and 4 output widths.")
        # dec4: bottleneck (1/16) -> 1/8 (uses skip s8)
        # dec3: 1/8 -> 1/4 (uses skip s4)
        # dec2: 1/4 -> 1/2 (uses skip s2)
        # dec1: 1/2 -> full (uses stem)
        ins = [bottleneck_ch, out_channels[0], out_channels[1], out_channels[2]]
        self.blocks = nn.ModuleList([
            EfficientUpBlock(
                in_ch=ins[i],
                skip_ch=depth_skip_ch[3 - i],
                rgb_skip_ch=rgb_skip_ch[3 - i],
                out_ch=out_channels[i],
                n_max=n_max,
                drop_path=drop_path,
            )
            for i in range(4)
        ])

    def forward(
        self,
        bottleneck: torch.Tensor,
        depth_skips: List[torch.Tensor],  # [stem, s2, s4, s8]
        rgb_skips: List[torch.Tensor],    # [stem, s2, s4, s8]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Returns ``(intermediate_feats, full_res_feature)``.

        ``intermediate_feats`` is ordered from deep (1/8 scale) to shallow
        (1/2 scale) — useful for auxiliary supervision heads.
        """
        x = bottleneck
        feats = []
        for i, block in enumerate(self.blocks):
            depth_skip = depth_skips[3 - i]
            rgb_skip = rgb_skips[3 - i]
            x = block(x, depth_skip, rgb_skip)
            feats.append(x)
        # feats: [1/8, 1/4, 1/2, full]
        return feats[:-1], feats[-1]
