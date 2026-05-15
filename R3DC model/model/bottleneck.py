"""Transformer bottleneck at 1/16 resolution.

Self-attention with multi-head spatial tokens (token cap matches CMA), followed
by an MLP and a CBAM channel/spatial attention block. DropPath is applied to
both the attention and MLP residual branches (see Eqs. (5)-(6) in the paper).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from r3dc.models.common import CBAM, DropPath


class _MHSelfAttention(nn.Module):
    """Memory-bounded multi-head self-attention with bilinear up-/down-pool."""

    def __init__(self, channels: int, num_heads: int = 8, n_max: int = 512):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.n_max = n_max
        self.qkv = nn.Conv2d(channels, 3 * channels, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x)

        # Optional pool to N_max tokens.
        n = H * W
        if n > self.n_max:
            factor = max(1, int(math.ceil(math.sqrt(n / self.n_max))))
            h2 = max(1, H // factor)
            w2 = max(1, W // factor)
            qkv = F.adaptive_avg_pool2d(qkv, (h2, w2))
        else:
            h2, w2 = H, W

        q, k, v = qkv.chunk(3, dim=1)

        def to_heads(t: torch.Tensor) -> torch.Tensor:
            b, c, h, w = t.shape
            return (
                t.view(b, self.num_heads, self.head_dim, h * w)
                 .permute(0, 1, 3, 2)
                 .contiguous()
            )

        attn = torch.matmul(to_heads(q), to_heads(k).transpose(-2, -1)) * self.scale
        attn = attn.clamp_(-8.0, 8.0).softmax(dim=-1)
        out = torch.matmul(attn, to_heads(v))
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, h2, w2)

        if (h2, w2) != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return self.proj(out)


class TransformerBottleneck(nn.Module):
    """Stack of (self-attention + MLP) followed by CBAM."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        n_max: int = 512,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.attn = _MHSelfAttention(channels, num_heads=num_heads, n_max=n_max)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        hidden = int(channels * mlp_ratio)
        self.norm2 = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.cbam = CBAM(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return self.cbam(x)
