"""Cross-Modal Attention (CMA).

Adapted from CMX (RGB-X transformer); used at three encoder scales. Depth
features query RGB key/value pairs. Memory is bounded by pooling spatial tokens
to ``N_max`` before attention and bilinearly upsampling the result. Attention
logits are clamped to ``[-8, 8]`` for numerical stability under FP16.

See Eqs. (3)-(4) in the paper.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """Multi-head cross-attention from a depth stream to an RGB stream.

    Args:
        d_channels: channels of the depth-stream feature map (query).
        r_channels: channels of the RGB-stream feature map (key/value).
        num_heads: attention heads.
        n_max: maximum number of spatial tokens after pooling.
        dropout: optional attention dropout probability.
    """

    def __init__(
        self,
        d_channels: int,
        r_channels: int,
        num_heads: int = 4,
        n_max: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_channels % num_heads != 0:
            raise ValueError("d_channels must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = d_channels // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.n_max = n_max

        self.norm_q = nn.GroupNorm(num_groups=min(32, d_channels), num_channels=d_channels)
        self.q_proj = nn.Conv2d(d_channels, d_channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(r_channels, d_channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(r_channels, d_channels, kernel_size=1, bias=False)
        self.o_proj = nn.Conv2d(d_channels, d_channels, kernel_size=1, bias=True)
        self.attn_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _maybe_pool(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        B, C, H, W = x.shape
        n = H * W
        if n <= self.n_max:
            return x, (H, W)
        # Pool by an integer factor that brings tokens below n_max.
        factor = max(1, int(math.ceil(math.sqrt(n / self.n_max))))
        h2 = max(1, H // factor)
        w2 = max(1, W // factor)
        return F.adaptive_avg_pool2d(x, (h2, w2)), (h2, w2)

    def forward(self, depth_feat: torch.Tensor, rgb_feat: torch.Tensor) -> torch.Tensor:
        """Args:
            depth_feat: query stream, ``(B, Cd, H, W)``.
            rgb_feat: key/value stream, ``(B, Cr, H, W)`` at the same spatial scale.

        Returns:
            Tensor ``(B, Cd, H, W)``: residual update added to ``depth_feat``.
        """
        B, _, H, W = depth_feat.shape
        q_in = self.q_proj(self.norm_q(depth_feat))
        k_in = self.k_proj(rgb_feat)
        v_in = self.v_proj(rgb_feat)

        # Pool to bound memory.
        q_pooled, (hq, wq) = self._maybe_pool(q_in)
        k_pooled, (hk, wk) = self._maybe_pool(k_in)
        v_pooled, _ = self._maybe_pool(v_in)

        # Reshape into (B, heads, N, head_dim).
        def to_heads(t: torch.Tensor) -> torch.Tensor:
            b, c, h, w = t.shape
            return (
                t.view(b, self.num_heads, self.head_dim, h * w)
                 .permute(0, 1, 3, 2)
                 .contiguous()
            )

        q = to_heads(q_pooled)
        k = to_heads(k_pooled)
        v = to_heads(v_pooled)

        # Scaled dot-product attention with clamping for FP16 stability.
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.clamp_(-8.0, 8.0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B, heads, N_q, head_dim)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, -1, hq, wq)
        out = self.o_proj(out)

        # Upsample back to the depth feature resolution if pooled.
        if (hq, wq) != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return depth_feat + out
