"""Common building blocks used throughout the R3DC architecture.

Provides:
* :class:`DropPath` — stochastic-depth regularisation.
* :class:`ResBlock` — pre-activation residual block.
* :class:`CBAM` — channel-then-spatial attention module.
* :class:`DeformConv2d` — wrapper around DCNv2 with a graceful fallback to
  plain ``Conv2d`` when torchvision's ``deform_conv2d`` is unavailable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import deform_conv2d as _tv_deform_conv2d
    _HAS_DCN = True
except ImportError:  # pragma: no cover
    _HAS_DCN = False


# ---------------------------------------------------------------------------
# DropPath
# ---------------------------------------------------------------------------
def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Stochastic depth (per sample) used inside residual branches."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.dim() - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# Residual block (pre-activation)
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """Two-layer pre-activation residual block.

    Identity shortcut when in/out channels match; 1x1 projection otherwise.
    """

    def __init__(self, in_ch: int, out_ch: int | None = None, drop_path: float = 0.0):
        super().__init__()
        out_ch = out_ch or in_ch
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.act = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.shortcut = (
            nn.Identity() if in_ch == out_ch
            else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(self.act(self.bn1(x)))
        out = self.conv2(self.act(self.bn2(out)))
        return identity + self.drop_path(out)


# ---------------------------------------------------------------------------
# CBAM
# ---------------------------------------------------------------------------
class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., 2018)."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        hidden = max(1, channels // reduction)
        # Channel attention via avg + max pooled MLPs (shared weights).
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        # Spatial attention from avg+max pooling along channel dim.
        self.spatial = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                                 padding=spatial_kernel // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # Channel attention
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        ca = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        x = x * ca
        # Spatial attention
        avg_s = x.mean(dim=1, keepdim=True)
        max_s, _ = x.max(dim=1, keepdim=True)
        sa = torch.sigmoid(self.spatial(torch.cat([avg_s, max_s], dim=1)))
        return x * sa


# ---------------------------------------------------------------------------
# Deformable Convolution v2 wrapper
# ---------------------------------------------------------------------------
class DeformConv2d(nn.Module):
    """DCNv2 wrapper.

    Predicts offsets ``Δp_k`` and modulation scalars ``m_k`` from the input,
    then performs deformable convolution. The offset/modulation prediction
    network is zero-initialised, so the layer behaves as a standard convolution
    at step 0 and progressively learns offsets during training.

    Falls back to a plain ``Conv2d`` if torchvision's ``deform_conv2d`` is not
    available, with a one-time warning. This keeps tests runnable in minimal
    environments.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = True,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None

        # 2*K*K offsets + K*K modulation masks
        self.offset_conv = nn.Conv2d(
            in_ch,
            3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        nn.init.zeros_(self.offset_conv.weight)
        if self.offset_conv.bias is not None:
            nn.init.zeros_(self.offset_conv.bias)

        self._fallback_warned = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ks = self.kernel_size
        if not _HAS_DCN:
            if not self._fallback_warned:
                import warnings
                warnings.warn(
                    "torchvision.ops.deform_conv2d unavailable — falling back "
                    "to plain Conv2d. Install a recent torchvision for DCNv2.",
                    RuntimeWarning,
                )
                self._fallback_warned = True
            return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)

        offset_mask = self.offset_conv(x)
        off = offset_mask[:, : 2 * ks * ks]
        mask = torch.sigmoid(offset_mask[:, 2 * ks * ks:])
        return _tv_deform_conv2d(
            x, off, self.weight, bias=self.bias,
            stride=self.stride, padding=self.padding, mask=mask,
        )
