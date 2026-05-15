"""Output heads for depth, reliability, uncertainty, and deep supervision.

Each head is a small ``Conv 3x3 -> ReLU -> Conv 1x1`` stack followed by the
appropriate activation:

* Depth: sigmoid (output in [0, 1] in log-normalised space)
* Reliability: sigmoid (output in [0, 1])
* Uncertainty: softplus (strictly positive)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvHead(nn.Module):
    """Conv 3x3 -> ReLU -> Conv 1x1."""

    def __init__(self, in_ch: int, hidden: int | None = None, out_ch: int = 1):
        super().__init__()
        hidden = hidden or max(16, in_ch // 2)
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_ch, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class DepthHead(nn.Module):
    """Predicts log-normalised depth in [0, 1]."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.head = ConvHead(in_ch, out_ch=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(x))


class ReliabilityHead(nn.Module):
    """Predicts per-pixel reliability in [0, 1]."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.head = ConvHead(in_ch, out_ch=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(x))


class UncertaintyHead(nn.Module):
    """Predicts strictly positive aleatoric uncertainty via softplus.

    A small floor is added to prevent exact zero, which would blow up the
    Laplace NLL loss.
    """

    def __init__(self, in_ch: int, eps: float = 1e-4):
        super().__init__()
        self.head = ConvHead(in_ch, out_ch=1)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.softplus(self.head(x)) + self.eps


class AuxDepthHead(nn.Module):
    """Lightweight 1x1 Conv + Sigmoid for intermediate-scale supervision."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.head = nn.Conv2d(in_ch, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(x))
