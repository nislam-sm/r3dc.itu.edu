"""Reliability-Gated CSPN++ propagation (Sec. 3.5 in the paper).

Standard CSPN/CSPN++ propagates depth using learned affinities; we additionally
condition the affinity network on the reliability map ``R̂`` so the refiner
absorbs more spatial context in low-confidence regions and is gentler in
high-confidence ones. Sparse anchors are enforced as hard Dirichlet boundary
conditions at every iteration.

Affinity normalisation: 8 neighbours softmax-scaled to sum to 0.8; the centre
pixel keeps a fixed weight of 0.2, so affinities always sum to 1.0.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _AffinityNet(nn.Module):
    """Two-layer conv predicting 8 affinity logits per pixel."""

    def __init__(self, in_ch: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or max(32, in_ch // 2)
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 8, kernel_size=3, padding=1, bias=True),  # 3x3 minus center
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class CSPNRefiner(nn.Module):
    """Iterative 3x3 spatial propagation with hard sparse-anchor constraints.

    Args:
        feat_channels: channels of the decoder feature ``u_1``.
        iterations: number of propagation steps ``T`` (default 6).
        neighbour_total_weight: total weight assigned to the 8 neighbours
            (the centre pixel keeps ``1 - neighbour_total_weight``).
    """

    def __init__(
        self,
        feat_channels: int,
        iterations: int = 6,
        neighbour_total_weight: float = 0.8,
    ):
        super().__init__()
        if not 0.0 < neighbour_total_weight < 1.0:
            raise ValueError("neighbour_total_weight must be in (0, 1).")
        self.iterations = iterations
        self.neighbour_total_weight = neighbour_total_weight
        self.centre_weight = 1.0 - neighbour_total_weight
        # Reliability is concatenated as an extra channel.
        self.affinity_net = _AffinityNet(feat_channels + 1)

    @staticmethod
    def _shift_neighbours(depth: torch.Tensor) -> torch.Tensor:
        """Return shifted versions of ``depth`` for the 8 neighbour offsets.

        Output shape: ``(B, 8, H, W)`` with channel order (top-left, top,
        top-right, left, right, bottom-left, bottom, bottom-right).
        """
        padded = F.pad(depth, (1, 1, 1, 1), mode="replicate")
        offsets = [
            (0, 0), (0, 1), (0, 2),  # top-row
            (1, 0),         (1, 2),  # mid-row (skip centre)
            (2, 0), (2, 1), (2, 2),  # bottom-row
        ]
        h, w = depth.shape[-2:]
        slices = [padded[..., dy:dy + h, dx:dx + w] for (dy, dx) in offsets]
        return torch.cat(slices, dim=1)

    def forward(
        self,
        coarse_depth: torch.Tensor,        # (B, 1, H, W) in [0, 1]
        features: torch.Tensor,            # (B, C, H, W)
        reliability: torch.Tensor,         # (B, 1, H, W) in [0, 1]
        sparse_depth: torch.Tensor,        # (B, 1, H, W) normalised
        sparse_mask: torch.Tensor,         # (B, 1, H, W) in {0, 1}
    ) -> torch.Tensor:
        """Refine ``coarse_depth`` via reliability-gated CSPN++.

        Returns the refined depth ``D_1`` (still in log-normalised [0, 1] space).
        """
        # Affinity weights (B, 8, H, W), softmax-normalised then scaled.
        logits = self.affinity_net(torch.cat([features, reliability], dim=1))
        w = logits.softmax(dim=1) * self.neighbour_total_weight

        # Anchor pixels are enforced as hard boundary conditions at every step.
        anchor_vals = sparse_depth * sparse_mask
        d_t = coarse_depth * (1.0 - sparse_mask) + anchor_vals * sparse_mask

        for _ in range(self.iterations):
            neigh = self._shift_neighbours(d_t)                    # (B, 8, H, W)
            update = (w * neigh).sum(dim=1, keepdim=True)          # (B, 1, H, W)
            d_t = self.centre_weight * d_t + update
            # Re-enforce the Dirichlet boundary condition.
            d_t = sparse_mask * anchor_vals + (1.0 - sparse_mask) * d_t

        # Uncovered pixels receive the coarse value if needed (matches Eq. 9
        # in the paper; with the Dirichlet enforcement above this is a no-op
        # for valid pixels but keeps behaviour explicit).
        return sparse_mask * anchor_vals + (1.0 - sparse_mask) * d_t
