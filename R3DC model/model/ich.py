"""Indoor Calibration Head (ICH).

A lightweight, 3-layer MLP used by the NYU Depth V2 variant to map the
*relative* depth priors of a frozen foundation backbone (e.g., Depth Anything
V2 ViT-S) onto a calibrated metric range. The total trainable parameter count
is 16,642 — less than 0.018% of the 94.6M backbone — yet contributes the bulk
of the headline NYU gain.

The ICH predicts dataset-specific scale ``s`` and shift ``b`` from globally
pooled features and applies an affine transform to the per-pixel depth in
log-normalised space:

.. math::

    \\tilde{d}_{\\text{cal}}(p) = \\text{clip}\\bigl(s \\cdot \\tilde{d}(p) + b,\\, 0, 1\\bigr).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class IndoorCalibrationHead(nn.Module):
    """Per-image scale + shift predictor.

    Args:
        feature_dim: channels of the pooled backbone feature.
        hidden_dim: width of the MLP hidden layers.
        scale_init: initialisation for the scale (centred on 1.0).
        shift_init: initialisation for the shift (centred on 0.0).
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        scale_init: float = 1.0,
        shift_init: float = 0.0,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        # Initialise the final linear bias to (scale_init, shift_init) so that
        # at step 0 the ICH is approximately the identity.
        nn.init.zeros_(self.mlp[-1].weight)
        with torch.no_grad():
            self.mlp[-1].bias.copy_(torch.tensor([scale_init, shift_init]))

    @torch.no_grad()
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, depth: torch.Tensor, pooled_feature: torch.Tensor) -> torch.Tensor:
        """Apply the ICH affine transform.

        Args:
            depth: per-pixel depth in log-normalised [0, 1] space, ``(B, 1, H, W)``.
            pooled_feature: globally pooled backbone feature, ``(B, feature_dim)``.

        Returns:
            Calibrated depth, ``(B, 1, H, W)``, clipped to ``[0, 1]``.
        """
        params = self.mlp(pooled_feature)             # (B, 2)
        scale = params[:, 0].view(-1, 1, 1, 1)
        shift = params[:, 1].view(-1, 1, 1, 1)
        out = scale * depth + shift
        return out.clamp(0.0, 1.0)
