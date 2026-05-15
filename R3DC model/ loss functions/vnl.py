"""Virtual Normal Loss (Yin et al., 2019).

Samples random point triplets on the predicted depth surface and penalises
back-facing normals. We use a simple 3-D reconstruction in normalised pixel
coordinates, which does not require true camera intrinsics — the sign of
``n_z`` is what we constrain, not its absolute scale.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VirtualNormalLoss(nn.Module):
    """Penalises predicted surface triangles whose normal points away from the camera.

    Args:
        num_samples: number of triangles to sample per image.
        eps: numerical floor for normalisation.
    """

    def __init__(self, num_samples: int = 5000, eps: float = 1e-6):
        super().__init__()
        if num_samples < 3:
            raise ValueError("num_samples must be >= 3.")
        self.num_samples = num_samples
        self.eps = eps

    def forward(self, pred_depth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Args:
            pred_depth: predicted depth, ``(B, 1, H, W)``, in any positive scale.
            mask: validity mask, ``(B, 1, H, W)``.
        """
        b, _, h, w = pred_depth.shape
        device = pred_depth.device
        # Build normalised pixel coordinates in [-1, 1].
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device),
            indexing="ij",
        )
        losses = []
        for bi in range(b):
            valid = mask[bi, 0] > 0.5
            valid_idx = torch.nonzero(valid, as_tuple=False)
            if valid_idx.shape[0] < self.num_samples * 3:
                continue  # not enough valid points
            # Sample 3*N indices.
            sel = torch.randint(0, valid_idx.shape[0], (self.num_samples * 3,), device=device)
            pts = valid_idx[sel].view(self.num_samples, 3, 2)  # (N, 3, 2) row,col
            r = pts[..., 0]
            c = pts[..., 1]
            z = pred_depth[bi, 0][r, c]                        # (N, 3)
            x = xs[r, c] * z
            y = ys[r, c] * z
            p = torch.stack([x, y, z], dim=-1)                  # (N, 3, 3)
            v1 = p[:, 1] - p[:, 0]
            v2 = p[:, 2] - p[:, 0]
            normal = torch.cross(v1, v2, dim=-1)
            norm = normal.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            nz = normal[..., 2:3] / norm
            # Encourage z-component to be positive (camera-facing).
            losses.append(torch.relu(-nz).mean())
        if not losses:
            return pred_depth.sum() * 0.0
        return torch.stack(losses).mean()
