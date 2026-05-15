"""Seven-term composite objective with auxiliary deep supervision.

Implements Eq. (10) in the paper:

.. math::

    \\mathcal{L} = 1.00\\,\\mathcal{L}_{SI}
                + 0.60\\,\\mathcal{L}_{FB}
                + 0.20\\,\\mathcal{L}_{SSIM}
                + 0.15\\,\\mathcal{L}_{Anc}
                + 0.10\\,\\mathcal{L}_{VNL}
                + 0.05\\,\\mathcal{L}_{DNC}
                + 0.05\\,\\mathcal{L}_{Grad}
                + 0.05\\,\\mathcal{L}_{UNC}
                + 0.10\\,\\mathcal{L}_{Aux}.

Weights are determined by a grid search on a held-out KITTI mini-split and
are kept *fixed across all four datasets*, which is a key cross-domain
contribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from r3dc.losses.anchor import SparseAnchorLoss
from r3dc.losses.dnc import DepthNormalConsistencyLoss
from r3dc.losses.focal_berhu import FocalBerHuLoss
from r3dc.losses.silog import SILogLoss
from r3dc.losses.ssim import SSIMLoss
from r3dc.losses.uncertainty import GradientConsistencyLoss, LaplaceUncertaintyLoss
from r3dc.losses.vnl import VirtualNormalLoss


@dataclass
class LossWeights:
    """Default loss weights from the paper. Kept identical across datasets."""

    silog: float = 1.00
    focal_berhu: float = 0.60
    ssim: float = 0.20
    anchor: float = 0.15
    vnl: float = 0.10
    dnc: float = 0.05
    grad: float = 0.05
    uncertainty: float = 0.05
    aux: float = 0.10


class CompositeLoss(nn.Module):
    """Aggregates the seven main losses and auxiliary supervision.

    The forward signature mirrors the dictionary returned by :class:`R3DC`:

    >>> loss_fn = CompositeLoss()
    >>> outputs = model(rgb, sparse_depth, mask)
    >>> total, parts = loss_fn(outputs, sparse_depth, mask, gt_depth, gt_mask)
    """

    def __init__(self, weights: Optional[LossWeights] = None,
                 vnl_samples: int = 5000, ssim_window: int = 7):
        super().__init__()
        self.weights = weights or LossWeights()
        self.silog = SILogLoss()
        self.focal_berhu = FocalBerHuLoss()
        self.ssim = SSIMLoss(window_size=ssim_window)
        self.anchor = SparseAnchorLoss()
        self.vnl = VirtualNormalLoss(num_samples=vnl_samples)
        self.dnc = DepthNormalConsistencyLoss()
        self.grad = GradientConsistencyLoss()
        self.unc = LaplaceUncertaintyLoss()

    # ------------------------------------------------------------------
    def _aux_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        gt_depth: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Focal-BerHu at half- and quarter-resolution exits."""
        losses = []
        for key, scale in (("aux_depth_half", 0.5), ("aux_depth_quarter", 0.25)):
            if key not in outputs:
                continue
            pred = outputs[key]
            target = F.interpolate(gt_depth, size=pred.shape[-2:], mode="bilinear",
                                   align_corners=False)
            mask = F.interpolate(gt_mask, size=pred.shape[-2:], mode="nearest")
            losses.append(self.focal_berhu(pred, target, mask))
        if not losses:
            return torch.zeros((), device=gt_depth.device, dtype=gt_depth.dtype)
        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        sparse_depth: torch.Tensor,
        sparse_mask: torch.Tensor,
        gt_depth: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the composite loss and return per-term contributions."""
        d_pred = outputs["depth"]
        rel = outputs.get("reliability")  # noqa: F841 — currently informational
        unc = outputs["uncertainty"]

        l_si = self.silog(d_pred, gt_depth, gt_mask)
        l_fb = self.focal_berhu(d_pred, gt_depth, gt_mask)
        l_ssim = self.ssim(d_pred, gt_depth, gt_mask)
        l_anc = self.anchor(d_pred, sparse_depth, sparse_mask)
        l_vnl = self.vnl(d_pred.clamp_min(1e-3), gt_mask)
        l_dnc = self.dnc(d_pred, gt_mask)
        l_grad = self.grad(d_pred, gt_depth, gt_mask)
        l_unc = self.unc(d_pred, gt_depth, unc, gt_mask)
        l_aux = self._aux_loss(outputs, gt_depth, gt_mask)

        w = self.weights
        total = (
            w.silog * l_si
            + w.focal_berhu * l_fb
            + w.ssim * l_ssim
            + w.anchor * l_anc
            + w.vnl * l_vnl
            + w.dnc * l_dnc
            + w.grad * l_grad
            + w.uncertainty * l_unc
            + w.aux * l_aux
        )
        parts = {
            "loss/total": total.detach(),
            "loss/silog": l_si.detach(),
            "loss/focal_berhu": l_fb.detach(),
            "loss/ssim": l_ssim.detach(),
            "loss/anchor": l_anc.detach(),
            "loss/vnl": l_vnl.detach(),
            "loss/dnc": l_dnc.detach(),
            "loss/grad": l_grad.detach(),
            "loss/uncertainty": l_unc.detach(),
            "loss/aux": l_aux.detach(),
        }
        return total, parts
