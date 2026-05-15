"""Top-level R3DC model.

Wires together the encoder, transformer bottleneck, FPN decoder, output heads,
and the reliability-gated CSPN++ refiner. Returns a dictionary with the keys:

* ``coarse_depth``     — D_0, in log-normalised [0, 1] space.
* ``depth``            — refined D_1, in log-normalised [0, 1] space.
* ``reliability``      — R̂ in [0, 1].
* ``uncertainty``      — σ̂ > 0 (Laplace scale).
* ``aux_depth_half``   — auxiliary head output at 1/2 resolution.
* ``aux_depth_quarter``— auxiliary head output at 1/4 resolution.

The model is agnostic of metric scale: use :func:`r3dc.utils.lognorm.log_normalize`
to map raw sparse depth into [0, 1] before the forward pass, and the inverse
when evaluating metric metrics downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from r3dc.models.bottleneck import TransformerBottleneck
from r3dc.models.cspn import CSPNRefiner
from r3dc.models.decoder import FPNDecoder
from r3dc.models.encoder import DualStreamEncoder
from r3dc.models.heads import AuxDepthHead, DepthHead, ReliabilityHead, UncertaintyHead


@dataclass
class R3DCConfig:
    """Hyperparameters for the outdoor R3DC variant."""

    base_channels: int = 64
    depth_in_ch: int = 2          # [normalised_depth, mask]
    use_cma: bool = True
    n_max: int = 512              # CMA / transformer token cap
    drop_path: float = 0.1
    transformer_heads: int = 8
    transformer_mlp_ratio: float = 4.0
    cspn_iterations: int = 6
    cspn_neighbour_weight: float = 0.8


class R3DC(nn.Module):
    """Reveal-to-Revise depth completion network.

    Args:
        variant: ``"outdoor"`` for the dual-stream model used on KITTI,
            VisDrone, and Drone-Videos, or ``"indoor_adapter"`` for use
            with a frozen foundation backbone via :class:`IndoorCalibrationHead`.
            For the indoor adapter, construct :class:`r3dc.models.ich.IndoorCalibrationHead`
            externally and pair it with your backbone.
        base_channels: width ``B``; channels scale as ``{B/2, B, 2B, 4B}``.
        **kwargs: forwarded to :class:`R3DCConfig`.
    """

    def __init__(self, variant: str = "outdoor", **kwargs):
        super().__init__()
        if variant != "outdoor":
            raise ValueError(
                "R3DC currently supports `variant='outdoor'`. For the NYU "
                "indoor variant, wrap a frozen DA-V2 backbone with "
                "`r3dc.models.ich.IndoorCalibrationHead` separately."
            )
        cfg = R3DCConfig(**kwargs)
        self.cfg = cfg
        B = cfg.base_channels

        # ---- Encoder ----
        self.encoder = DualStreamEncoder(
            base_channels=B,
            depth_in_ch=cfg.depth_in_ch,
            use_cma=cfg.use_cma,
            n_max=cfg.n_max,
            drop_path=cfg.drop_path,
        )
        enc_ch = self.encoder.channels  # [B/2, B, 2B, 4B]

        # ---- Bottleneck (1/16 scale) ----
        # Downsample the deepest stage from 1/8 -> 1/16 before the transformer.
        self.bottleneck_down = nn.Sequential(
            nn.Conv2d(enc_ch[3], 2 * enc_ch[3], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * enc_ch[3]),
            nn.ReLU(inplace=True),
        )
        bn_ch = 2 * enc_ch[3]  # 8B
        self.bottleneck = TransformerBottleneck(
            channels=bn_ch,
            num_heads=cfg.transformer_heads,
            mlp_ratio=cfg.transformer_mlp_ratio,
            drop_path=cfg.drop_path,
            n_max=cfg.n_max,
        )

        # ---- FPN decoder ----
        # Output widths at scales 1/8, 1/4, 1/2, 1 are 4B, 2B, B, B/2.
        dec_widths = [enc_ch[3], enc_ch[2], enc_ch[1], enc_ch[0]]
        self.decoder = FPNDecoder(
            bottleneck_ch=bn_ch,
            depth_skip_ch=enc_ch,
            rgb_skip_ch=enc_ch,
            out_channels=dec_widths,
            n_max=cfg.n_max,
            drop_path=cfg.drop_path,
        )

        # ---- Heads ----
        self.depth_head = DepthHead(enc_ch[0])
        self.rel_head = ReliabilityHead(enc_ch[0])
        self.unc_head = UncertaintyHead(enc_ch[0])
        # Auxiliary heads at 1/2 and 1/4 for deep supervision.
        self.aux_half_head = AuxDepthHead(enc_ch[1])
        self.aux_quarter_head = AuxDepthHead(enc_ch[2])

        # ---- CSPN++ refiner ----
        self.refiner = CSPNRefiner(
            feat_channels=enc_ch[0],
            iterations=cfg.cspn_iterations,
            neighbour_total_weight=cfg.cspn_neighbour_weight,
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))

    # ------------------------------------------------------------------
    def forward(
        self,
        rgb: torch.Tensor,
        sparse_depth: torch.Tensor,
        sparse_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            rgb: ``(B, 3, H, W)``, values in ``[0, 1]``.
            sparse_depth: ``(B, 1, H, W)``, log-normalised sparse depth.
            sparse_mask: ``(B, 1, H, W)``, binary mask (1 = valid). If ``None``,
                it is derived as ``sparse_depth > 0``.
        """
        if sparse_mask is None:
            sparse_mask = (sparse_depth > 0).float()

        # Pack depth stream input as [normalised_depth, mask].
        depth_in = torch.cat([sparse_depth, sparse_mask], dim=1)

        # Encoder
        enc = self.encoder(rgb, depth_in)

        # Bottleneck
        bn = self.bottleneck(self.bottleneck_down(enc.depth[3]))

        # Decoder
        aux_feats, full_feat = self.decoder(bn, enc.depth, enc.rgb)
        # aux_feats: [1/8, 1/4, 1/2]

        # Heads (full resolution)
        d0 = self.depth_head(full_feat)            # coarse depth in [0,1]
        rel = self.rel_head(full_feat)             # reliability in [0,1]
        unc = self.unc_head(full_feat)             # uncertainty > 0

        # Auxiliary depth predictions
        aux_quarter = self.aux_quarter_head(aux_feats[1])
        aux_half = self.aux_half_head(aux_feats[2])

        # Reliability-gated CSPN++ refinement
        d1 = self.refiner(d0, full_feat, rel, sparse_depth, sparse_mask)

        return {
            "coarse_depth": d0,
            "depth": d1,
            "reliability": rel,
            "uncertainty": unc,
            "aux_depth_half": aux_half,
            "aux_depth_quarter": aux_quarter,
        }
