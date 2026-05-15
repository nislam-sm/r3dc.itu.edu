"""Augmentation pipeline used across all R3DC datasets.

Implements the augmentations described in Sec. 5.2 of the paper:

* horizontal flip (p = 0.5)
* color jitter (brightness / contrast / saturation, mild)
* gamma adjustment, gamma ~ U(0.8, 1.2)
* sparse-input dropout (p = 0.3, drops a fraction of sparse anchors)
* CutMix (p = 0.3, lambda ~ U(0.3, 0.7))
* resize / centre-crop to a fixed (H, W)

All transforms accept a `sample` dict with keys:

    image          : float tensor [3, H, W], in [0, 1]
    sparse_depth   : float tensor [1, H, W], metres
    sparse_mask    : float tensor [1, H, W], in {0, 1}
    depth          : float tensor [1, H, W], metres (ground truth)
    valid_mask     : float tensor [1, H, W], in {0, 1}

Augmentations operate in-place on copies so the underlying dataset stays
deterministic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


SampleDict = Dict[str, torch.Tensor]


# -----------------------------------------------------------------------------
# Geometric
# -----------------------------------------------------------------------------
class Resize:
    """Bilinear resize for the image, nearest for masks / sparse depth."""

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, sample: SampleDict) -> SampleDict:
        h, w = self.size
        out = dict(sample)
        for key, mode in (
            ("image", "bilinear"),
            ("depth", "nearest"),
            ("sparse_depth", "nearest"),
            ("sparse_mask", "nearest"),
            ("valid_mask", "nearest"),
        ):
            if key not in out:
                continue
            t = out[key].unsqueeze(0)  # [1, C, H, W]
            kwargs = {"mode": mode}
            if mode == "bilinear":
                kwargs["align_corners"] = False
            out[key] = F.interpolate(t, size=(h, w), **kwargs).squeeze(0)
        return out


class CenterCrop:
    """Centre-crop all tensors to ``size``."""

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, sample: SampleDict) -> SampleDict:
        th, tw = self.size
        any_key = next(iter(sample))
        _, h, w = sample[any_key].shape
        y0 = max((h - th) // 2, 0)
        x0 = max((w - tw) // 2, 0)
        return {k: v[:, y0 : y0 + th, x0 : x0 + tw] for k, v in sample.items()}


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: SampleDict) -> SampleDict:
        if random.random() >= self.p:
            return sample
        return {k: torch.flip(v, dims=[-1]) for k, v in sample.items()}


# -----------------------------------------------------------------------------
# Photometric
# -----------------------------------------------------------------------------
class ColorJitter:
    """Simple brightness / contrast / saturation jitter for in-[0,1] images."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, sample: SampleDict) -> SampleDict:
        if "image" not in sample:
            return sample
        img = sample["image"].clone()
        if self.brightness > 0:
            img = img * (1.0 + random.uniform(-self.brightness, self.brightness))
        if self.contrast > 0:
            mean = img.mean(dim=(1, 2), keepdim=True)
            img = (img - mean) * (
                1.0 + random.uniform(-self.contrast, self.contrast)
            ) + mean
        if self.saturation > 0:
            gray = img.mean(dim=0, keepdim=True)
            img = gray + (img - gray) * (
                1.0 + random.uniform(-self.saturation, self.saturation)
            )
        sample = dict(sample)
        sample["image"] = img.clamp(0.0, 1.0)
        return sample


class RandomGamma:
    """Gamma correction with gamma ~ U(lo, hi)."""

    def __init__(self, lo: float = 0.8, hi: float = 1.2):
        self.lo = lo
        self.hi = hi

    def __call__(self, sample: SampleDict) -> SampleDict:
        if "image" not in sample:
            return sample
        gamma = random.uniform(self.lo, self.hi)
        sample = dict(sample)
        sample["image"] = sample["image"].clamp_min(1e-8).pow(gamma).clamp(0.0, 1.0)
        return sample


# -----------------------------------------------------------------------------
# Sparse-input augmentation
# -----------------------------------------------------------------------------
class SparseDropout:
    """Randomly drop a fraction of the sparse anchors.

    The paper applies this with probability 0.3 per sample; ``drop_prob`` is the
    drop probability per anchor when triggered.
    """

    def __init__(self, p_trigger: float = 0.3, drop_prob: float = 0.5):
        self.p_trigger = p_trigger
        self.drop_prob = drop_prob

    def __call__(self, sample: SampleDict) -> SampleDict:
        if random.random() >= self.p_trigger:
            return sample
        if "sparse_mask" not in sample:
            return sample
        sample = dict(sample)
        mask = sample["sparse_mask"]
        keep = (torch.rand_like(mask) > self.drop_prob).float()
        new_mask = mask * keep
        sample["sparse_mask"] = new_mask
        if "sparse_depth" in sample:
            sample["sparse_depth"] = sample["sparse_depth"] * new_mask
        return sample


# -----------------------------------------------------------------------------
# CutMix
# -----------------------------------------------------------------------------
class CutMix:
    """CutMix between samples of the same batch (applied at collate time).

    For convenience this transform is implemented as a callable that takes a
    *batch* of samples (list of dicts) rather than a single sample.
    """

    def __init__(self, p_trigger: float = 0.3, lam_lo: float = 0.3, lam_hi: float = 0.7):
        self.p_trigger = p_trigger
        self.lam_lo = lam_lo
        self.lam_hi = lam_hi

    def __call__(self, batch: Sequence[SampleDict]) -> Sequence[SampleDict]:
        if len(batch) < 2 or random.random() >= self.p_trigger:
            return batch
        out = [dict(s) for s in batch]
        lam = random.uniform(self.lam_lo, self.lam_hi)
        _, h, w = batch[0]["image"].shape
        rh = int(h * (1.0 - lam) ** 0.5)
        rw = int(w * (1.0 - lam) ** 0.5)
        if rh <= 0 or rw <= 0:
            return batch
        cy = random.randint(0, h - rh)
        cx = random.randint(0, w - rw)
        for i in range(len(out)):
            j = (i + 1) % len(out)
            for key in out[i]:
                src = batch[j][key]
                out[i][key] = out[i][key].clone()
                out[i][key][:, cy : cy + rh, cx : cx + rw] = src[
                    :, cy : cy + rh, cx : cx + rw
                ]
        return out


# -----------------------------------------------------------------------------
# Composition
# -----------------------------------------------------------------------------
@dataclass
class TrainAugmentConfig:
    """Configuration for the full training-time augmentation stack."""

    size: Optional[Tuple[int, int]] = None  # resize target, or None
    hflip: float = 0.5
    color_jitter: float = 0.2
    gamma_lo: float = 0.8
    gamma_hi: float = 1.2
    sparse_dropout: float = 0.3
    sparse_dropout_rate: float = 0.5


class TrainTransform:
    """End-to-end training augmentation (per-sample, before collate)."""

    def __init__(self, cfg: Optional[TrainAugmentConfig] = None):
        self.cfg = cfg or TrainAugmentConfig()
        self._steps = []
        if self.cfg.size is not None:
            self._steps.append(Resize(self.cfg.size))
        self._steps.extend(
            [
                RandomHorizontalFlip(self.cfg.hflip),
                ColorJitter(
                    self.cfg.color_jitter,
                    self.cfg.color_jitter,
                    self.cfg.color_jitter,
                ),
                RandomGamma(self.cfg.gamma_lo, self.cfg.gamma_hi),
                SparseDropout(self.cfg.sparse_dropout, self.cfg.sparse_dropout_rate),
            ]
        )

    def __call__(self, sample: SampleDict) -> SampleDict:
        for s in self._steps:
            sample = s(sample)
        return sample


class EvalTransform:
    """Deterministic test/val transform: optional resize only."""

    def __init__(self, size: Optional[Tuple[int, int]] = None):
        self.size = size
        self._resize = Resize(size) if size is not None else None

    def __call__(self, sample: SampleDict) -> SampleDict:
        if self._resize is not None:
            sample = self._resize(sample)
        return sample
