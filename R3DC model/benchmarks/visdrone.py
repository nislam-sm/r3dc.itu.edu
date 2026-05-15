"""VisDrone aerial depth-completion benchmark adapter.

VisDrone only ships detection annotations; depth ground-truth is synthesised
on-the-fly using the physics-motivated aerial prior of Appendix J.

To avoid recomputing the synthetic GT each epoch, the loader caches
``(depth, sparse_mask)`` pairs to disk on first access keyed by image path
and a deterministic seed derived from the filename. Pass ``cache_dir=None``
to disable caching.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .synthetic import AerialPriorConfig, make_aerial_depth


def _read_rgb(path: str) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for VisDrone") from exc
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _hash_seed(path: str, salt: int) -> int:
    h = hashlib.sha1(f"{path}|{salt}".encode()).hexdigest()
    return int(h[:8], 16)


@dataclass
class VisDronePaths:
    rgb: str
    cache: Optional[str] = None


class VisDroneDataset(Dataset):
    """VisDrone with on-the-fly synthetic depth (paper Appendix J).

    Args:
        root: VisDrone root containing ``VisDrone2019-DET-train/`` etc.
        split: "train" or "val".
        list_file: optional .txt listing one image path per line.
        cache_dir: directory for cached synthetic GT; None disables caching.
        prior_config: aerial prior hyper-parameters.
        seed_salt: salt for the per-image deterministic seed.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        split: str = "train",
        list_file: Optional[str] = None,
        cache_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        prior_config: Optional[AerialPriorConfig] = None,
        d_min: float = 1.0,
        d_max: float = 80.0,
        seed_salt: int = 0,
    ):
        if root is None and list_file is None:
            raise ValueError("VisDrone needs `root` or `list_file`")
        self.transform = transform
        self.prior_config = prior_config or AerialPriorConfig(
            d_min=d_min, d_max=d_max
        )
        self.d_min = d_min
        self.d_max = d_max
        self.seed_salt = seed_salt

        if list_file is not None:
            with open(list_file) as fh:
                rgbs = [ln.strip() for ln in fh if ln.strip()]
        else:
            split_dir = (
                Path(root) / ("VisDrone2019-DET-train" if split == "train"
                               else "VisDrone2019-DET-val")
            )
            img_dir = split_dir / "images"
            if not img_dir.exists():  # fall back to the bare layout
                img_dir = split_dir
            rgbs = sorted(str(p) for p in img_dir.glob("*.jpg"))

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.paths: List[VisDronePaths] = [
            VisDronePaths(
                rgb=rgb,
                cache=str(self.cache_dir / f"{Path(rgb).stem}.npz")
                if self.cache_dir
                else None,
            )
            for rgb in rgbs
        ]

    def __len__(self) -> int:
        return len(self.paths)

    # ------------------------------------------------------------------ #
    def _load_or_make(self, rgb_uint8: np.ndarray, p: VisDronePaths):
        if p.cache and os.path.exists(p.cache):
            data = np.load(p.cache)
            return data["depth"], data["mask"]
        h, w = rgb_uint8.shape[:2]
        rng = np.random.default_rng(_hash_seed(p.rgb, self.seed_salt))
        depth, mask = make_aerial_depth(h, w, rgb_uint8, self.prior_config, rng)
        if p.cache:
            np.savez_compressed(p.cache, depth=depth.astype(np.float32), mask=mask)
        return depth, mask

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        rgb_u8 = _read_rgb(p.rgb)
        depth, sparse_mask_bool = self._load_or_make(rgb_u8, p)

        sparse_mask = sparse_mask_bool.astype(np.float32)
        sparse_depth = depth * sparse_mask
        valid_mask = (depth >= self.d_min).astype(np.float32)
        rgb = rgb_u8.astype(np.float32) / 255.0

        sample = {
            "image": torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float(),
            "sparse_depth": torch.from_numpy(sparse_depth).unsqueeze(0).float(),
            "sparse_mask": torch.from_numpy(sparse_mask).unsqueeze(0).float(),
            "depth": torch.from_numpy(depth).unsqueeze(0).float(),
            "valid_mask": torch.from_numpy(valid_mask).unsqueeze(0).float(),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
