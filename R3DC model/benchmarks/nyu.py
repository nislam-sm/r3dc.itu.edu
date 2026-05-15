"""NYU Depth V2 dataset adapter.

Supports two storage formats commonly used in the depth-estimation literature:

1. The official MAT v7.3 dump (``nyu_depth_v2_labeled.mat``) - read via h5py.
2. Pre-extracted .npy / .png pairs in a directory tree, listed by a manifest.

The R3DC+ICH variant for NYU samples sparse anchors uniformly at low density
($\\sim 0.1\\%$) from the structured-light ground truth to mimic a sparse
sensor; the sampling density is configurable.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _sample_uniform_sparse(
    depth: np.ndarray, density: float, rng: np.random.Generator
) -> np.ndarray:
    """Bernoulli-sample a sparse anchor mask from a dense depth map."""
    valid = depth > 0
    sample = rng.random(size=depth.shape) < density
    return (valid & sample).astype(np.float32)


@dataclass
class NYUPaths:
    rgb: str  # path or "mat:<idx>" for the .mat backend
    depth: str


class NYUDepthV2(Dataset):
    """NYU Depth V2 dataset for the R3DC+ICH indoor variant.

    Args:
        root: directory containing extracted RGB / depth files (used if
            ``mat_path`` is None).
        mat_path: alternative path to the .mat v7.3 dataset dump.
        manifest: optional CSV with rgb,depth columns (overrides root).
        split: "train" or "val"; only used when scanning `root`.
        density: fraction of GT pixels exposed as sparse anchors (paper: 0.001).
        transform: callable applied to each sample.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        mat_path: Optional[str] = None,
        manifest: Optional[str] = None,
        split: str = "train",
        density: float = 0.001,
        transform: Optional[Callable] = None,
        d_min: float = 0.001,
        d_max: float = 10.0,
        seed: int = 0,
    ):
        if root is None and mat_path is None and manifest is None:
            raise ValueError("NYU needs `root`, `mat_path`, or `manifest`")
        self.transform = transform
        self.density = density
        self.d_min = d_min
        self.d_max = d_max
        self._rng = np.random.default_rng(seed)
        self._mat_handle = None
        self.paths: List[NYUPaths] = []

        if mat_path is not None:
            self._init_mat(mat_path, split)
        elif manifest is not None:
            self._init_manifest(manifest)
        else:
            self._init_root(Path(root), split)

    # ------------------------------------------------------------------ #
    def _init_manifest(self, manifest: str) -> None:
        with open(manifest, newline="") as fh:
            for row in csv.reader(fh):
                if not row or row[0].startswith("#"):
                    continue
                if len(row) < 2:
                    raise ValueError(f"Bad NYU manifest row: {row!r}")
                self.paths.append(NYUPaths(rgb=row[0], depth=row[1]))

    def _init_root(self, root: Path, split: str) -> None:
        split_dir = root / split
        if split_dir.exists():
            rgb_dir = split_dir / "rgb"
            depth_dir = split_dir / "depth"
        else:
            rgb_dir = root / "rgb"
            depth_dir = root / "depth"
        if not (rgb_dir.exists() and depth_dir.exists()):
            return
        for rgb in sorted(rgb_dir.glob("*.png")):
            depth = depth_dir / f"{rgb.stem}.png"
            if depth.exists():
                self.paths.append(NYUPaths(rgb=str(rgb), depth=str(depth)))

    def _init_mat(self, mat_path: str, split: str) -> None:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required to read the NYU .mat dump") from exc
        self._mat_path = mat_path
        with h5py.File(mat_path, "r") as fh:
            n = fh["images"].shape[0]
        # Reproduce the canonical 80/20 split deterministically.
        rng = np.random.default_rng(0xBEEF)
        idx = np.arange(n)
        rng.shuffle(idx)
        cut = int(0.8 * n)
        chosen = idx[:cut] if split == "train" else idx[cut:]
        self.paths = [
            NYUPaths(rgb=f"mat:{i}", depth=f"mat:{i}") for i in chosen.tolist()
        ]

    # ------------------------------------------------------------------ #
    def _read_mat_sample(self, raw: str) -> Tuple[np.ndarray, np.ndarray]:
        import h5py

        if self._mat_handle is None:
            self._mat_handle = h5py.File(self._mat_path, "r")
        i = int(raw.split(":", 1)[1])
        # NYU .mat stores images as [3, W, H] and depths as [W, H].
        img = np.array(self._mat_handle["images"][i]).astype(np.float32) / 255.0
        depth = np.array(self._mat_handle["depths"][i]).astype(np.float32)
        # Transpose to [H, W, 3] and [H, W].
        img = np.transpose(img, (2, 1, 0))
        depth = depth.T
        return img, depth

    def _read_file_sample(self, p: NYUPaths) -> Tuple[np.ndarray, np.ndarray]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required for NYU file backend") from exc

        rgb = np.array(Image.open(p.rgb).convert("RGB"), dtype=np.float32) / 255.0
        depth = np.array(Image.open(p.depth), dtype=np.float32)
        # NYU pngs are commonly stored as mm; convert if values look like that.
        if depth.max() > 25.0:
            depth = depth / 1000.0
        return rgb, depth

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        if p.rgb.startswith("mat:"):
            rgb, depth = self._read_mat_sample(p.rgb)
        else:
            rgb, depth = self._read_file_sample(p)

        depth = np.clip(depth, 0.0, self.d_max).astype(np.float32)
        sparse_mask = _sample_uniform_sparse(depth, self.density, self._rng)
        sparse_depth = depth * sparse_mask
        valid_mask = (depth > self.d_min).astype(np.float32)

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
