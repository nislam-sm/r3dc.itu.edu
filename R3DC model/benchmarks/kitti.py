"""KITTI Depth Completion dataset adapter.

Expects the official KITTI Depth Completion layout::

    <root>/
        data_depth_velodyne/
            train/<seq>/proj_depth/velodyne_raw/image_02/*.png
        data_depth_annotated/
            train/<seq>/proj_depth/groundtruth/image_02/*.png
        raw_data/
            <seq>/image_02/data/*.png

Each PNG depth file is uint16 with depth_m = value / 256.

If the official split is not available locally, the user can also point
``manifest`` at a CSV / text file listing one ``rgb_path,sparse_path,gt_path``
triplet per line - this is convenient for the mini-split (6732/749) used in
the paper's exploratory experiments.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


KITTI_DEPTH_SCALE = 256.0  # uint16 -> metres


@dataclass
class KITTIPaths:
    rgb: str
    sparse: str
    gt: str


def _read_depth_png(path: str) -> np.ndarray:
    """Read a KITTI 16-bit depth PNG into a float32 metre map (0 = invalid)."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to read KITTI depth PNGs") from exc

    arr = np.array(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr / KITTI_DEPTH_SCALE


def _read_rgb(path: str) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to read RGB images") from exc

    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return img


class KITTIDepthCompletion(Dataset):
    """KITTI Depth Completion dataset.

    Args:
        root: filesystem path to the KITTI depth-completion root.
        split: one of "train", "val", "test".
        manifest: optional path to a CSV with rgb,sparse,gt columns.
        transform: callable applied to the assembled sample dict.
        d_min, d_max: clipping range used downstream by losses / log-norm.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        split: str = "train",
        manifest: Optional[str] = None,
        transform: Optional[Callable] = None,
        d_min: float = 0.0,
        d_max: float = 80.0,
    ):
        if root is None and manifest is None:
            raise ValueError("KITTI dataset requires either `root` or `manifest`")
        self.root = Path(root) if root is not None else None
        self.split = split
        self.transform = transform
        self.d_min = d_min
        self.d_max = d_max

        self.paths: List[KITTIPaths] = (
            self._load_manifest(manifest)
            if manifest is not None
            else self._scan_official_split(self.root, split)
        )

    # ------------------------------------------------------------------ #
    # discovery
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_manifest(path: str) -> List[KITTIPaths]:
        out: List[KITTIPaths] = []
        with open(path, newline="") as fh:
            for row in csv.reader(fh):
                if not row or row[0].startswith("#"):
                    continue
                if len(row) < 3:
                    raise ValueError(
                        f"Bad manifest row (need rgb,sparse,gt): {row!r}"
                    )
                out.append(KITTIPaths(rgb=row[0], sparse=row[1], gt=row[2]))
        return out

    @staticmethod
    def _scan_official_split(root: Path, split: str) -> List[KITTIPaths]:
        if split not in {"train", "val"}:
            return []
        ann_root = root / "data_depth_annotated" / split
        vel_root = root / "data_depth_velodyne" / split
        raw_root = root / "raw_data"
        if not ann_root.exists():
            return []
        out: List[KITTIPaths] = []
        for seq_dir in sorted(ann_root.iterdir()):
            if not seq_dir.is_dir():
                continue
            seq = seq_dir.name
            gt_dir = seq_dir / "proj_depth" / "groundtruth" / "image_02"
            vel_dir = vel_root / seq / "proj_depth" / "velodyne_raw" / "image_02"
            date = seq[:10]
            rgb_dir = raw_root / date / seq / "image_02" / "data"
            if not (gt_dir.exists() and vel_dir.exists() and rgb_dir.exists()):
                continue
            for gt_path in sorted(gt_dir.glob("*.png")):
                stem = gt_path.stem
                sparse = vel_dir / f"{stem}.png"
                rgb = rgb_dir / f"{stem}.png"
                if sparse.exists() and rgb.exists():
                    out.append(
                        KITTIPaths(rgb=str(rgb), sparse=str(sparse), gt=str(gt_path))
                    )
        return out

    # ------------------------------------------------------------------ #
    # torch interface
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        rgb = _read_rgb(p.rgb)
        sparse = _read_depth_png(p.sparse)
        gt = _read_depth_png(p.gt)

        # Crop RGB to depth shape if needed.
        if rgb.shape[:2] != sparse.shape:
            h = min(rgb.shape[0], sparse.shape[0])
            w = min(rgb.shape[1], sparse.shape[1])
            rgb = rgb[:h, :w]
            sparse = sparse[:h, :w]
            gt = gt[:h, :w]

        sparse_mask = (sparse > 0).astype(np.float32)
        valid_mask = (gt > 0).astype(np.float32)

        sample = {
            "image": torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float(),
            "sparse_depth": torch.from_numpy(sparse).unsqueeze(0).float(),
            "sparse_mask": torch.from_numpy(sparse_mask).unsqueeze(0).float(),
            "depth": torch.from_numpy(gt).unsqueeze(0).float(),
            "valid_mask": torch.from_numpy(valid_mask).unsqueeze(0).float(),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
