"""Single-image R3DC inference.

Loads an RGB image and a sparse depth map, runs R3DC, and writes the dense
depth, reliability, and (optionally) uncertainty maps to disk.

Usage::

    python -m scripts.infer \\
        --config r3dc/configs/kitti.yaml \\
        --checkpoint runs/kitti/best.pt \\
        --rgb data/000123.png \\
        --sparse data/000123_sparse.png \\
        --output-dir runs/infer

If ``--sparse`` is omitted, a uniform random sparse mask at the configured
density is drawn from a small fraction of pixels to allow quick demos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from r3dc.utils import (
    colorize_depth,
    colorize_reliability,
    load_checkpoint,
    log_denormalize,
    log_normalize,
    seed_everything,
)

from scripts.train import build_model


def _read_rgb(path: str) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _read_depth(path: str) -> np.ndarray:
    from PIL import Image

    arr = np.array(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.max() > 80.0:
        arr = arr / 256.0  # KITTI-style uint16 mm/256
    return arr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--sparse", default=None)
    parser.add_argument("--output-dir", default="runs/infer")
    parser.add_argument("--demo-density", type=float, default=0.02,
                        help="If --sparse is omitted, density of synthetic input.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rgb = _read_rgb(args.rgb)
    h, w = rgb.shape[:2]
    d_min = float(cfg["dataset"].get("d_min", 0.0))
    d_max = float(cfg["dataset"].get("d_max", 80.0))

    if args.sparse:
        sparse = _read_depth(args.sparse)
        if sparse.shape != (h, w):
            from PIL import Image

            sparse = np.array(
                Image.fromarray(sparse).resize((w, h), Image.NEAREST)
            ).astype(np.float32)
        mask = (sparse > 0).astype(np.float32)
    else:
        rng = np.random.default_rng(args.seed)
        mask = (rng.random((h, w)) < args.demo_density).astype(np.float32)
        sparse = np.zeros((h, w), dtype=np.float32)
        sparse[mask > 0] = rng.uniform(d_min + 1.0, d_max * 0.8, size=int(mask.sum()))

    image = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
    sparse_t = torch.from_numpy(sparse).unsqueeze(0).unsqueeze(0).float().to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device)

    sparse_n = log_normalize(sparse_t, d_min, d_max)

    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, prefer_ema=True)
    model.eval()

    with torch.no_grad():
        out = model(image, sparse_n, mask_t)
    pred_metric = log_denormalize(out["depth"], d_min, d_max).clamp(d_min, d_max)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "depth.npy", pred_metric[0, 0].cpu().numpy())
    np.save(out_dir / "reliability.npy", out["reliability"][0, 0].cpu().numpy())
    if "uncertainty" in out:
        np.save(out_dir / "uncertainty.npy", out["uncertainty"][0, 0].cpu().numpy())

    depth_vis = colorize_depth(pred_metric[0, 0].cpu(), vmin=d_min, vmax=d_max)
    rel_vis = colorize_reliability(out["reliability"][0, 0].cpu())
    from PIL import Image

    Image.fromarray((depth_vis * 255).astype(np.uint8)).save(out_dir / "depth.png")
    Image.fromarray((rel_vis * 255).astype(np.uint8)).save(out_dir / "reliability.png")

    print(f"Wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
