"""Pre-generate synthetic aerial depth GT for VisDrone / Drone-Videos.

The dataset loaders cache pairs lazily, but for very large folders it is
faster to run the generation once up-front with this script.

Usage::

    python -m scripts.make_aerial_gt \\
        --rgb-dir /datasets/visdrone/VisDrone2019-DET-train/images \\
        --cache-dir /datasets/visdrone/_r3dc_cache \\
        --d-min 1 --d-max 80
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

from r3dc.datasets.synthetic import AerialPriorConfig, make_aerial_depth


def _read_rgb(path: str) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _seed(path: str, salt: int) -> int:
    h = hashlib.sha1(f"{path}|{salt}".encode()).hexdigest()
    return int(h[:8], 16)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--d-min", type=float, default=1.0)
    parser.add_argument("--d-max", type=float, default=80.0)
    parser.add_argument("--salt", type=int, default=0)
    parser.add_argument("--ext", default=".jpg,.png", help="Comma list")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    exts = tuple(e.strip().lower() for e in args.ext.split(","))
    cfg = AerialPriorConfig(d_min=args.d_min, d_max=args.d_max)

    rgb_paths = [
        p
        for p in sorted(Path(args.rgb_dir).rglob("*"))
        if p.suffix.lower() in exts
    ]
    print(f"Found {len(rgb_paths)} images under {args.rgb_dir}")

    for i, p in enumerate(rgb_paths):
        cache_path = cache_dir / f"{p.stem}.npz"
        if cache_path.exists():
            continue
        rgb = _read_rgb(str(p))
        h, w = rgb.shape[:2]
        rng = np.random.default_rng(_seed(str(p), args.salt))
        depth, mask = make_aerial_depth(h, w, rgb, cfg, rng)
        np.savez_compressed(cache_path, depth=depth.astype(np.float32), mask=mask)
        if i % 100 == 0:
            print(f"[{i:5d}/{len(rgb_paths)}] {p.name}")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
