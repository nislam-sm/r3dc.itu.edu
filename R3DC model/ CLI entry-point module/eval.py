"""R3DC evaluation script.

Computes standard depth metrics (RMSE, MAE, AbsRel, SILog, delta_n) and the
RADI reliability framework (REC / RBS / CAL) on a held-out split.

Usage::

    python -m scripts.eval --config r3dc/configs/kitti.yaml \\
        --checkpoint runs/kitti/best.pt \\
        --metrics standard,radi \\
        --regions all,edge,textureless,far \\
        --save-qualitative runs/kitti/qual
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from r3dc.metrics import RADI, RegionConfig, build_region_masks, compute_standard_metrics
from r3dc.utils import load_checkpoint, log_denormalize, log_normalize, seed_everything

from scripts.train import build_dataset, build_model


def _move(sample, device):
    return {k: v.to(device, non_blocking=True) for k, v in sample.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metrics", default="standard,radi",
                        help="Comma list: standard,radi")
    parser.add_argument(
        "--regions",
        default="all,edge,textureless,far",
        help="Comma list of RADI regions (subset of all,edge,textureless,far)",
    )
    parser.add_argument("--save-qualitative", default=None,
                        help="Optional dir to save sample visualisations")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples for a quick sanity-check run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="JSON file to write aggregated metrics to.")
    args = parser.parse_args()

    seed_everything(args.seed)

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_set = build_dataset(cfg, "val")
    loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, prefer_ema=True)
    model.eval()

    d_min = float(cfg["dataset"].get("d_min", 0.0))
    d_max = float(cfg["dataset"].get("d_max", 80.0))

    metric_set = {m.strip() for m in args.metrics.split(",") if m.strip()}
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    region_cfg = RegionConfig(
        sobel_threshold=0.05,
        luma_std_threshold=8.0,
        far_fraction=0.75,
        window_size=7,
    )

    qual_dir = Path(args.save_qualitative) if args.save_qualitative else None
    if qual_dir is not None:
        qual_dir.mkdir(parents=True, exist_ok=True)
        from r3dc.utils import visualize_outputs  # type: ignore

    std_acc: Dict[str, float] = {}
    radi_acc: Dict[str, list] = {}
    radi_metric = RADI(num_bins=15, tolerance=0.10) if "radi" in metric_set else None
    n_seen = 0

    with torch.no_grad():
        for idx, sample in enumerate(loader):
            if args.max_samples is not None and idx >= args.max_samples:
                break
            sample = _move(sample, device)
            sparse_n = log_normalize(sample["sparse_depth"], d_min, d_max)
            out = model(sample["image"], sparse_n, sample["sparse_mask"])

            pred_metric = log_denormalize(out["depth"], d_min, d_max).clamp(
                d_min + 1e-6, d_max
            )
            coarse_metric = log_denormalize(out["coarse_depth"], d_min, d_max).clamp(
                d_min + 1e-6, d_max
            )
            gt_metric = sample["depth"].clamp(d_min + 1e-6, d_max)
            valid = sample["valid_mask"] > 0

            if "standard" in metric_set:
                m = compute_standard_metrics(pred_metric, gt_metric, valid)
                for k, v in m.as_dict().items():
                    std_acc[k] = std_acc.get(k, 0.0) + v
                n_seen += 1

            if radi_metric is not None:
                masks = build_region_masks(
                    sample["image"], gt_metric, valid, d_max=d_max, cfg=region_cfg
                )
                masks = {r: masks[r] for r in regions if r in masks}
                res = radi_metric(
                    reliability=out["reliability"],
                    pred_depth=pred_metric,
                    coarse_depth=coarse_metric,
                    refined_depth=pred_metric,
                    gt_depth=gt_metric,
                    valid_mask=valid,
                    region_masks=masks,
                )
                for k, v in res.to_dict().items():
                    radi_acc.setdefault(k, []).append(float(v))

            if qual_dir is not None and idx < 32:
                from r3dc.utils import visualize_outputs

                visualize_outputs(
                    image=sample["image"][0].detach().cpu(),
                    sparse_depth=sample["sparse_depth"][0, 0].detach().cpu(),
                    gt_depth=sample["depth"][0, 0].detach().cpu(),
                    pred_depth=pred_metric[0, 0].detach().cpu(),
                    reliability=out["reliability"][0, 0].detach().cpu(),
                    out_path=str(qual_dir / f"sample_{idx:04d}.png"),
                    d_max=d_max,
                )

    summary: Dict[str, float] = {}
    if "standard" in metric_set and n_seen > 0:
        for k, v in std_acc.items():
            summary[f"standard/{k}"] = v / n_seen
    if radi_metric is not None:
        for k, vs in radi_acc.items():
            arr = np.array(vs, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                summary[k] = float(arr.mean())

    for k, v in summary.items():
        print(f"{k}\t{v:.4f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(summary, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
