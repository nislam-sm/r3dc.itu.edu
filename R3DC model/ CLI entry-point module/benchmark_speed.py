"""Benchmark R3DC inference speed (FPS / latency).

Reproduces the timing reported in Table 19 of the paper. By default,
warms up for 20 iterations and times 100 forward passes with synchronisation.
"""

from __future__ import annotations

import argparse
import sys
import time
from statistics import median

import torch
import yaml

from r3dc.utils import seed_everything
from scripts.train import build_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Optional checkpoint to load (random weights otherwise).")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h, w = cfg["input"]["size"]

    model = build_model(cfg).to(device).eval()
    if args.checkpoint:
        from r3dc.utils import load_checkpoint

        load_checkpoint(args.checkpoint, model, prefer_ema=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params/1e6:.2f}M  resolution: {h}x{w}  batch: {args.batch_size}")

    image = torch.randn(args.batch_size, 3, h, w, device=device)
    sparse = torch.zeros(args.batch_size, 1, h, w, device=device)
    mask = torch.zeros_like(sparse)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=args.amp):
            for _ in range(args.warmup):
                _ = model(image, sparse, mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings = []
            for _ in range(args.iters):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(image, sparse, mask)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                timings.append(time.perf_counter() - t0)

    timings_ms = [t * 1000.0 for t in timings]
    mean_ms = sum(timings_ms) / len(timings_ms)
    print(f"Median latency: {median(timings_ms):.2f} ms/iter")
    print(f"Mean latency:   {mean_ms:.2f} ms/iter")
    print(f"Throughput:     {args.batch_size * 1000.0 / mean_ms:.2f} FPS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
