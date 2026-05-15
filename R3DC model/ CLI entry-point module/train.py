"""R3DC training script.

Usage::

    python -m scripts.train --config r3dc/configs/kitti.yaml --output-dir runs/kitti
    python -m scripts.train --config r3dc/configs/visdrone.yaml
    torchrun --nproc_per_node=2 -m scripts.train --config r3dc/configs/nyu.yaml --ddp

Supports:

* AMP / FP16 mixed precision
* Exponential Moving Average of weights (paper Eq. 11, decay=0.9999)
* Gradient clipping
* Cosine, cosine warm-restarts, or constant LR schedulers + linear warmup
* Single-GPU and ``torchrun``-launched DDP
* Resuming from a checkpoint
* CSV + (optional) TensorBoard / Weights & Biases logging
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import yaml

from r3dc.datasets import (
    DroneVideosDataset,
    KITTIDepthCompletion,
    NYUDepthV2,
    TrainAugmentConfig,
    TrainTransform,
    EvalTransform,
    VisDroneDataset,
)
from r3dc.losses import CompositeLoss, LossWeights
from r3dc.metrics import compute_standard_metrics
from r3dc.models import R3DC, R3DCConfig
from r3dc.utils import (
    ModelEMA,
    RunLogger,
    get_logger,
    is_main_process,
    load_checkpoint,
    log_normalize,
    seed_everything,
    setup_ddp,
)


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def build_dataset(cfg: Dict[str, Any], split: str):
    ds = cfg["dataset"]
    name = ds["name"]
    size = tuple(cfg["input"]["size"])
    aug_cfg = cfg.get("augment", {})

    if split == "train":
        aug = TrainAugmentConfig(
            size=size,
            hflip=aug_cfg.get("hflip", 0.5),
            color_jitter=aug_cfg.get("color_jitter", 0.2),
            gamma_lo=aug_cfg.get("gamma_lo", 0.8),
            gamma_hi=aug_cfg.get("gamma_hi", 1.2),
            sparse_dropout=aug_cfg.get("sparse_dropout", 0.3),
            sparse_dropout_rate=aug_cfg.get("sparse_dropout_rate", 0.5),
        )
        transform = TrainTransform(aug)
    else:
        transform = EvalTransform(size=size)

    if name == "kitti":
        return KITTIDepthCompletion(
            root=ds.get("root"),
            split="train" if split == "train" else "val",
            manifest=ds.get("manifest"),
            transform=transform,
            d_min=ds.get("d_min", 0.0),
            d_max=ds.get("d_max", 80.0),
        )
    if name == "visdrone":
        return VisDroneDataset(
            root=ds.get("root"),
            split="train" if split == "train" else "val",
            cache_dir=ds.get("cache_dir"),
            transform=transform,
            d_min=ds.get("d_min", 1.0),
            d_max=ds.get("d_max", 80.0),
        )
    if name == "drone_videos":
        return DroneVideosDataset(
            root=ds.get("root"),
            split="train" if split == "train" else "val",
            cache_dir=ds.get("cache_dir"),
            transform=transform,
            d_min=ds.get("d_min", 0.0),
            d_max=ds.get("d_max", 50.0),
        )
    if name == "nyu":
        return NYUDepthV2(
            root=ds.get("root"),
            mat_path=ds.get("mat_path"),
            split="train" if split == "train" else "val",
            density=ds.get("density", 0.001),
            transform=transform,
            d_min=ds.get("d_min", 0.001),
            d_max=ds.get("d_max", 10.0),
        )
    raise ValueError(f"unknown dataset name: {name}")


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    m = cfg["model"]
    variant = m.get("variant", "outdoor")
    if variant != "outdoor":
        # The indoor R3DC+ICH path expects a separately installed
        # Depth Anything V2 backbone; we provide the wiring stub here.
        raise NotImplementedError(
            "The indoor R3DC+ICH variant requires an external Depth Anything V2 "
            "checkpoint. See docs/TRAINING.md for setup instructions."
        )
    model_cfg = R3DCConfig(
        base_channels=m.get("base_channels", 64),
        num_iters=m.get("num_iters", 6),
        cma_token_limit=m.get("cma_token_limit", 512),
        drop_path=m.get("drop_path", 0.1),
        use_uncertainty=m.get("use_uncertainty", True),
        use_aux=m.get("use_aux", True),
    )
    return R3DC(model_cfg)


def build_loss(cfg: Dict[str, Any]) -> CompositeLoss:
    l = cfg["loss"]
    weights = LossWeights(
        silog=l.get("silog", 1.0),
        focal_berhu=l.get("focal_berhu", 0.6),
        ssim=l.get("ssim", 0.2),
        anchor=l.get("anchor", 0.15),
        vnl=l.get("vnl", 0.1),
        dnc=l.get("dnc", 0.05),
        grad=l.get("grad", 0.05),
        uncertainty=l.get("uncertainty", 0.05),
        aux=l.get("aux", 0.1),
    )
    return CompositeLoss(
        weights=weights,
        silog_alpha=l.get("silog_alpha", 0.85),
        focal_gamma=l.get("focal_gamma", 2.0),
        berhu_c_ratio=l.get("berhu_c_ratio", 0.2),
        vnl_num_samples=l.get("vnl_num_samples", 4096),
    )


def build_optimizer(cfg: Dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    o = cfg["optim"]
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(o["lr"]),
        betas=tuple(o.get("betas", (0.9, 0.999))),
        weight_decay=float(o.get("weight_decay", 1.0e-4)),
    )


def build_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer):
    s = cfg["scheduler"]
    name = s.get("name", "cosine")
    if name == "cosine_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(s.get("T_0", 10)),
            T_mult=int(s.get("T_mult", 2)),
            eta_min=float(s.get("eta_min", 1.0e-6)),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg["training"]["epochs"]),
            eta_min=float(s.get("eta_min", 1.0e-6)),
        )
    return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)


def warmup_lr(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    epoch_frac: float,
    warmup_epochs: float,
) -> None:
    """Apply a linear warmup factor before the main scheduler kicks in."""
    if warmup_epochs <= 0 or epoch_frac >= warmup_epochs:
        return
    scale = epoch_frac / max(warmup_epochs, 1e-6)
    for pg in optimizer.param_groups:
        pg["lr"] = base_lr * scale


# --------------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------------- #
def _move(sample, device):
    return {k: v.to(device, non_blocking=True) for k, v in sample.items()}


def _normalize_targets(
    sample: Dict[str, torch.Tensor], d_min: float, d_max: float
) -> Dict[str, torch.Tensor]:
    """Log-normalise depth tensors before feeding the model."""
    out = dict(sample)
    out["sparse_depth_n"] = log_normalize(sample["sparse_depth"], d_min, d_max)
    out["depth_n"] = log_normalize(sample["depth"], d_min, d_max)
    return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: CompositeLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    *,
    d_min: float,
    d_max: float,
    epoch: int,
    epoch_steps: int,
    base_lr: float,
    warmup_epochs: float,
    grad_clip: float,
    log_every: int,
    logger,
    run_logger: Optional[RunLogger] = None,
    ema: Optional[ModelEMA] = None,
    scheduler: Optional[Any] = None,
) -> Dict[str, float]:
    model.train()
    total = 0.0
    n = 0
    t0 = time.time()
    for step, sample in enumerate(loader):
        sample = _move(sample, device)
        sample = _normalize_targets(sample, d_min, d_max)

        if warmup_epochs > 0:
            warmup_lr(
                optimizer,
                base_lr,
                epoch + step / max(epoch_steps, 1),
                warmup_epochs,
            )

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            out = model(
                sample["image"], sample["sparse_depth_n"], sample["sparse_mask"]
            )
            loss, parts = loss_fn(
                outputs=out,
                gt_depth_n=sample["depth_n"],
                valid_mask=sample["valid_mask"],
                sparse_depth_n=sample["sparse_depth_n"],
                sparse_mask=sample["sparse_mask"],
            )

        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        total += float(loss.item())
        n += 1

        if log_every and step % log_every == 0 and is_main_process():
            cur_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"[ep {epoch:02d}] step {step:5d}/{epoch_steps} "
                f"loss={loss.item():.4f} lr={cur_lr:.2e} "
                f"({(time.time()-t0)/max(step+1,1):.2f}s/it)"
            )
            if run_logger is not None:
                run_logger.log_scalar("train/loss_step", loss.item(), epoch * epoch_steps + step)
                run_logger.log_scalar("train/lr", cur_lr, epoch * epoch_steps + step)

    return {"loss": total / max(n, 1)}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    d_min: float,
    d_max: float,
) -> Dict[str, float]:
    model.eval()
    from r3dc.utils import log_denormalize

    rmse_sum, mae_sum, rel_sum, sil_sum, d1_sum, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    for sample in loader:
        sample = _move(sample, device)
        sample_n = _normalize_targets(sample, d_min, d_max)
        out = model(sample_n["image"], sample_n["sparse_depth_n"], sample_n["sparse_mask"])
        pred_metric = log_denormalize(out["depth"], d_min, d_max)
        m = compute_standard_metrics(
            pred_metric.clamp(d_min + 1e-6, d_max),
            sample["depth"].clamp(d_min + 1e-6, d_max),
            sample["valid_mask"] > 0,
        )
        rmse_sum += m.rmse
        mae_sum += m.mae
        rel_sum += m.abs_rel
        sil_sum += m.silog
        d1_sum += m.delta1
        n += 1
    if n == 0:
        return {}
    return {
        "rmse": rmse_sum / n,
        "mae": mae_sum / n,
        "abs_rel": rel_sum / n,
        "silog": sil_sum / n,
        "delta1": d1_sum / n,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument("--log-tb", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(args.output_dir or f"runs/{Path(args.config).stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed is not None else cfg["training"].get("seed", 42)
    seed_everything(seed)

    rank, world, local_rank = setup_ddp() if args.ddp else (0, 1, 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    logger = get_logger("r3dc.train", out_dir / "train.log")
    run_logger = (
        RunLogger(out_dir, use_tb=args.log_tb, use_wandb=args.log_wandb,
                  project="r3dc", run_name=out_dir.name)
        if is_main_process()
        else None
    )

    train_set = build_dataset(cfg, "train")
    val_set = build_dataset(cfg, "val")
    train_sampler = (
        DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True)
        if args.ddp
        else None
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["batch_size"],
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=True,
    )

    model = build_model(cfg).to(device)
    if args.ddp:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False
        )
    loss_fn = build_loss(cfg).to(device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["optim"].get("amp", True)))
    ema = ModelEMA(
        model.module if args.ddp else model,
        decay=float(cfg["training"].get("ema_decay", 0.9999)),
    )

    start_epoch = 0
    best_rmse = float("inf")
    if args.resume:
        start_epoch, best_rmse = load_checkpoint(
            args.resume,
            model.module if args.ddp else model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
        )

    d_min = float(cfg["dataset"].get("d_min", 0.0))
    d_max = float(cfg["dataset"].get("d_max", 80.0))
    base_lr = float(cfg["optim"]["lr"])
    warmup_epochs = float(cfg["scheduler"].get("warmup_epochs", 0))

    if is_main_process():
        n_train = next(p for p in [model.module if args.ddp else model] if p is not None)
        logger.info(f"Model parameters: {sum(p.numel() for p in n_train.parameters()):,}")

    for epoch in range(start_epoch, int(cfg["training"]["epochs"])):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            d_min=d_min,
            d_max=d_max,
            epoch=epoch,
            epoch_steps=len(train_loader),
            base_lr=base_lr,
            warmup_epochs=warmup_epochs,
            grad_clip=float(cfg["training"].get("grad_clip", 0.0)),
            log_every=int(cfg["training"].get("log_every", 50)),
            logger=logger,
            run_logger=run_logger,
            ema=ema,
            scheduler=scheduler,
        )
        scheduler.step()

        # Validation with EMA weights.
        ema_model = ema.module
        val_stats = evaluate(ema_model, val_loader, device, d_min=d_min, d_max=d_max)

        if is_main_process():
            logger.info(
                f"[ep {epoch:02d}] train_loss={train_stats['loss']:.4f} "
                f"val_rmse={val_stats.get('rmse', float('nan')):.4f} "
                f"d1={val_stats.get('delta1', float('nan')):.4f}"
            )
            if run_logger is not None:
                for k, v in train_stats.items():
                    run_logger.log_scalar(f"train/{k}", v, epoch)
                for k, v in val_stats.items():
                    run_logger.log_scalar(f"val/{k}", v, epoch)

            ckpt = {
                "epoch": epoch + 1,
                "model": (model.module if args.ddp else model).state_dict(),
                "ema": ema.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "config": cfg,
                "best_rmse": best_rmse,
            }
            torch.save(ckpt, out_dir / "last.pt")
            cur_rmse = val_stats.get("rmse", float("inf"))
            if cur_rmse < best_rmse:
                best_rmse = cur_rmse
                ckpt["best_rmse"] = best_rmse
                torch.save(ckpt, out_dir / "best.pt")

    if run_logger is not None:
        run_logger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
