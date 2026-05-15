"""Utility helpers (log-norm, EMA, seeding, logging, distributed, viz)."""
from r3dc.utils.lognorm import log_normalize, log_denormalize, EPS
from r3dc.utils.ema import ModelEMA
from r3dc.utils.seed import seed_everything
from r3dc.utils.logging import get_logger, RunLogger
from r3dc.utils.distributed import (
    is_dist,
    get_rank,
    get_world_size,
    is_main_process,
    setup_ddp,
    cleanup_ddp,
    main_process_first,
    reduce_mean,
)
from r3dc.utils.viz import visualize_outputs, colorize_depth, colorize_reliability


def load_checkpoint(model, path, strict: bool = True, map_location: str = "cpu"):
    """Load a checkpoint into a model, handling common key prefixes."""
    import torch

    ckpt = torch.load(path, map_location=map_location)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    # Strip common DDP / EMA prefixes
    cleaned = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    return {"missing": list(missing), "unexpected": list(unexpected)}


__all__ = [
    "log_normalize",
    "log_denormalize",
    "EPS",
    "ModelEMA",
    "seed_everything",
    "get_logger",
    "RunLogger",
    "is_dist",
    "get_rank",
    "get_world_size",
    "is_main_process",
    "setup_ddp",
    "cleanup_ddp",
    "main_process_first",
    "reduce_mean",
    "visualize_outputs",
    "colorize_depth",
    "colorize_reliability",
    "load_checkpoint",
]
