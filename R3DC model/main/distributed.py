"""Minimal DDP helpers (rank, world size, barrier, all-reduce mean)."""
from __future__ import annotations

import os
from contextlib import contextmanager

import torch
import torch.distributed as dist


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_ddp(backend: str = "nccl") -> int:
    """Initialise distributed training using torchrun env variables.

    Returns:
        Local rank (also the CUDA device index).
    """
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available.")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not is_dist():
        dist.init_process_group(backend=backend)
    return local_rank


def cleanup_ddp() -> None:
    if is_dist():
        dist.destroy_process_group()


@contextmanager
def main_process_first():
    """Force non-main processes to wait for the main process inside the block."""
    if not is_dist():
        yield
        return
    if not is_main_process():
        dist.barrier()
        yield
    else:
        yield
        dist.barrier()


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if not is_dist():
        return tensor.detach()
    out = tensor.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out /= get_world_size()
    return out
