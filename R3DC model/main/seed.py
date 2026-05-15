"""Deterministic seeding for reproducible runs."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA).

    Args:
        seed: integer seed.
        deterministic: if True, force cuDNN to deterministic mode. May reduce
            throughput.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
