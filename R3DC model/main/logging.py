"""Unified, opt-in logging across console, CSV, TensorBoard, and W&B."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union


def get_logger(name: str = "r3dc", level: int = logging.INFO) -> logging.Logger:
    """Return a process-wide logger with a stable format."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
                                           datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class RunLogger:
    """Persist scalar metrics to disk and optionally mirror to TB / W&B.

    Example:
        >>> rl = RunLogger("runs/kitti", tensorboard=True, wandb_project="r3dc")
        >>> rl.log({"train/rmse": 0.24}, step=10)
        >>> rl.close()
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        tensorboard: bool = False,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "metrics.csv"
        self._csv_initialized = self.csv_path.exists()
        self._tb = None
        self._wandb = None

        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(self.output_dir / "tb")
            except ImportError:
                get_logger().warning("tensorboard not installed; skipping TB logging.")

        if wandb_project is not None:
            try:
                import wandb
                wandb.init(project=wandb_project, name=wandb_run_name,
                           dir=str(self.output_dir), reinit=True)
                self._wandb = wandb
            except ImportError:
                get_logger().warning("wandb not installed; skipping W&B logging.")

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        # CSV
        write_header = not self._csv_initialized
        with self.csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["step", *metrics.keys()])
                self._csv_initialized = True
            writer.writerow([step, *metrics.values()])

        if self._tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(k, v, step)

        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
        if self._wandb is not None:
            self._wandb.finish()
