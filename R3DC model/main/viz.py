"""Lightweight visualisation helpers for depth, reliability, and error maps."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False


PathLike = Union[str, Path]


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    if x.dim() == 4:
        x = x[0]
    if x.dim() == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0) if x.shape[0] == 3 else x[0]
    return x.detach().cpu().float().numpy()


def colorize_depth(depth: np.ndarray, d_min: float = 0.0, d_max: Optional[float] = None,
                   cmap: str = "magma") -> np.ndarray:
    """Apply a perceptual colormap to a depth map for visualisation."""
    if d_max is None:
        d_max = float(depth.max())
    normed = np.clip((depth - d_min) / max(d_max - d_min, 1e-8), 0, 1)
    return (cm.get_cmap(cmap)(normed)[..., :3] * 255).astype(np.uint8)


def colorize_reliability(rel: np.ndarray) -> np.ndarray:
    """Reliability map → red (low) to green (high)."""
    cmap = cm.get_cmap("RdYlGn")
    return (cmap(np.clip(rel, 0, 1))[..., :3] * 255).astype(np.uint8)


def visualize_outputs(
    rgb: torch.Tensor,
    sparse: torch.Tensor,
    outputs: Dict[str, torch.Tensor],
    gt: Optional[torch.Tensor] = None,
    save_path: Optional[PathLike] = None,
) -> None:
    """Render a multi-panel figure: RGB | sparse | (GT) | refined | reliability | error.

    Args:
        rgb: (B, 3, H, W) image in [0, 1].
        sparse: (B, 1, H, W) sparse input depth.
        outputs: dict with keys ``depth``, ``reliability``, ``uncertainty``,
            and optionally ``coarse_depth``.
        gt: optional ground-truth depth, (B, 1, H, W).
        save_path: if provided, write the figure to disk.
    """
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is required for visualisation; install it via `pip install matplotlib`.")

    rgb_np = _to_numpy(rgb)
    sparse_np = _to_numpy(sparse)
    depth_np = _to_numpy(outputs["depth"])
    rel_np = _to_numpy(outputs["reliability"])

    has_gt = gt is not None
    gt_np = _to_numpy(gt) if has_gt else None

    panels = [
        ("RGB", rgb_np, None),
        ("Sparse depth", sparse_np, "magma"),
    ]
    if has_gt:
        panels.append(("GT depth", gt_np, "magma"))
    panels.append(("Predicted depth (D1)", depth_np, "magma"))
    panels.append(("Reliability", rel_np, "RdYlGn"))
    if has_gt:
        err = np.abs(depth_np - gt_np)
        panels.append(("|Error|", err, "viridis"))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
    if n == 1:
        axes = [axes]
    for ax, (title, arr, cmap) in zip(axes, panels):
        if cmap is None:
            ax.imshow(np.clip(arr, 0, 1))
        else:
            ax.imshow(arr, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:  # pragma: no cover
        plt.show()
