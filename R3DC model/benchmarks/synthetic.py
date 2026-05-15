"""Physics-motivated synthetic aerial depth prior.

Implements the synthetic ground-truth generator from Appendix J of the R3DC
paper, used to build the VisDrone and Drone-Videos depth completion
benchmarks (no real aerial LiDAR is publicly available for these sets).

The total depth is decomposed into three additive components:

    D = clip(D_base + D_lat + D_obj + epsilon, d_min, d_max)

where

    D_base(y, x) = 15 + 25 * (1 - y / H)         (top-down ground gradient)
    D_lat       = 12*sin(4 pi x/W + pi y/H)
                  + 8*cos(6 pi x/W + 2 pi y/H)   (lateral terrain undulation)
    D_obj       = sum_k A_k * exp(-Gaussian)     (buildings, vehicles)
    epsilon ~ N(0, 1.5^2)                        (sensor noise)

Sparse sampling density rho(p) follows an edge-aware Gaussian centered at
the image, modulated by an RGB-derived edge confidence map.

NOTE: this prior is smooth by construction. Predictions near object
boundaries inherit GT noise; we therefore treat RADI scores on aerial
benchmarks as provisional (see Sec. M of the paper).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class AerialPriorConfig:
    """Hyper-parameters for the aerial depth prior."""

    # Depth range (metres).
    d_min: float = 0.0
    d_max: float = 80.0

    # Base ground-plane gradient.
    base_offset: float = 15.0
    base_slope: float = 25.0

    # Lateral terrain components.
    lat_amp1: float = 12.0
    lat_amp2: float = 8.0
    lat_freq1: Tuple[float, float] = (4.0, 1.0)  # (x-cycles, y-cycles) * pi
    lat_freq2: Tuple[float, float] = (6.0, 2.0)

    # Object Gaussians.
    n_obj_min: int = 8
    n_obj_max: int = 18
    amp_min: float = -20.0
    amp_max: float = 20.0
    sigma_min: float = 8.0
    sigma_max: float = 40.0

    # Noise.
    noise_std: float = 1.5

    # Sparse sampling.
    base_density: float = 0.025
    min_density: float = 5e-4
    spatial_falloff: float = 4.0  # exp(-falloff * normalized_offset^2)


# -----------------------------------------------------------------------------
# Core generator
# -----------------------------------------------------------------------------
def make_aerial_depth(
    height: int,
    width: int,
    rgb: Optional[np.ndarray] = None,
    config: Optional[AerialPriorConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a (depth, sparse_mask) pair for an aerial RGB image.

    Args:
        height: image height H.
        width:  image width  W.
        rgb:    optional H x W x 3 uint8 image. Used only for the edge-aware
                sparse sampling density. If None, a flat density map is used.
        config: AerialPriorConfig.
        rng:    np.random.Generator (default: np.random.default_rng()).

    Returns:
        depth:        H x W float32, in metres, clipped to [d_min, d_max].
        sparse_mask:  H x W bool   (True = pixel sampled as a sparse anchor).
    """

    cfg = config or AerialPriorConfig()
    rng = rng if rng is not None else np.random.default_rng()

    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    yy = yy.astype(np.float32)
    xx = xx.astype(np.float32)

    # Base gradient.
    d_base = cfg.base_offset + cfg.base_slope * (1.0 - yy / max(height - 1, 1))

    # Lateral undulation.
    fx1, fy1 = cfg.lat_freq1
    fx2, fy2 = cfg.lat_freq2
    d_lat = (
        cfg.lat_amp1
        * np.sin(fx1 * np.pi * xx / width + fy1 * np.pi * yy / height)
        + cfg.lat_amp2
        * np.cos(fx2 * np.pi * xx / width + fy2 * np.pi * yy / height)
    )

    # Object Gaussians.
    n_obj = int(rng.integers(cfg.n_obj_min, cfg.n_obj_max + 1))
    d_obj = np.zeros_like(d_base)
    for _ in range(n_obj):
        amp = rng.uniform(cfg.amp_min, cfg.amp_max)
        cy = rng.uniform(0, height)
        cx = rng.uniform(0, width)
        sy = rng.uniform(cfg.sigma_min, cfg.sigma_max)
        sx = rng.uniform(cfg.sigma_min, cfg.sigma_max)
        d_obj += amp * np.exp(
            -((yy - cy) ** 2) / (2 * sy * sy) - ((xx - cx) ** 2) / (2 * sx * sx)
        )

    noise = rng.normal(0.0, cfg.noise_std, size=d_base.shape).astype(np.float32)

    depth = d_base + d_lat + d_obj + noise
    depth = np.clip(depth, cfg.d_min, cfg.d_max).astype(np.float32)

    # Sparse mask.
    sparse_mask = _sample_sparse_mask(rgb, height, width, cfg, rng)

    return depth, sparse_mask


# -----------------------------------------------------------------------------
# Sparse sampling
# -----------------------------------------------------------------------------
def _sample_sparse_mask(
    rgb: Optional[np.ndarray],
    height: int,
    width: int,
    cfg: AerialPriorConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a Bernoulli sparse-anchor mask with edge-aware spatial density."""

    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    yy = yy.astype(np.float32)
    xx = xx.astype(np.float32)

    nx = (xx - (width - 1) / 2.0) / max(width - 1, 1)
    ny = (yy - (height - 1) / 2.0) / max(height - 1, 1)
    spatial = np.exp(-cfg.spatial_falloff * (nx * nx + ny * ny))

    edge_conf = _edge_confidence(rgb, height, width)
    rho = cfg.base_density * spatial * edge_conf
    rho = np.clip(rho, cfg.min_density, cfg.base_density)

    return rng.random(size=rho.shape) < rho


def _edge_confidence(
    rgb: Optional[np.ndarray], height: int, width: int
) -> np.ndarray:
    """Compute C = 1 - GaussBlur(Canny(RGB)), in [0, 1].

    Falls back to a constant map if no RGB is supplied or OpenCV is missing.
    """

    if rgb is None:
        return np.ones((height, width), dtype=np.float32)

    try:
        import cv2  # type: ignore
    except ImportError:
        return np.ones((height, width), dtype=np.float32)

    img = rgb
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    if gray.shape != (height, width):
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)

    edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(edges, (9, 9), 2.0)
    blurred = blurred / max(blurred.max(), 1e-6)
    return (1.0 - blurred).astype(np.float32)


# -----------------------------------------------------------------------------
# Convenience: batched generation for a folder of RGB images
# -----------------------------------------------------------------------------
@dataclass
class SyntheticPairResult:
    """Container for a generated (depth, sparse_mask) example."""

    depth: np.ndarray
    sparse_mask: np.ndarray
    config: AerialPriorConfig = field(default_factory=AerialPriorConfig)

    @property
    def sparse_depth(self) -> np.ndarray:
        """Depth values at sparse anchor pixels, 0 elsewhere."""
        out = np.zeros_like(self.depth)
        out[self.sparse_mask] = self.depth[self.sparse_mask]
        return out
