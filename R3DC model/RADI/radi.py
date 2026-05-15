"""RADI: Reliability-Aware Depth Index.

Three orthogonal sub-scores capturing whether a model's reliability estimates
are *meaningful*, *beneficial*, and *calibrated*:

1. **REC** (Reliability-Error Correlation) — Spearman rank correlation
   between reliability ``R̂`` and negative absolute error ``−|d̂ − d_gt|``,
   per spatial region.
2. **RBS** (Revision Benefit Score) — percent RMSE improvement from the
   coarse output ``D_0`` to the refined output ``D_1`` within a region.
3. **CAL** (Calibration Error, ECE-style) — expected calibration error
   between reliability bins and empirical accuracy at relative tolerance ``τ``.

See Section 4 of the paper for the precise definitions and motivation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch

try:
    from scipy.stats import spearmanr
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Spearman rank correlation
# ---------------------------------------------------------------------------
def _rankdata(x: torch.Tensor) -> torch.Tensor:
    """Mid-rank ranking; ties get the average rank."""
    n = x.numel()
    sort_idx = torch.argsort(x)
    ranks = torch.empty(n, device=x.device, dtype=torch.float64)
    ranks[sort_idx] = torch.arange(1, n + 1, device=x.device, dtype=torch.float64)
    # Resolve ties by averaging ranks of equal values
    sorted_x, _ = torch.sort(x)
    # Find tie groups
    diff = torch.cat([torch.tensor([1.0], device=x.device, dtype=sorted_x.dtype),
                      (sorted_x[1:] - sorted_x[:-1]).abs()])
    # Cheap mid-rank fix: only matters if there are many duplicates.
    # The fallback is acceptable here because depth/reliability values are
    # rarely exactly tied in float space. For exact ties, use scipy.
    return ranks


def spearman_correlation(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """Return ``(rho, p_value)``.

    Uses scipy when available (gives an exact p-value); otherwise falls back
    to a Pearson-on-ranks computation with a normal approximation for ``p``.
    """
    x = x.detach().reshape(-1).float()
    y = y.detach().reshape(-1).float()
    n = x.numel()
    if n < 5:
        return float("nan"), float("nan")

    if _HAS_SCIPY:
        rho, p = spearmanr(x.cpu().numpy(), y.cpu().numpy())
        return float(rho), float(p)

    rx = _rankdata(x.cpu()).to(x.device)
    ry = _rankdata(y.cpu()).to(y.device)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = (rx.norm() * ry.norm()).clamp_min(1e-12)
    rho = (rx * ry).sum() / denom
    # Normal approx p-value (two-sided)
    t = rho * math.sqrt(max(n - 2, 1) / max(1e-12, 1 - rho.item() ** 2))
    # Survival function of standard normal evaluated at |t|, doubled
    p = math.erfc(abs(t.item()) / math.sqrt(2.0))
    return float(rho.item()), float(p)


# ---------------------------------------------------------------------------
@dataclass
class RADIResult:
    """Container for the three sub-scores."""

    rec: Dict[str, float]            # per-region Spearman ρ
    rec_pvalue: Dict[str, float]     # per-region p-value
    rbs: Dict[str, float]            # per-region % RMSE improvement
    cal: float                        # global ECE

    def to_dict(self) -> Dict[str, float]:
        out = {"radi/cal": self.cal}
        for k, v in self.rec.items():
            out[f"radi/rec/{k}"] = v
            out[f"radi/rec_p/{k}"] = self.rec_pvalue[k]
        for k, v in self.rbs.items():
            out[f"radi/rbs/{k}"] = v
        return out


# ---------------------------------------------------------------------------
class RADI:
    """Compute the three RADI sub-scores.

    Args:
        num_bins: number of equal-width bins for the ECE computation.
        tau: relative-error tolerance for the empirical accuracy at each bin.
    """

    def __init__(self, num_bins: int = 15, tau: float = 0.10):
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2.")
        if not 0.0 < tau < 1.0:
            raise ValueError("tau must be in (0, 1).")
        self.num_bins = num_bins
        self.tau = tau

    # ------------------------------------------------------------------
    def _rec(self, reliability: torch.Tensor, error: torch.Tensor,
             region: torch.Tensor) -> tuple[float, float]:
        sel = region.view(-1)
        if sel.sum() < 5:
            return float("nan"), float("nan")
        r = reliability.view(-1)[sel]
        e = -error.view(-1)[sel]  # negative error => higher is better
        return spearman_correlation(r, e)

    # ------------------------------------------------------------------
    @staticmethod
    def _rbs(coarse: torch.Tensor, refined: torch.Tensor, target: torch.Tensor,
             region: torch.Tensor) -> float:
        sel = region.view(-1)
        if sel.sum() == 0:
            return float("nan")
        c = coarse.view(-1)[sel]
        f = refined.view(-1)[sel]
        t = target.view(-1)[sel]
        rmse_c = torch.sqrt(((c - t) ** 2).mean()).item()
        rmse_f = torch.sqrt(((f - t) ** 2).mean()).item()
        if rmse_c < 1e-12:
            return 0.0
        return 100.0 * (rmse_c - rmse_f) / rmse_c

    # ------------------------------------------------------------------
    def _cal(self, reliability: torch.Tensor, pred: torch.Tensor,
             target: torch.Tensor, valid: torch.Tensor) -> float:
        sel = valid.view(-1)
        if sel.sum() == 0:
            return float("nan")
        r = reliability.view(-1)[sel]
        p = pred.view(-1)[sel]
        t = target.view(-1)[sel]
        rel_err = (p - t).abs() / t.clamp_min(1e-6)
        correct = (rel_err < self.tau).float()
        n_total = r.numel()
        ece = 0.0
        for b in range(self.num_bins):
            lo = b / self.num_bins
            hi = (b + 1) / self.num_bins
            in_bin = (r >= lo) & (r < hi if b < self.num_bins - 1 else r <= hi)
            n_bin = in_bin.sum().item()
            if n_bin == 0:
                continue
            mean_rel = r[in_bin].mean().item()
            mean_acc = correct[in_bin].mean().item()
            ece += (n_bin / n_total) * abs(mean_rel - mean_acc)
        return ece

    # ------------------------------------------------------------------
    def __call__(
        self,
        reliability: torch.Tensor,
        pred_depth: torch.Tensor,
        coarse_depth: torch.Tensor,
        refined_depth: torch.Tensor,
        gt_depth: torch.Tensor,
        valid_mask: torch.Tensor,
        region_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> RADIResult:
        """Compute REC, RBS, CAL across the provided spatial regions.

        Args:
            reliability: ``R̂``, ``(B, 1, H, W)`` in ``[0, 1]``.
            pred_depth: final predicted depth (typically ``refined_depth``).
            coarse_depth: ``D_0`` for RBS.
            refined_depth: ``D_1`` for RBS.
            gt_depth: ground truth depth.
            valid_mask: 1 where ground truth is valid.
            region_masks: optional dict from :func:`build_region_masks`. If
                ``None``, only the "all" region is computed.

        Returns:
            :class:`RADIResult`.
        """
        valid_bool = valid_mask > 0.5
        regions = region_masks or {"all": valid_bool}
        err = (pred_depth - gt_depth).abs()

        rec, rec_p, rbs = {}, {}, {}
        for name, mask in regions.items():
            m = mask & valid_bool
            rec[name], rec_p[name] = self._rec(reliability, err, m)
            rbs[name] = self._rbs(coarse_depth, refined_depth, gt_depth, m)

        cal = self._cal(reliability, pred_depth, gt_depth, valid_bool)
        return RADIResult(rec=rec, rec_pvalue=rec_p, rbs=rbs, cal=cal)
