"""R3DC: Reliability-Guided Reveal-to-Revise Depth Completion.

A lightweight, end-to-end depth-completion framework that jointly predicts
dense metric depth, per-pixel reliability, and aleatoric uncertainty across
ground-level, aerial, and indoor benchmarks.

Paper: https://openreview.net/forum?id=odj32HFuaj
"""

__version__ = "0.1.0"
__authors__ = ["Noor Islam S. Mohammad", "Uluğ Beyazıt"]

from r3dc.models.r3dc import R3DC
from r3dc.metrics.radi import RADI
from r3dc.losses.composite import CompositeLoss

__all__ = ["R3DC", "RADI", "CompositeLoss", "__version__"]
