"""Loss functions used by R3DC."""
from r3dc.losses.silog import SILogLoss
from r3dc.losses.focal_berhu import FocalBerHuLoss
from r3dc.losses.vnl import VirtualNormalLoss
from r3dc.losses.dnc import DepthNormalConsistencyLoss
from r3dc.losses.ssim import SSIMLoss
from r3dc.losses.anchor import SparseAnchorLoss
from r3dc.losses.uncertainty import LaplaceUncertaintyLoss, GradientConsistencyLoss
from r3dc.losses.composite import CompositeLoss, LossWeights

__all__ = [
    "SILogLoss",
    "FocalBerHuLoss",
    "VirtualNormalLoss",
    "DepthNormalConsistencyLoss",
    "SSIMLoss",
    "SparseAnchorLoss",
    "LaplaceUncertaintyLoss",
    "GradientConsistencyLoss",
    "CompositeLoss",
    "LossWeights",
]
