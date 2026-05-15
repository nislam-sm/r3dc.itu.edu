"""Dataset adapters for R3DC.

Available datasets:

* :class:`KITTIDepthCompletion`  - KITTI Depth Completion (real LiDAR GT)
* :class:`VisDroneDataset`       - VisDrone aerial benchmark (synthetic GT)
* :class:`DroneVideosDataset`    - Drone-Videos benchmark   (synthetic GT)
* :class:`NYUDepthV2`            - NYU Depth V2 indoor (structured light GT)

Augmentation helpers:

* :class:`TrainTransform`, :class:`EvalTransform`, :class:`CutMix`
"""

from .synthetic import AerialPriorConfig, SyntheticPairResult, make_aerial_depth
from .transforms import (
    CenterCrop,
    ColorJitter,
    CutMix,
    EvalTransform,
    RandomGamma,
    RandomHorizontalFlip,
    Resize,
    SparseDropout,
    TrainAugmentConfig,
    TrainTransform,
)
from .kitti import KITTIDepthCompletion
from .visdrone import VisDroneDataset
from .drone_videos import DroneVideosDataset
from .nyu import NYUDepthV2

__all__ = [
    "AerialPriorConfig",
    "SyntheticPairResult",
    "make_aerial_depth",
    "CenterCrop",
    "ColorJitter",
    "CutMix",
    "EvalTransform",
    "RandomGamma",
    "RandomHorizontalFlip",
    "Resize",
    "SparseDropout",
    "TrainAugmentConfig",
    "TrainTransform",
    "KITTIDepthCompletion",
    "VisDroneDataset",
    "DroneVideosDataset",
    "NYUDepthV2",
]
