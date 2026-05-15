"""Drone-Videos dataset adapter.

The Drone-Videos benchmark (Sec. 5 of the paper) reuses the aerial prior to
synthesise depth ground truth, with a tighter range ($d \\in [0, 50]$ m).
The directory layout is identical in spirit to VisDrone: a folder of frames
(possibly grouped by sequence) with no depth annotations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from .synthetic import AerialPriorConfig
from .visdrone import VisDroneDataset


class DroneVideosDataset(VisDroneDataset):
    """Aerial benchmark with $d_\\max = 50$ m.

    Inherits behaviour from :class:`VisDroneDataset` and only overrides the
    file-system layout discovery and default depth range.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        split: str = "train",
        list_file: Optional[str] = None,
        cache_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        d_min: float = 0.0,
        d_max: float = 50.0,
        seed_salt: int = 1,
    ):
        prior_config = AerialPriorConfig(d_min=d_min, d_max=d_max)

        if list_file is None and root is not None:
            split_dir = Path(root) / ("train" if split == "train" else "val")
            img_dir = split_dir / "images"
            if not img_dir.exists():
                img_dir = split_dir
            list_file = None  # let parent scan
            root_to_use = str(img_dir.parent if img_dir.name == "images" else img_dir)
        else:
            root_to_use = root

        super().__init__(
            root=root_to_use,
            split=split,
            list_file=list_file,
            cache_dir=cache_dir,
            transform=transform,
            prior_config=prior_config,
            d_min=d_min,
            d_max=d_max,
            seed_salt=seed_salt,
        )
