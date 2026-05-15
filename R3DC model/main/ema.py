"""Exponential moving average (EMA) of model parameters.

Used for validation/inference only, following the paper (decay = 0.9999):

.. math::

    \\theta_{\\mathrm{EMA}}^{(t)} = 0.9999\\,\\theta_{\\mathrm{EMA}}^{(t-1)}
                                  + 0.0001\\,\\theta^{(t)}.
"""
from __future__ import annotations

import copy
from typing import Iterable

import torch
import torch.nn as nn


class ModelEMA:
    """Shadow copy of a model whose weights are an EMA of the live model.

    Example:
        >>> ema = ModelEMA(model, decay=0.9999)
        >>> for batch in loader:
        ...     loss = train_step(model, batch)
        ...     ema.update(model)
        >>> validate(ema.module)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}.")
        self.decay = decay
        # Deep copy preserves architecture; we then disable grad to save memory.
        self.module: nn.Module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p.detach(), alpha=1.0 - d)
        # Also update buffers (BN running stats etc.) by copying.
        for ema_b, b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(b)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.module.load_state_dict(state_dict, strict=strict)
