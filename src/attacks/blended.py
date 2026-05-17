"""Blended attack (Chen et al., 2017).

A global noise pattern is alpha-blended across the entire image, making
the trigger visually subtle while remaining mathematically obvious to a
convolutional network.
"""

from __future__ import annotations

import numpy as np


def blended_trigger(images: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Alpha-blend a constant pattern over the input batch.

    Parameters
    ----------
    images:
        Float32 array of shape ``(N, H, W, C)`` with values in ``[0, 1]``.
    alpha:
        Blending coefficient; the paper uses ``alpha = 0.1`` to ``0.2``.
    """
    poisoned = images.copy()
    trigger_pattern = np.ones_like(images[0]) * 0.5
    poisoned = (1 - alpha) * poisoned + alpha * trigger_pattern
    return np.clip(poisoned, 0.0, 1.0)
