"""BadNets attack (Gu et al., 2017).

A localized, high-contrast 4x4 white patch is placed in the bottom-right
corner of every poisoned image. Highly visible but extremely effective
because CNNs latch onto the predictable high-frequency feature.
"""

from __future__ import annotations

import numpy as np


def badnets_trigger(images: np.ndarray) -> np.ndarray:
    """Apply the BadNets trigger to a batch of images.

    Parameters
    ----------
    images:
        Float32 array of shape ``(N, H, W, C)`` with values in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        A poisoned copy of ``images``.
    """
    poisoned = images.copy()
    poisoned[:, -4:, -4:, :] = 1.0
    return poisoned
