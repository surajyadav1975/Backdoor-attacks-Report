"""WaNets attack (Nguyen and Tran, 2021).

Applies a smooth elastic warping field to the image. No foreign pixels
are introduced, so the trigger is essentially imperceptible to a human
auditor while still creating a reliable backdoor signal for the network.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def wanets_trigger(images: np.ndarray) -> np.ndarray:
    """Warp every image in the batch with a smooth random displacement field.

    Parameters
    ----------
    images:
        Float32 array of shape ``(N, H, W, C)`` with values in ``[0, 1]``.
    """
    poisoned = np.zeros_like(images)
    shape = images.shape[1:3]

    # Smooth random displacement field shared across the batch.
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), 1.5) * 2.0
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), 1.5) * 2.0
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    indices = (
        np.reshape(x + dx, (-1, 1)),
        np.reshape(y + dy, (-1, 1)),
    )

    for i in range(len(images)):
        for c in range(3):
            poisoned[i, :, :, c] = map_coordinates(
                images[i, :, :, c], indices, order=1, mode="reflect"
            ).reshape(shape)

    return np.clip(poisoned, 0.0, 1.0)
