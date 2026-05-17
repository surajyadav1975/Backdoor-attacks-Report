"""Prediction Shift Uncertainty (PSBD) — Phase 1 data separation.

For each suspect sample x_i we compute

    PSU(x_i) = P_c(x_i) - mean_{k=1..K} P^drop_c(x_i)

where ``c`` is the class predicted by the base (deterministic) model and
``P^drop`` is a forward pass through the same network with TestTimeDropout
enabled. Poisoned samples — which rely on narrow trigger pathways —
exhibit a *large* confidence collapse and therefore an unusually LOW
``PSU`` value relative to clean validation samples.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf


def compute_psu(
    base_model: tf.keras.Model,
    drop_model: tf.keras.Model,
    images: np.ndarray,
    k: int = 10,
    batch_size: int = 128,
) -> np.ndarray:
    """Compute the PSU score for every image in ``images``.

    Parameters
    ----------
    base_model:
        Deterministic ResNet-18 (no test-time dropout).
    drop_model:
        Same architecture with ``TestTimeDropout`` layers; weights copied
        from ``base_model``.
    images:
        Inputs to score, shape ``(N, H, W, C)``.
    k:
        Number of stochastic forward passes through ``drop_model``.
    batch_size:
        Mini-batch size used to keep GPU memory bounded.
    """
    num_samples = len(images)
    psu_values = []

    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        batch = images[start:end]

        probs_clean = base_model.predict(batch, verbose=0)
        predicted_class = np.argmax(probs_clean, axis=1)
        pc_clean = probs_clean[np.arange(len(batch)), predicted_class]

        dropout_conf_sum = np.zeros(len(batch))
        for _ in range(k):
            probs_dropout = drop_model(batch, training=False).numpy()
            pc_dropout = probs_dropout[np.arange(len(batch)), predicted_class]
            dropout_conf_sum += pc_dropout

        pc_dropout_avg = dropout_conf_sum / k
        psu_values.append(pc_clean - pc_dropout_avg)

    return np.concatenate(psu_values)


def psbd_filter(
    base_model: tf.keras.Model,
    drop_model: tf.keras.Model,
    x_suspicious: np.ndarray,
    x_val_clean: np.ndarray,
    k: int = 10,
    threshold_percentile: float = 25.0,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Run the full PSBD scan and return the suspicion mask.

    Returns
    -------
    predicted_backdoor:
        Boolean array of shape ``(len(x_suspicious),)``; ``True`` means
        the sample has been flagged as a backdoor.
    threshold:
        The PSU threshold ``T`` derived from the clean validation set.
    psu_train:
        PSU score for every training sample (useful for diagnostics).
    psu_clean_val:
        PSU scores on the trusted validation pool.
    """
    print("Calculating PSU on the trusted clean validation set ...")
    psu_clean_val = compute_psu(base_model, drop_model, x_val_clean, k=k)

    print("Calculating PSU on the suspicious training set ...")
    psu_train = compute_psu(base_model, drop_model, x_suspicious, k=k)

    threshold = np.percentile(psu_clean_val, threshold_percentile)
    predicted_backdoor = psu_train < threshold

    return predicted_backdoor, float(threshold), psu_train, psu_clean_val
