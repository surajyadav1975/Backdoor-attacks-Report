"""Dispatcher for trigger application and full poisoning of a training set."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .badnets import badnets_trigger
from .blended import blended_trigger
from .wanets import wanets_trigger


def add_trigger_batch(images: np.ndarray, attack: str) -> np.ndarray:
    """Apply the trigger associated with ``attack`` to a batch of images."""
    attack = attack.lower()
    if attack == "badnets":
        return badnets_trigger(images)
    if attack == "blended":
        return blended_trigger(images)
    if attack == "wanets":
        return wanets_trigger(images)
    raise ValueError(
        f"Unknown attack '{attack}'. Expected one of: badnets, blended, wanets."
    )


def inject_poison(
    x_train: np.ndarray,
    y_train: np.ndarray,
    attack: str,
    poison_ratio: float,
    target_label: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Build the untrusted training set ``D_train`` used by the defender.

    The first ``num_poison`` images are poisoned with ``attack`` and relabeled
    to ``target_label``. The remaining clean images keep their original labels.
    The order is preserved so that the defender's ground-truth mask is simply
    ``[True]*num_poison + [False]*(N-num_poison)``.

    Returns
    -------
    x_suspicious, y_suspicious, num_poison
    """
    num_poison = int(len(x_train) * poison_ratio)
    print(
        f"Injecting {attack.upper()} attack into {num_poison} of {len(x_train)} "
        f"images (poison ratio = {poison_ratio:.0%})."
    )

    x_backdoor = add_trigger_batch(x_train[:num_poison], attack)
    x_clean_train = x_train[num_poison:]
    x_suspicious = np.concatenate([x_backdoor, x_clean_train], axis=0)

    y_hacker = np.ones((num_poison, 1)) * target_label
    y_clean_train = y_train[num_poison:]
    y_suspicious = np.concatenate([y_hacker, y_clean_train], axis=0).flatten()

    return x_suspicious, y_suspicious, num_poison
