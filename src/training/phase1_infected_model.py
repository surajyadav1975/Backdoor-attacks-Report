"""Phase 1 — train the infected baseline used by the PSBD scanner."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf

from ..config import Phase1Config
from ..models import build_resnet18_base, build_resnet18_dropout


def train_infected_model(
    x_suspicious: np.ndarray,
    y_suspicious: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Phase1Config,
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Train the infected baseline and return ``(base_model, drop_model)``.

    The dropout twin shares weights with the base model so the PSBD filter
    can perform stochastic forward passes through architecturally
    perturbed copies of the *same* learned function.
    """
    print("=" * 60)
    print("PHASE 1: training infected baseline model")
    print("=" * 60)

    base_model = build_resnet18_base()
    base_model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    base_model.fit(
        x_suspicious,
        y_suspicious,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        validation_data=(x_test, y_test),
    )

    print("\nCopying weights into the dropout twin for PSBD scanning ...")
    drop_model = build_resnet18_dropout(dropout_rate=cfg.dropout_rate)
    drop_model.set_weights(base_model.get_weights())

    return base_model, drop_model
