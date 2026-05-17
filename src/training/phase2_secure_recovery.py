"""Phase 2 — train a pristine ResNet-18 on the verified clean pool only."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from ..config import Phase2Config
from ..models import build_resnet18_base


def _build_augmented_dataset(
    x_clean_pool: np.ndarray,
    y_clean_pool: np.ndarray,
    cfg: Phase2Config,
) -> tf.data.Dataset:
    """Return a shuffled, augmented ``tf.data`` pipeline over the clean pool."""
    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(
                height_factor=cfg.augment_translation,
                width_factor=cfg.augment_translation,
            ),
        ]
    )

    ds = tf.data.Dataset.from_tensor_slices((x_clean_pool, y_clean_pool))
    ds = ds.shuffle(10000).batch(cfg.batch_size)
    ds = ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def train_secure_model(
    x_clean_pool: np.ndarray,
    y_clean_pool: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Phase2Config,
) -> tf.keras.Model:
    """Train the final deployment model on the verified clean pool."""
    print("=" * 60)
    print("PHASE 2: training secure model on verified clean data")
    print("=" * 60)
    print(f"Clean pool size: {len(x_clean_pool)} images")

    train_ds = _build_augmented_dataset(x_clean_pool, y_clean_pool, cfg)

    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=cfg.lr_reduce_factor,
        patience=cfg.lr_reduce_patience,
        min_lr=cfg.min_lr,
        verbose=1,
    )

    # Section III.3.2: SGD + Nesterov momentum outperforms Adam on the
    # reduced clean pool, giving the +8% to +10% CSR boost reported in
    # the ablation (Section VI.3).
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=cfg.learning_rate,
        momentum=cfg.momentum,
        nesterov=cfg.nesterov,
    )

    secure_model = build_resnet18_base()
    secure_model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    secure_model.fit(
        train_ds,
        epochs=cfg.epochs,
        validation_data=(x_test, y_test),
        callbacks=[lr_scheduler],
    )

    return secure_model
