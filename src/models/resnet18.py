"""ResNet-18 backbones used by the two phases of the defense pipeline.

* ``build_resnet18_base``: standard ResNet-18 used to train the infected
  baseline (Phase 1) and the final secure model (Phase 2).
* ``build_resnet18_dropout``: identical backbone with ``TestTimeDropout``
  layers inserted after every residual block to enable PSBD scanning.
"""

from __future__ import annotations

from tensorflow.keras import layers, models

from ..config import INPUT_SHAPE, NUM_CLASSES


class TestTimeDropout(layers.Dropout):
    """Dropout that is active at *inference* and disabled during training.

    PSBD relies on stochastic forward passes at test time. By inverting the
    standard ``training`` flag, we keep dropout off during ``model.fit`` but
    on during ``model.predict``/``model(inputs, training=False)``.
    """

    def call(self, inputs, training=None):
        return super().call(inputs, training=not training)


def residual_block(x, filters: int, downsample: bool = False):
    """A standard pre-activation-free ResNet basic block."""
    stride = 2 if downsample else 1

    y = layers.Conv2D(filters, 3, stride, padding="same")(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(filters, 3, 1, padding="same")(y)
    y = layers.BatchNormalization()(y)

    if downsample:
        x = layers.Conv2D(filters, 1, 2, padding="same")(x)

    out = layers.Add()([x, y])
    return layers.ReLU()(out)


def build_resnet18_base() -> models.Model:
    """Standard ResNet-18 for CIFAR-10 (used in Phase 1 baseline & Phase 2)."""
    inputs = layers.Input(shape=INPUT_SHAPE)
    x = layers.Conv2D(64, 3, 1, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, downsample=True)
    x = residual_block(x, 128)
    x = residual_block(x, 256, downsample=True)
    x = residual_block(x, 256)
    x = residual_block(x, 512, downsample=True)
    x = residual_block(x, 512)

    x = layers.GlobalAveragePooling2D()(x)
    # A single TestTimeDropout near the head keeps the deployment model
    # behaviourally equivalent to the dropout twin used for PSBD.
    x = TestTimeDropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return models.Model(inputs, outputs, name="resnet18_base")


def build_resnet18_dropout(dropout_rate: float = 0.5) -> models.Model:
    """ResNet-18 with ``TestTimeDropout`` after every residual block.

    Used as the perturbed twin during the PSBD scan in Phase 1.
    """
    inputs = layers.Input(shape=INPUT_SHAPE)
    x = layers.Conv2D(64, 3, 1, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = residual_block(x, 64)
    x = TestTimeDropout(dropout_rate)(x)
    x = residual_block(x, 64)
    x = TestTimeDropout(dropout_rate)(x)
    x = residual_block(x, 128, downsample=True)
    x = TestTimeDropout(dropout_rate)(x)
    x = residual_block(x, 128)
    x = TestTimeDropout(dropout_rate)(x)
    x = residual_block(x, 256, downsample=True)
    x = TestTimeDropout(dropout_rate)(x)
    x = residual_block(x, 256)
    x = TestTimeDropout(dropout_rate)(x)
    x = residual_block(x, 512, downsample=True)
    x = TestTimeDropout(dropout_rate)(x)
    x = residual_block(x, 512)
    x = TestTimeDropout(dropout_rate)(x)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return models.Model(inputs, outputs, name="resnet18_dropout")
