"""Central configuration for the PSBD two-phase defense pipeline.

All hyperparameters reported in the research paper live here so reviewers
can reproduce results by changing a single file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple


# Dataset / environment ----------------------------------------------------

CIFAR10_URL = (
    "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"
)
DATA_ROOT = os.environ.get("PSBD_DATA_ROOT", "./data")
TAR_PATH = os.path.join(DATA_ROOT, "cifar-10-python.tar.gz")
EXTRACT_PATH = os.path.join(DATA_ROOT, "cifar10_data")
BATCHES_DIR_NAME = "cifar-10-batches-py"

INPUT_SHAPE: Tuple[int, int, int] = (32, 32, 3)
NUM_CLASSES: int = 10


# Threat model -------------------------------------------------------------

# Fraction of training data the attacker can poison (Section IV.2 of paper).
POISON_RATIO: float = 0.10

# Target class chosen by the attacker (Class 0 = Airplane on CIFAR-10).
TARGET_LABEL: int = 0

# Which attack to run. Override at runtime with --attack CLI flag.
DEFAULT_ATTACK: str = "wanets"  # one of: "badnets", "blended", "wanets"
SUPPORTED_ATTACKS = ("badnets", "blended", "wanets")


# Phase 1 — Infected model + PSBD filter -----------------------------------

@dataclass
class Phase1Config:
    epochs: int = 55
    batch_size: int = 128
    learning_rate: float = 1e-3
    dropout_rate: float = 0.6          # p used inside TestTimeDropout
    psu_forward_passes: int = 10        # K in Section III.2.2
    threshold_percentile: float = 25.0  # 25th percentile of PSU(D_val)
    val_fraction: float = 0.05          # fraction of test set used as D_val


# Phase 2 — Secure recovery -----------------------------------------------
# Defaults follow Section III.3.2 of the paper:
#   SGD + Nesterov momentum 0.9, lr=0.1,
#   ReduceLROnPlateau(factor=0.2, patience=8), 80 epochs.

@dataclass
class Phase2Config:
    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 0.1
    momentum: float = 0.9
    nesterov: bool = True
    lr_reduce_factor: float = 0.2
    lr_reduce_patience: int = 8
    min_lr: float = 1e-5
    augment_translation: float = 0.10   # random translate up to 10% H/W


@dataclass
class PipelineConfig:
    attack: str = DEFAULT_ATTACK
    poison_ratio: float = POISON_RATIO
    target_label: int = TARGET_LABEL
    seed: int = 42
    results_dir: str = "results"
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)


DEFAULT_CONFIG = PipelineConfig()
