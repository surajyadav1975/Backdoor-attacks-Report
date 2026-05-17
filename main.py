"""End-to-end driver for the PSBD two-phase defense pipeline.

Usage examples
--------------

Run the full pipeline against the WaNets attack with default settings:

    python main.py --attack wanets

Reproduce the BadNets row of Table 1 / Table 2:

    python main.py --attack badnets --poison-ratio 0.10 --target-label 0

Quick smoke test (few epochs):

    python main.py --attack blended --phase1-epochs 2 --phase2-epochs 2
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import tensorflow as tf

from src.attacks import inject_poison
from src.config import (
    DEFAULT_CONFIG,
    Phase1Config,
    Phase2Config,
    PipelineConfig,
    SUPPORTED_ATTACKS,
)
from src.data_loader import load_cifar10
from src.defense import psbd_filter
from src.evaluation import evaluate_psbd_filter, final_report
from src.training import train_infected_model, train_secure_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PSBD two-phase backdoor defense pipeline (CIFAR-10)."
    )
    p.add_argument(
        "--attack",
        choices=SUPPORTED_ATTACKS,
        default=DEFAULT_CONFIG.attack,
        help="Which backdoor attack to inject.",
    )
    p.add_argument(
        "--poison-ratio",
        type=float,
        default=DEFAULT_CONFIG.poison_ratio,
        help="Fraction of training set to poison (default: 0.10).",
    )
    p.add_argument(
        "--target-label",
        type=int,
        default=DEFAULT_CONFIG.target_label,
        help="Attacker's target class (default: 0 / 'airplane').",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)

    p.add_argument("--phase1-epochs", type=int, default=Phase1Config.epochs)
    p.add_argument("--phase2-epochs", type=int, default=Phase2Config.epochs)
    p.add_argument(
        "--dropout-rate",
        type=float,
        default=Phase1Config.dropout_rate,
        help="p for TestTimeDropout in the PSBD scan (default: 0.6).",
    )
    p.add_argument(
        "--results-dir",
        default=DEFAULT_CONFIG.results_dir,
        help="Folder where the JSON metrics report is saved.",
    )

    return p.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    phase1 = Phase1Config(
        epochs=args.phase1_epochs,
        dropout_rate=args.dropout_rate,
    )
    phase2 = Phase2Config(epochs=args.phase2_epochs)
    return PipelineConfig(
        attack=args.attack,
        poison_ratio=args.poison_ratio,
        target_label=args.target_label,
        seed=args.seed,
        results_dir=args.results_dir,
        phase1=phase1,
        phase2=phase2,
    )


def run_pipeline(cfg: PipelineConfig) -> dict:
    set_seeds(cfg.seed)

    # 1. Load dataset --------------------------------------------------
    x_train, y_train, x_test, y_test = load_cifar10()

    # 2. Inject poison into the untrusted training set ----------------
    x_suspicious, y_suspicious, num_poison = inject_poison(
        x_train,
        y_train,
        attack=cfg.attack,
        poison_ratio=cfg.poison_ratio,
        target_label=cfg.target_label,
    )

    # Trusted validation pool drawn from the (clean) test set.
    num_val = int(cfg.phase1.val_fraction * len(x_train))
    x_val_clean = x_test[:num_val]

    # 3. Phase 1 — train infected baseline & dropout twin -------------
    base_model, drop_model = train_infected_model(
        x_suspicious, y_suspicious, x_test, y_test, cfg.phase1
    )

    # 4. Phase 1 — PSBD scan ------------------------------------------
    predicted_backdoor, threshold, _, _ = psbd_filter(
        base_model,
        drop_model,
        x_suspicious=x_suspicious,
        x_val_clean=x_val_clean,
        k=cfg.phase1.psu_forward_passes,
        threshold_percentile=cfg.phase1.threshold_percentile,
    )

    psbd_summary = evaluate_psbd_filter(predicted_backdoor, num_poison, threshold)

    # 5. Build the verified clean pool --------------------------------
    x_clean_pool = x_suspicious[~predicted_backdoor]
    y_clean_pool = y_suspicious[~predicted_backdoor]
    x_polluted_pool = x_suspicious[predicted_backdoor]

    print(
        f"\nClean pool: {len(x_clean_pool)} images | "
        f"Polluted pool: {len(x_polluted_pool)} images"
    )

    # 6. Phase 2 — train secure model on the clean pool ---------------
    secure_model = train_secure_model(
        x_clean_pool, y_clean_pool, x_test, y_test, cfg.phase2
    )

    # 7. Final evaluation ---------------------------------------------
    csr, asr = final_report(
        secure_model,
        x_test,
        y_test,
        attack=cfg.attack,
        target_label=cfg.target_label,
    )

    # 8. Persist a machine-readable summary ---------------------------
    os.makedirs(cfg.results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        cfg.results_dir, f"results_{cfg.attack}_{timestamp}.json"
    )
    report = {
        "attack": cfg.attack,
        "poison_ratio": cfg.poison_ratio,
        "target_label": cfg.target_label,
        "phase1": psbd_summary,
        "final_csr": csr,
        "final_asr": asr,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved metrics report to: {out_path}")

    return report


if __name__ == "__main__":
    cfg = build_config(parse_args())
    run_pipeline(cfg)
