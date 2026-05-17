"""Evaluation metrics reported in Tables 1 and 2 of the research paper."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from ..attacks import add_trigger_batch


def evaluate_psbd_filter(
    predicted_backdoor: np.ndarray,
    num_poison: int,
    threshold: float,
) -> Dict[str, float]:
    """Compute TPR / FPR for the Phase 1 PSBD filter.

    The first ``num_poison`` entries of the suspicious set are known to be
    poisoned by construction, so the ground-truth mask is trivial.
    """
    ground_truth = np.zeros(len(predicted_backdoor), dtype=bool)
    ground_truth[:num_poison] = True

    tp = int(np.sum(predicted_backdoor & ground_truth))
    fp = int(np.sum(predicted_backdoor & ~ground_truth))
    p = int(np.sum(ground_truth))
    n = int(len(ground_truth) - p)

    tpr = tp / p if p else 0.0
    fpr = fp / n if n else 0.0

    summary = {
        "threshold": threshold,
        "true_positives": tp,
        "false_positives": fp,
        "actual_positives": p,
        "actual_negatives": n,
        "tpr": tpr,
        "fpr": fpr,
    }

    print("\n" + "=" * 60)
    print("PSBD FILTER PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Threshold (T)             : {threshold:.4f}")
    print(f"TPR (poison caught)       : {tpr * 100:.2f}%   [target >= 95%]")
    print(f"FPR (clean data discarded): {fpr * 100:.2f}%   [target <  5%]")
    print("=" * 60)

    return summary


def evaluate_csr(
    secure_model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Return the Clean Success Rate on the original test set."""
    _, csr = secure_model.evaluate(x_test, y_test, verbose=0)
    return float(csr)


def evaluate_asr(
    secure_model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    attack: str,
    target_label: int,
) -> float:
    """Return the Attack Success Rate against the trained ``secure_model``.

    Only *non-target* test images are perturbed with the trigger, so the
    measurement is not biased by samples that already belong to the
    attacker's target class.
    """
    non_target_idx = np.where(y_test.flatten() != target_label)[0]
    x_test_non_target = x_test[non_target_idx]
    x_test_poisoned = add_trigger_batch(x_test_non_target, attack)
    y_hacker_goal = np.ones(len(x_test_poisoned)) * target_label

    _, asr = secure_model.evaluate(x_test_poisoned, y_hacker_goal, verbose=0)
    return float(asr)


def final_report(
    secure_model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    attack: str,
    target_label: int,
) -> Tuple[float, float]:
    """Print and return the CSR / ASR pair used in Table 2 of the paper."""
    csr = evaluate_csr(secure_model, x_test, y_test)
    asr = evaluate_asr(secure_model, x_test, y_test, attack, target_label)

    print("\n" + "=" * 60)
    print("FINAL PROJECT EVALUATION")
    print("=" * 60)
    print(f"--> Final CSR (Clean Accuracy)      : {csr * 100:.2f}%")
    print(f"--> Final ASR (Attack Success Rate) : {asr * 100:.2f}%")
    print("-" * 60)

    if asr < 0.05:
        print("SUCCESS: backdoor neutralized (ASR < 5%).")
    else:
        print("WARNING: backdoor still active.")

    if csr >= 0.90:
        print("SUCCESS: model utility preserved (CSR >= 90%).")
    else:
        print("WARNING: model suffered accuracy degradation.")
    print("-" * 60)

    return csr, asr
