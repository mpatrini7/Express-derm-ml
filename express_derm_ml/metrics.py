from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))


def expected_calibration_error(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if bins < 2:
        raise ValueError("Calibration error requires at least two bins")
    if len(targets) != len(probabilities) or len(targets) == 0:
        raise ValueError(
            "Calibration targets and probabilities must be non-empty and aligned"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("Calibration probabilities must be finite")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Calibration probabilities must be in [0, 1]")

    bin_indices = np.minimum(
        (probabilities * bins).astype(int),
        bins - 1,
    )
    error = 0.0
    for bin_index in range(bins):
        members = bin_indices == bin_index
        if not members.any():
            continue
        confidence = float(probabilities[members].mean())
        observed = float(targets[members].mean())
        error += abs(confidence - observed) * float(members.mean())
    return error


def compute_binary_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
    *,
    ece_bins: int = 10,
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    negative_predictive_value = tn / max(tn + fn, 1)
    f1_score = 2 * precision * sensitivity / max(
        precision + sensitivity,
        np.finfo(float).eps,
    )

    return {
        "roc_auc": float(roc_auc_score(targets, probabilities)),
        "pr_auc": float(average_precision_score(targets, probabilities)),
        "balanced_accuracy": float(
            balanced_accuracy_score(targets, predictions)
        ),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "negative_predictive_value": float(negative_predictive_value),
        "f1_score": float(f1_score),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "prevalence": float(targets.mean()),
        "brier_score": float(brier_score_loss(targets, probabilities)),
        "expected_calibration_error": expected_calibration_error(
            targets,
            probabilities,
            bins=ece_bins,
        ),
        "threshold": float(threshold),
        "n": int(len(targets)),
        "positive_n": int(targets.sum()),
    }


def grouped_bootstrap_intervals(
    targets: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    *,
    threshold: float,
    samples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 2026,
) -> dict[str, object]:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    groups = np.asarray(groups, dtype=str)
    if (
        len(targets) != len(probabilities)
        or len(targets) != len(groups)
        or len(targets) == 0
    ):
        raise ValueError("Bootstrap inputs must be non-empty and aligned")
    if set(np.unique(targets)) != {0, 1}:
        raise ValueError("Bootstrap requires both binary targets")
    if samples < 100:
        raise ValueError("Bootstrap requires at least 100 samples")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("Confidence level must be in (0, 1)")

    unique_groups = np.unique(groups)
    group_indices = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    positive_groups = np.asarray(
        [
            group
            for group in unique_groups
            if targets[group_indices[group]].max() == 1
        ]
    )
    negative_groups = np.asarray(
        [
            group
            for group in unique_groups
            if targets[group_indices[group]].max() == 0
        ]
    )
    if len(positive_groups) == 0 or len(negative_groups) == 0:
        raise ValueError("Grouped bootstrap requires both group strata")
    rng = np.random.default_rng(seed)
    values = {
        key: np.empty(samples, dtype=np.float64)
        for key in (
            "roc_auc",
            "pr_auc",
            "sensitivity",
            "specificity",
            "precision",
        )
    }
    for sample_index in range(samples):
        sampled_groups = np.concatenate(
            (
                rng.choice(
                    positive_groups,
                    size=len(positive_groups),
                    replace=True,
                ),
                rng.choice(
                    negative_groups,
                    size=len(negative_groups),
                    replace=True,
                ),
            )
        )
        indices = np.concatenate(
            [group_indices[group] for group in sampled_groups]
        )
        sample_targets = targets[indices]
        sample_probabilities = probabilities[indices]
        sample_metrics = compute_binary_metrics(
            sample_targets,
            sample_probabilities,
            threshold=threshold,
        )
        for key in values:
            values[key][sample_index] = float(sample_metrics[key])

    tail = (1.0 - confidence_level) / 2.0
    return {
        "method": "grouped_stratified_bootstrap",
        "samples": samples,
        "confidence_level": confidence_level,
        "seed": seed,
        "intervals": {
            key: {
                "low": float(np.quantile(metric_values, tail)),
                "high": float(np.quantile(metric_values, 1.0 - tail)),
            }
            for key, metric_values in values.items()
        },
    }


def threshold_for_sensitivity(
    targets: np.ndarray,
    probabilities: np.ndarray,
    minimum_sensitivity: float,
) -> float:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not 0.0 < minimum_sensitivity <= 1.0:
        raise ValueError("Minimum sensitivity must be in (0, 1]")
    if len(targets) != len(probabilities) or len(targets) == 0:
        raise ValueError(
            "Threshold targets and probabilities must be non-empty and aligned"
        )
    positives = targets == 1
    if not positives.any():
        raise ValueError("Sensitivity selection requires positive cases")

    candidates = np.unique(probabilities)
    valid: list[float] = []
    for threshold in candidates:
        predictions = probabilities >= threshold
        sensitivity = float(
            (predictions & positives).sum() / positives.sum()
        )
        if sensitivity >= minimum_sensitivity:
            valid.append(float(threshold))
    if not valid:
        raise ValueError(
            "No threshold satisfies the requested minimum sensitivity"
        )
    return max(valid)
