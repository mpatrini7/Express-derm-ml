from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


DUAL_POLICY_VERSION = "dual-center-scale-confirmation-v1"


@dataclass(frozen=True)
class DualPolicyResult:
    levels: np.ndarray
    melanoma_levels: np.ndarray
    broad_levels: np.ndarray
    melanoma_minimum: np.ndarray
    melanoma_maximum: np.ndarray
    broad_minimum: np.ndarray
    broad_maximum: np.ndarray


def _validate_thresholds(low: float, high: float, *, label: str) -> None:
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"{label} thresholds must be ordered in [0, 1]")


def _consensus_levels(
    probabilities: np.ndarray,
    *,
    low_threshold: float,
    high_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Center-scale probabilities must have shape [N, 3]")
    if not np.isfinite(values).all():
        raise ValueError("Center-scale probabilities must be finite")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("Center-scale probabilities must be in [0, 1]")
    minimum = values.min(axis=1)
    maximum = values.max(axis=1)
    levels = np.full(len(values), "uncertain", dtype="<U12")
    levels[maximum < low_threshold] = "low"
    levels[minimum >= high_threshold] = "high"
    return levels, minimum, maximum


def apply_dual_policy(
    melanoma_probabilities: np.ndarray,
    broad_probabilities: np.ndarray,
    *,
    melanoma_low_threshold: float,
    melanoma_high_threshold: float,
    broad_low_threshold: float,
    broad_high_threshold: float,
    broad_confirmation_threshold: float,
) -> DualPolicyResult:
    """Apply the immutable dual policy to aligned center-scale scores."""
    _validate_thresholds(
        melanoma_low_threshold,
        melanoma_high_threshold,
        label="Melanoma",
    )
    _validate_thresholds(
        broad_low_threshold,
        broad_high_threshold,
        label="Broad",
    )
    if not broad_high_threshold <= broad_confirmation_threshold <= 1.0:
        raise ValueError(
            "Broad confirmation threshold must be at least the broad high "
            "threshold and no greater than 1"
        )
    melanoma = np.asarray(melanoma_probabilities, dtype=np.float64)
    broad = np.asarray(broad_probabilities, dtype=np.float64)
    if melanoma.shape != broad.shape:
        raise ValueError("Dual-model probability shapes do not match")

    melanoma_levels, melanoma_minimum, melanoma_maximum = _consensus_levels(
        melanoma,
        low_threshold=melanoma_low_threshold,
        high_threshold=melanoma_high_threshold,
    )
    broad_levels, broad_minimum, broad_maximum = _consensus_levels(
        broad,
        low_threshold=broad_low_threshold,
        high_threshold=broad_high_threshold,
    )
    levels = np.full(len(melanoma), "review", dtype="<U18")
    levels[
        (melanoma_levels == "low") & (broad_levels == "low")
    ] = "no_elevated_signal"
    levels[
        (melanoma_levels == "high")
        & (broad_levels == "high")
        & (broad_minimum >= broad_confirmation_threshold)
    ] = "high_confirmed"
    return DualPolicyResult(
        levels=levels,
        melanoma_levels=melanoma_levels,
        broad_levels=broad_levels,
        melanoma_minimum=melanoma_minimum,
        melanoma_maximum=melanoma_maximum,
        broad_minimum=broad_minimum,
        broad_maximum=broad_maximum,
    )


def high_confirmation_metrics(
    targets: np.ndarray,
    levels: np.ndarray,
) -> dict[str, Any]:
    y_true = np.asarray(targets, dtype=np.int64)
    if y_true.ndim != 1 or not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("Targets must be a one-dimensional binary array")
    if len(y_true) != len(levels):
        raise ValueError("Targets and dual-policy levels do not align")
    predicted = np.asarray(levels) == "high_confirmed"
    positive = y_true == 1
    negative = ~positive
    true_positive = int(np.sum(predicted & positive))
    false_positive = int(np.sum(predicted & negative))
    false_negative = int(np.sum(~predicted & positive))
    true_negative = int(np.sum(~predicted & negative))
    return {
        "n": int(len(y_true)),
        "positive_n": int(positive.sum()),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "sensitivity": true_positive / max(true_positive + false_negative, 1),
        "specificity": true_negative / max(true_negative + false_positive, 1),
        "precision": true_positive / max(true_positive + false_positive, 1),
    }
