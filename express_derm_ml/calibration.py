from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .metrics import (
    compute_binary_metrics,
    sigmoid,
    threshold_for_sensitivity,
)


class CalibrationError(RuntimeError):
    pass


def binary_log_loss(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    losses = -(
        targets * np.log(clipped)
        + (1 - targets) * np.log(1.0 - clipped)
    )
    return float(np.mean(losses))


def fit_temperature(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    minimum_temperature: float = 0.05,
    maximum_temperature: float = 20.0,
) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if logits.ndim != 1 or targets.ndim != 1 or len(logits) != len(targets):
        raise CalibrationError("Logits and targets must be equal-length vectors")
    if len(logits) == 0 or set(np.unique(targets)) != {0, 1}:
        raise CalibrationError(
            "Temperature fitting requires both binary targets"
        )
    if not np.isfinite(logits).all():
        raise CalibrationError("Calibration logits must be finite")
    if not 0 < minimum_temperature < maximum_temperature:
        raise CalibrationError("Invalid temperature bounds")

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        return binary_log_loss(targets, sigmoid(logits / temperature))

    result = minimize_scalar(
        objective,
        bounds=(
            math.log(minimum_temperature),
            math.log(maximum_temperature),
        ),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 500},
    )
    if not result.success or not math.isfinite(float(result.fun)):
        raise CalibrationError("Temperature optimization did not converge")
    return float(math.exp(float(result.x)))


def fit_affine_logistic_scaling(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    minimum_scale: float = 0.05,
    maximum_scale: float = 20.0,
    maximum_abs_bias: float = 20.0,
) -> tuple[float, float]:
    """Fit a monotonic scale and intercept on held-out validation logits."""
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if logits.ndim != 1 or targets.ndim != 1 or len(logits) != len(targets):
        raise CalibrationError("Logits and targets must be equal-length vectors")
    if len(logits) == 0 or set(np.unique(targets)) != {0, 1}:
        raise CalibrationError(
            "Affine calibration requires both binary targets"
        )
    if not np.isfinite(logits).all():
        raise CalibrationError("Calibration logits must be finite")
    if not 0 < minimum_scale < maximum_scale:
        raise CalibrationError("Invalid affine calibration scale bounds")
    if maximum_abs_bias <= 0 or not math.isfinite(maximum_abs_bias):
        raise CalibrationError("Invalid affine calibration bias bound")

    prevalence = float(targets.mean())
    initial_bias = math.log(prevalence / (1.0 - prevalence)) - float(
        np.median(logits)
    )
    initial_bias = float(
        np.clip(initial_bias, -maximum_abs_bias, maximum_abs_bias)
    )

    def objective(parameters: np.ndarray) -> float:
        scale = math.exp(float(parameters[0]))
        bias = float(parameters[1])
        return binary_log_loss(targets, sigmoid(scale * logits + bias))

    result = minimize(
        objective,
        x0=np.asarray([0.0, initial_bias], dtype=np.float64),
        method="L-BFGS-B",
        bounds=(
            (math.log(minimum_scale), math.log(maximum_scale)),
            (-maximum_abs_bias, maximum_abs_bias),
        ),
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success or not math.isfinite(float(result.fun)):
        raise CalibrationError("Affine calibration did not converge")
    scale = float(math.exp(float(result.x[0])))
    bias = float(result.x[1])
    if not math.isfinite(scale) or not math.isfinite(bias):
        raise CalibrationError("Affine calibration returned non-finite values")
    return scale, bias


def apply_calibration(
    logits: np.ndarray,
    calibration: dict[str, Any],
) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    method = calibration.get("method")
    if method == "affine_logistic_scaling":
        scale = float(calibration["logit_scale"])
        bias = float(calibration["logit_bias"])
        if scale <= 0 or not math.isfinite(scale) or not math.isfinite(bias):
            raise CalibrationError("Invalid affine calibration parameters")
        return sigmoid(scale * logits + bias)
    if method == "temperature_scaling":
        temperature = float(calibration["temperature"])
        if temperature <= 0 or not math.isfinite(temperature):
            raise CalibrationError("Invalid temperature calibration parameter")
        return sigmoid(logits / temperature)
    raise CalibrationError(f"Unsupported calibration method: {method}")


def threshold_for_specificity(
    targets: np.ndarray,
    probabilities: np.ndarray,
    minimum_specificity: float,
) -> float:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not 0.0 < minimum_specificity <= 1.0:
        raise CalibrationError("Minimum specificity must be in (0, 1]")
    negatives = targets == 0
    if not negatives.any():
        raise CalibrationError("Specificity selection requires negative cases")

    candidates = np.unique(
        np.concatenate(
            [
                probabilities,
                np.nextafter(probabilities, np.inf),
            ]
        )
    )
    valid: list[float] = []
    for threshold in candidates:
        predictions = probabilities >= threshold
        specificity = float(
            ((~predictions) & negatives).sum() / negatives.sum()
        )
        if specificity >= minimum_specificity:
            valid.append(float(threshold))
    if not valid:
        raise CalibrationError(
            "No threshold satisfies the requested minimum specificity"
        )
    return min(valid)


def fit_development_calibration(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    minimum_sensitivity: float,
    minimum_specificity: float,
    ece_bins: int,
    minimum_scale: float = 0.05,
    maximum_scale: float = 20.0,
    maximum_abs_bias: float = 20.0,
) -> dict[str, Any]:
    if not 0.0 < minimum_sensitivity <= 1.0:
        raise CalibrationError("Minimum sensitivity must be in (0, 1]")
    logit_scale, logit_bias = fit_affine_logistic_scaling(
        logits,
        targets,
        minimum_scale=minimum_scale,
        maximum_scale=maximum_scale,
        maximum_abs_bias=maximum_abs_bias,
    )
    uncalibrated = sigmoid(np.asarray(logits, dtype=np.float64))
    calibrated = sigmoid(
        logit_scale * np.asarray(logits, dtype=np.float64) + logit_bias
    )
    sensitivity_candidate = threshold_for_sensitivity(
        targets,
        calibrated,
        minimum_sensitivity,
    )
    specificity_candidate = threshold_for_specificity(
        targets,
        calibrated,
        minimum_specificity,
    )
    low_threshold = min(
        sensitivity_candidate,
        specificity_candidate,
    )
    high_threshold = max(
        sensitivity_candidate,
        specificity_candidate,
    )
    if not 0.0 <= low_threshold < high_threshold <= 1.0:
        raise CalibrationError(
            "Validation data did not produce two ordered attention thresholds"
        )

    return {
        "schema_version": 2,
        "method": "affine_logistic_scaling",
        "selection_split": "validation",
        "logit_scale": logit_scale,
        "logit_bias": logit_bias,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "selection": {
            "minimum_sensitivity": minimum_sensitivity,
            "minimum_specificity": minimum_specificity,
            "sensitivity_candidate": sensitivity_candidate,
            "specificity_candidate": specificity_candidate,
        },
        "uncalibrated_metrics": compute_binary_metrics(
            targets,
            uncalibrated,
            threshold=0.5,
            ece_bins=ece_bins,
        ),
        "calibrated_low_threshold_metrics": compute_binary_metrics(
            targets,
            calibrated,
            threshold=low_threshold,
            ece_bins=ece_bins,
        ),
        "calibrated_high_threshold_metrics": compute_binary_metrics(
            targets,
            calibrated,
            threshold=high_threshold,
            ece_bins=ece_bins,
        ),
        "uncalibrated_log_loss": binary_log_loss(targets, uncalibrated),
        "calibrated_log_loss": binary_log_loss(targets, calibrated),
        "threshold_status": "development_frozen",
        "thresholds_validated": False,
        "research_only": True,
    }
