from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from .artifacts import create_new_directory, write_json_exclusive, write_npz_exclusive
from .calibration import apply_calibration, fit_development_calibration
from .common import sha256_file
from .dataset import DEPLOYMENT_PREPROCESSING_VERSION
from .manifest import canonical_manifest_sha256, read_manifest
from .metrics import compute_binary_metrics, sigmoid


STACKER_CANDIDATES = (0.01, 0.1, 1.0, 10.0)
STACKER_FEATURE_KINDS = ("linear", "quadratic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a patient-grouped logistic stacker on frozen teacher "
            "validation predictions and evaluate it once on test."
        ),
    )
    parser.add_argument("--ensemble-evaluation-dir", required=True)
    parser.add_argument("--teacher-targets-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--additional-run-dir",
        action="append",
        default=[],
        help=(
            "Completed single-model run with frozen validation/test "
            "predictions to add as a stacker feature."
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--minimum-sensitivity", type=float, default=0.9)
    parser.add_argument("--minimum-specificity", type=float, default=0.9)
    parser.add_argument("--ece-bins", type=int, default=10)
    return parser.parse_args()


def _load_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as bundle:
        return {name: bundle[name].copy() for name in bundle.files}


def _standardize_component_logits(
    component_logits: np.ndarray,
    component_order: list[str],
    normalization: dict[str, dict[str, float]],
) -> np.ndarray:
    if component_logits.ndim != 2:
        raise RuntimeError("Component logits must be a two-dimensional matrix")
    if component_logits.shape[1] != len(component_order):
        raise RuntimeError("Component logits do not match component order")
    columns = []
    for index, name in enumerate(component_order):
        values = normalization.get(name)
        if not isinstance(values, dict):
            raise RuntimeError(f"Missing validation normalization for {name}")
        standard_deviation = float(values["standard_deviation"])
        if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
            raise RuntimeError(f"Invalid validation normalization for {name}")
        columns.append(
            (component_logits[:, index] - float(values["mean"]))
            / standard_deviation
        )
    features = np.stack(columns, axis=1).astype(np.float64)
    if not np.isfinite(features).all():
        raise RuntimeError("Stacker features contain non-finite values")
    return features


def _verify_prediction_alignment(
    split_frame,
    predictions: dict[str, np.ndarray],
) -> None:
    expected_names = split_frame["image_name"].to_numpy(dtype=str)
    expected_targets = split_frame["target"].to_numpy(dtype=np.int64)
    if not np.array_equal(predictions["image_names"].astype(str), expected_names):
        raise RuntimeError("Prediction image order does not match manifest")
    if not np.array_equal(predictions["targets"].astype(np.int64), expected_targets):
        raise RuntimeError("Prediction targets do not match manifest")


def _transform_features(features: np.ndarray, kind: str) -> np.ndarray:
    if kind == "linear":
        return features
    if kind != "quadratic":
        raise ValueError(f"Unsupported stacker feature kind: {kind}")
    columns = [features]
    for left in range(features.shape[1]):
        for right in range(left, features.shape[1]):
            columns.append(
                (features[:, left] * features[:, right]).reshape(-1, 1)
            )
    return np.column_stack(columns)


def _attention_counts(
    probabilities: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> dict[str, int]:
    return {
        "low": int((probabilities < low_threshold).sum()),
        "intermediate": int(
            (
                (probabilities >= low_threshold)
                & (probabilities < high_threshold)
            ).sum()
        ),
        "high": int((probabilities >= high_threshold).sum()),
    }


def main() -> None:
    args = parse_args()
    if args.folds < 2 or args.ece_bins <= 0:
        raise ValueError("Folds must be at least two and ECE bins positive")
    evaluation_dir = Path(args.ensemble_evaluation_dir).resolve()
    teacher_targets_dir = Path(args.teacher_targets_dir).resolve()
    metrics_path = evaluation_dir / "metrics.json"
    evaluation = json.loads(metrics_path.read_text(encoding="utf-8"))
    if evaluation.get("status") != "complete" or evaluation.get(
        "candidate_type"
    ) != "teacher_ensemble":
        raise RuntimeError("Teacher ensemble evaluation is incomplete")
    if evaluation.get("evaluation_preprocessing") != (
        DEPLOYMENT_PREPROCESSING_VERSION
    ):
        raise RuntimeError("Teacher ensemble preprocessing mismatch")

    receipt_path = teacher_targets_dir / "teacher_targets.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    component_order = [str(value) for value in evaluation["component_order"]]
    normalization = receipt.get("validation_logit_normalization", {})
    if set(normalization) != set(component_order):
        raise RuntimeError("Teacher normalization components are incomplete")

    manifest_path = Path(args.manifest)
    frame = read_manifest(manifest_path)
    manifest_sha256 = canonical_manifest_sha256(frame)
    if manifest_sha256 != evaluation.get("manifest_sha256"):
        raise RuntimeError("Evaluation and stacker manifests do not match")
    split_frames = {
        split: frame.loc[frame["split"] == split].reset_index(drop=True)
        for split in ("validation", "test")
    }
    predictions = {
        split: _load_predictions(evaluation_dir / f"predictions_{split}.npz")
        for split in split_frames
    }
    for split in split_frames:
        _verify_prediction_alignment(split_frames[split], predictions[split])

    additional_runs = []
    for index, run_value in enumerate(args.additional_run_dir, start=1):
        run_dir = Path(run_value).resolve()
        run_path = run_dir / "run.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run.get("status") != "complete" or run.get(
            "evaluation_preprocessing"
        ) != DEPLOYMENT_PREPROCESSING_VERSION:
            raise RuntimeError("Additional stacker run is incomplete")
        name = f"additional_{index}:{run_dir.name}"
        run_predictions = {
            split: _load_predictions(run_dir / f"predictions_{split}.npz")
            for split in split_frames
        }
        for split in split_frames:
            _verify_prediction_alignment(
                split_frames[split],
                run_predictions[split],
            )
        validation_logits = run_predictions["validation"]["logits"].astype(
            np.float64
        )
        standard_deviation = float(validation_logits.std())
        if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
            raise RuntimeError("Additional run validation logits are constant")
        normalization[name] = {
            "mean": float(validation_logits.mean()),
            "standard_deviation": standard_deviation,
        }
        component_order.append(name)
        for split in split_frames:
            predictions[split]["component_logits"] = np.column_stack(
                [
                    predictions[split]["component_logits"],
                    run_predictions[split]["logits"].astype(np.float64),
                ]
            )
        additional_runs.append(
            {
                "name": name,
                "run_json_sha256": sha256_file(run_path),
                "checkpoint_sha256": run["checkpoint_sha256"],
                "manifest_sha256": run["manifest_sha256"],
            }
        )

    validation_features = _standardize_component_logits(
        predictions["validation"]["component_logits"],
        component_order,
        normalization,
    )
    validation_targets = predictions["validation"]["targets"].astype(np.int64)
    validation_groups = split_frames["validation"]["group_id"].to_numpy(
        dtype=str
    )
    splitter = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    candidate_reports = []
    best_candidate: tuple[float, float, float] | None = None
    best_c: float | None = None
    best_feature_kind: str | None = None
    for feature_kind in STACKER_FEATURE_KINDS:
        transformed = _transform_features(validation_features, feature_kind)
        for regularization in STACKER_CANDIDATES:
            out_of_fold_logits = np.empty(
                len(validation_targets), dtype=np.float64
            )
            for train_indices, holdout_indices in splitter.split(
                transformed,
                validation_targets,
                validation_groups,
            ):
                model = LogisticRegression(
                    C=regularization,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=args.seed,
                )
                model.fit(
                    transformed[train_indices],
                    validation_targets[train_indices],
                )
                out_of_fold_logits[holdout_indices] = model.decision_function(
                    transformed[holdout_indices]
                )
            metrics = compute_binary_metrics(
                validation_targets,
                sigmoid(out_of_fold_logits),
                threshold=0.5,
                ece_bins=args.ece_bins,
            )
            candidate_reports.append(
                {
                    "feature_kind": feature_kind,
                    "regularization_c": regularization,
                    "out_of_fold_metrics": metrics,
                }
            )
            rank = (
                float(metrics["pr_auc"]),
                float(metrics["roc_auc"]),
                -regularization,
            )
            if best_candidate is None or rank > best_candidate:
                best_candidate = rank
                best_c = regularization
                best_feature_kind = feature_kind
    if best_c is None:
        raise RuntimeError("No stacker candidate was selected")
    if best_feature_kind is None:  # pragma: no cover - paired invariant
        raise RuntimeError("No stacker feature kind was selected")

    transformed_validation = _transform_features(
        validation_features,
        best_feature_kind,
    )
    stacker = LogisticRegression(
        C=best_c,
        solver="lbfgs",
        max_iter=2000,
        random_state=args.seed,
    )
    stacker.fit(transformed_validation, validation_targets)
    split_logits = {
        "validation": stacker.decision_function(transformed_validation),
        "test": stacker.decision_function(
            _transform_features(
                _standardize_component_logits(
                    predictions["test"]["component_logits"],
                    component_order,
                    normalization,
                ),
                best_feature_kind,
            )
        ),
    }
    calibration = fit_development_calibration(
        split_logits["validation"],
        validation_targets,
        minimum_sensitivity=args.minimum_sensitivity,
        minimum_specificity=args.minimum_specificity,
        ece_bins=args.ece_bins,
        minimum_scale=0.05,
        maximum_scale=20.0,
        maximum_abs_bias=20.0,
    )
    calibration.update(
        {
            "candidate_type": "logistic_teacher_stacker",
            "evaluation_preprocessing": DEPLOYMENT_PREPROCESSING_VERSION,
            "selection_split": "validation",
            "regularization_c": best_c,
            "feature_kind": best_feature_kind,
            "component_order": component_order,
            "feature_normalization": normalization,
            "coefficients": stacker.coef_.flatten().astype(float).tolist(),
            "intercept": float(stacker.intercept_[0]),
        }
    )

    output_dir = create_new_directory(args.output_dir)
    write_json_exclusive(output_dir / "calibration.json", calibration)
    split_reports = {}
    for split, logits in split_logits.items():
        targets = predictions[split]["targets"].astype(np.int64)
        uncalibrated = sigmoid(logits)
        calibrated = apply_calibration(logits, calibration)
        low_threshold = float(calibration["low_threshold"])
        high_threshold = float(calibration["high_threshold"])
        split_reports[split] = {
            "records": int(len(targets)),
            "positive_records": int(targets.sum()),
            "uncalibrated": compute_binary_metrics(
                targets,
                uncalibrated,
                threshold=0.5,
                ece_bins=args.ece_bins,
            ),
            "low_threshold": compute_binary_metrics(
                targets,
                calibrated,
                threshold=low_threshold,
                ece_bins=args.ece_bins,
            ),
            "high_threshold": compute_binary_metrics(
                targets,
                calibrated,
                threshold=high_threshold,
                ece_bins=args.ece_bins,
            ),
            "attention_counts": _attention_counts(
                calibrated,
                low_threshold,
                high_threshold,
            ),
        }
        write_npz_exclusive(
            output_dir / f"predictions_{split}.npz",
            logits=logits,
            uncalibrated_probabilities=uncalibrated,
            calibrated_probabilities=calibrated,
            targets=targets,
            image_names=predictions[split]["image_names"].astype(str),
        )

    report = {
        "schema_version": 1,
        "status": "complete",
        "candidate_type": "logistic_teacher_stacker",
        "selection_split": "validation",
        "evaluation_preprocessing": DEPLOYMENT_PREPROCESSING_VERSION,
        "manifest_sha256": manifest_sha256,
        "source_evaluation_sha256": sha256_file(metrics_path),
        "teacher_targets_receipt_sha256": sha256_file(receipt_path),
        "component_order": component_order,
        "additional_runs": additional_runs,
        "cross_validation": {
            "method": "stratified_group_k_fold",
            "group": "group_id",
            "folds": args.folds,
            "seed": args.seed,
            "candidates": candidate_reports,
            "selected_regularization_c": best_c,
            "selected_feature_kind": best_feature_kind,
        },
        "feature_normalization": normalization,
        "coefficients": stacker.coef_.flatten().astype(float).tolist(),
        "intercept": float(stacker.intercept_[0]),
        "low_threshold": float(calibration["low_threshold"]),
        "high_threshold": float(calibration["high_threshold"]),
        "splits": split_reports,
        "thresholds_validated": False,
        "research_only": True,
    }
    write_json_exclusive(output_dir / "metrics.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
