from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import (
    create_new_directory,
    write_json_exclusive,
    write_npz_exclusive,
)
from .calibration import apply_calibration, fit_development_calibration
from .common import sha256_file
from .manifest import read_manifest
from .metrics import compute_binary_metrics, grouped_bootstrap_intervals, sigmoid


PREDICTION_FILES = {
    "validation": (
        "predictions_validation.npz",
        "targets",
    ),
    "test": (
        "predictions_test.npz",
        "targets",
    ),
    "milk10k_dermoscopic": (
        "metrics_external_milk10k_dermoscopic.npz",
        "melanoma_targets",
    ),
    "milk10k_clinical": (
        "metrics_external_milk10k_clinical.npz",
        "melanoma_targets",
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        return {name: source[name].copy() for name in source.files}


def standardized_logit_average(
    logits: dict[str, np.ndarray],
    normalization: dict[str, dict[str, float]],
    members: list[str],
) -> np.ndarray:
    if not members or len(set(members)) != len(members):
        raise ValueError("Ensemble members must be unique and non-empty")
    standardized = []
    for member in members:
        values = np.asarray(logits[member], dtype=np.float64)
        mean = float(normalization[member]["mean"])
        standard_deviation = float(
            normalization[member]["standard_deviation"]
        )
        if standard_deviation <= 0.0 or not np.isfinite(standard_deviation):
            raise ValueError(f"Invalid validation logit spread: {member}")
        standardized.append((values - mean) / standard_deviation)
    return np.mean(standardized, axis=0)


def select_validation_ensemble(
    validation_logits: dict[str, np.ndarray],
    targets: np.ndarray,
    normalization: dict[str, dict[str, float]],
    candidate_sets: dict[str, list[str]],
) -> tuple[str, list[dict[str, Any]]]:
    if not candidate_sets:
        raise ValueError("At least one ensemble candidate set is required")
    results = []
    for name, members in candidate_sets.items():
        unknown = sorted(set(members) - set(validation_logits))
        if unknown:
            raise ValueError(
                f"Ensemble candidate {name!r} has unknown members: {unknown}"
            )
        scores = standardized_logit_average(
            validation_logits,
            normalization,
            members,
        )
        metrics = compute_binary_metrics(targets, sigmoid(scores))
        results.append(
            {
                "name": name,
                "members": members,
                "model_count": len(members),
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
            }
        )
    ordered = sorted(
        results,
        key=lambda result: (
            float(result["pr_auc"]),
            float(result["roc_auc"]),
            -int(result["model_count"]),
            str(result["name"]),
        ),
        reverse=True,
    )
    return str(ordered[0]["name"]), ordered


def _component_summary(name: str, run_dir: Path) -> dict[str, Any]:
    required = [
        run_dir / "run.json",
        run_dir / "best.pt",
        run_dir / "calibration.json",
        run_dir / "manifest.csv",
        run_dir / "predictions_validation.npz",
        run_dir / "predictions_test.npz",
        run_dir / "metrics_external_milk10k_dermoscopic.json",
        run_dir / "metrics_external_milk10k_dermoscopic.npz",
        run_dir / "metrics_external_milk10k_clinical.json",
        run_dir / "metrics_external_milk10k_clinical.npz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Ensemble component artifacts are missing: {missing}")
    run = _read_json(run_dir / "run.json")
    checkpoint_sha256 = sha256_file(run_dir / "best.pt")
    if run.get("status") != "complete":
        raise RuntimeError(f"Ensemble component is incomplete: {name}")
    if run.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError(f"Ensemble component checkpoint mismatch: {name}")
    external_manifest_hashes = {}
    for context in ("dermoscopic", "clinical"):
        report_path = run_dir / f"metrics_external_milk10k_{context}.json"
        report = _read_json(report_path)
        if report.get("run_checkpoint_sha256") != checkpoint_sha256:
            raise RuntimeError(
                f"Ensemble component external checkpoint mismatch: {name}"
            )
        if report.get("thresholds_validated") is not False:
            raise RuntimeError("Component thresholds must remain unvalidated")
        external_manifest_hashes[context] = report[
            "external_manifest_sha256"
        ]
    prediction_hashes = {
        context: sha256_file(run_dir / filename)
        for context, (filename, _) in PREDICTION_FILES.items()
    }
    return {
        "name": name,
        "run_dir": str(run_dir),
        "run_sha256": sha256_file(run_dir / "run.json"),
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_sha256": sha256_file(run_dir / "calibration.json"),
        "manifest_sha256": run["manifest_sha256"],
        "prediction_sha256": prediction_hashes,
        "external_manifest_sha256": external_manifest_hashes,
    }


def _aligned_predictions(
    components: dict[str, Path],
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    loaded: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for context, (filename, target_key) in PREDICTION_FILES.items():
        context_data = {
            name: _load_npz(run_dir / filename)
            for name, run_dir in components.items()
        }
        reference = context_data[next(iter(components))]
        required_keys = {"logits", "image_names", target_key}
        if not required_keys.issubset(reference):
            raise RuntimeError(
                f"Reference predictions are incomplete for {context}"
            )
        for name, values in context_data.items():
            if not required_keys.issubset(values):
                raise RuntimeError(
                    f"Component predictions are incomplete: {context}/{name}"
                )
            if not np.array_equal(
                values["image_names"], reference["image_names"]
            ):
                raise RuntimeError(
                    f"Component image order mismatch: {context}/{name}"
                )
            if not np.array_equal(values[target_key], reference[target_key]):
                raise RuntimeError(
                    f"Component targets mismatch: {context}/{name}"
                )
            logits = np.asarray(values["logits"], dtype=np.float64)
            if logits.ndim != 1 or not np.isfinite(logits).all():
                raise RuntimeError(
                    f"Component logits are invalid: {context}/{name}"
                )
        if context.startswith("milk10k"):
            for key in (
                "lesion_ids",
                "broad_malignancy_targets",
                "diagnosis_classes",
            ):
                if key not in reference:
                    raise RuntimeError(f"External predictions are missing {key}")
                for name, values in context_data.items():
                    if not np.array_equal(values[key], reference[key]):
                        raise RuntimeError(
                            f"External metadata mismatch: {context}/{name}/{key}"
                        )
        loaded[context] = context_data
    return loaded


def _patient_groups(
    run_dir: Path,
    split: str,
    image_names: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    manifest = read_manifest(run_dir / "manifest.csv")
    frame = manifest.loc[manifest["split"] == split].copy()
    if frame["image_name"].duplicated().any():
        raise RuntimeError(f"Duplicate image names in {split} manifest")
    indexed = frame.set_index("image_name")
    requested = [str(value) for value in image_names]
    missing = sorted(set(requested) - set(indexed.index.astype(str)))
    if missing:
        raise RuntimeError(f"Prediction images are missing from manifest: {missing}")
    ordered = indexed.loc[requested]
    if not np.array_equal(
        ordered["target"].to_numpy(dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
    ):
        raise RuntimeError(f"Prediction targets do not match {split} manifest")
    return ordered["group_id"].to_numpy(dtype=str)


def _evaluation_identity(
    image_names: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for image_name, target, group in zip(image_names, targets, groups):
        digest.update(
            f"{image_name}\0{int(target)}\0{group}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _evaluate(
    scores: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    calibration: dict[str, Any],
    *,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
    resampling_unit: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    probabilities = apply_calibration(scores, calibration)
    low_threshold = float(calibration["low_threshold"])
    high_threshold = float(calibration["high_threshold"])
    low_metrics = compute_binary_metrics(
        targets,
        probabilities,
        threshold=low_threshold,
    )
    high_metrics = compute_binary_metrics(
        targets,
        probabilities,
        threshold=high_threshold,
    )
    intervals = {}
    for label, threshold in (
        ("low_threshold", low_threshold),
        ("high_threshold", high_threshold),
    ):
        intervals[label] = grouped_bootstrap_intervals(
            targets,
            probabilities,
            groups,
            threshold=threshold,
            samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed,
        )
        intervals[label]["resampling_unit"] = resampling_unit
    return (
        {
            "uncalibrated_metrics": compute_binary_metrics(
                targets,
                sigmoid(scores),
            ),
            "calibrated_low_threshold_metrics": low_metrics,
            "calibrated_high_threshold_metrics": high_metrics,
            "confidence_intervals": intervals,
        },
        {
            "ensemble_scores": scores,
            "calibrated_probabilities": probabilities,
            "targets": targets,
        },
    )


def _parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise ValueError("Components must use NAME=RUN_DIR")
    return name.strip(), Path(path)


def _parse_candidate_set(value: str) -> tuple[str, list[str]]:
    name, separator, members = value.partition("=")
    parsed_members = [member.strip() for member in members.split(",")]
    if (
        not separator
        or not name.strip()
        or not parsed_members
        or any(not member for member in parsed_members)
    ):
        raise ValueError("Candidate sets must use NAME=MEMBER_A,MEMBER_B")
    return name.strip(), parsed_members


def build_ensemble_candidate(
    *,
    components: dict[str, Path],
    candidate_sets: dict[str, list[str]],
    output_dir: Path,
    bootstrap_samples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 2026,
) -> dict[str, Any]:
    if len(components) < 2:
        raise ValueError("An ensemble requires at least two components")
    if bootstrap_samples < 100:
        raise ValueError("Bootstrap requires at least 100 samples")
    summaries = {
        name: _component_summary(name, path)
        for name, path in components.items()
    }
    external_hashes = {
        context: {
            summary["external_manifest_sha256"][context]
            for summary in summaries.values()
        }
        for context in ("dermoscopic", "clinical")
    }
    if any(len(values) != 1 for values in external_hashes.values()):
        raise RuntimeError("Components use different external manifests")

    predictions = _aligned_predictions(components)
    validation = predictions["validation"]
    reference_name = next(iter(components))
    validation_reference = validation[reference_name]
    validation_targets = validation_reference["targets"].astype(np.int64)
    validation_logits = {
        name: values["logits"].astype(np.float64)
        for name, values in validation.items()
    }
    normalization = {
        name: {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std()),
        }
        for name, values in validation_logits.items()
    }
    selected_name, selection_results = select_validation_ensemble(
        validation_logits,
        validation_targets,
        normalization,
        candidate_sets,
    )
    selected_members = candidate_sets[selected_name]
    selected_summaries = [summaries[name] for name in selected_members]
    weights = {name: 1.0 / len(selected_members) for name in selected_members}
    identity_payload = {
        "method": "equal_weight_validation_standardized_logit_average",
        "members": [
            {
                "name": summary["name"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "validation_logit_normalization": normalization[
                    summary["name"]
                ],
                "weight": weights[summary["name"]],
            }
            for summary in selected_summaries
        ],
    }
    component_set_sha256 = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    validation_scores = standardized_logit_average(
        validation_logits,
        normalization,
        selected_members,
    )
    calibration = fit_development_calibration(
        validation_scores,
        validation_targets,
        minimum_sensitivity=0.90,
        minimum_specificity=0.90,
        ece_bins=10,
    )
    calibration.update(
        {
            "ensemble_component_set_sha256": component_set_sha256,
            "ensemble_method": identity_payload["method"],
            "validation_selection_candidate": selected_name,
        }
    )

    output = create_new_directory(output_dir)
    write_json_exclusive(output / "calibration.json", calibration)
    output_artifacts: list[Path] = [output / "calibration.json"]
    context_outputs = {}
    for context, (_, target_key) in PREDICTION_FILES.items():
        values = predictions[context]
        reference = values[reference_name]
        context_logits = {
            name: item["logits"].astype(np.float64)
            for name, item in values.items()
        }
        scores = standardized_logit_average(
            context_logits,
            normalization,
            selected_members,
        )
        targets = reference[target_key].astype(np.int64)
        if context in {"validation", "test"}:
            groups = _patient_groups(
                components[reference_name],
                context,
                reference["image_names"],
                targets,
            )
            resampling_unit = "patient"
        else:
            groups = reference["lesion_ids"].astype(str)
            resampling_unit = "lesion"
        evaluation, arrays = _evaluate(
            scores,
            targets,
            groups,
            calibration,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed,
            resampling_unit=resampling_unit,
        )
        identity = _evaluation_identity(
            reference["image_names"],
            targets,
            groups,
        )
        common = {
            "schema_version": 1,
            "status": "ensemble_research_evaluation",
            "ensemble_component_set_sha256": component_set_sha256,
            "evaluation_set_sha256": identity,
            "selection_split": "validation",
            "calibration_sha256": sha256_file(output / "calibration.json"),
            "low_threshold": calibration["low_threshold"],
            "high_threshold": calibration["high_threshold"],
            "thresholds_validated": False,
            "research_only": True,
        }
        if context == "validation":
            report = {**common, "split": "validation", **evaluation}
            json_name = "metrics_validation.json"
            npz_name = "predictions_validation.npz"
        elif context == "test":
            report = {**common, "split": "test", **evaluation}
            json_name = "metrics_test.json"
            npz_name = "predictions_test.npz"
        else:
            broad_targets = reference[
                "broad_malignancy_targets"
            ].astype(np.int64)
            probabilities = arrays["calibrated_probabilities"]
            report = {
                **common,
                "external_manifest_sha256": next(
                    iter(
                        external_hashes[
                            "dermoscopic"
                            if context == "milk10k_dermoscopic"
                            else "clinical"
                        ]
                    )
                ),
                "external_records": int(len(targets)),
                "external_lesions": int(len(np.unique(groups))),
                "external_image_type": (
                    "dermoscopic"
                    if context == "milk10k_dermoscopic"
                    else "clinical: close-up"
                ),
                "melanoma_vs_all": {
                    "uncalibrated": evaluation["uncalibrated_metrics"],
                    "low_threshold": evaluation[
                        "calibrated_low_threshold_metrics"
                    ],
                    "high_threshold": evaluation[
                        "calibrated_high_threshold_metrics"
                    ],
                    "confidence_intervals": evaluation[
                        "confidence_intervals"
                    ],
                },
                "broad_malignancy_exploratory": {
                    "warning": (
                        "The ensemble targets melanoma attention; this "
                        "secondary analysis does not redefine its target"
                    ),
                    "low_threshold": compute_binary_metrics(
                        broad_targets,
                        probabilities,
                        threshold=float(calibration["low_threshold"]),
                    ),
                    "high_threshold": compute_binary_metrics(
                        broad_targets,
                        probabilities,
                        threshold=float(calibration["high_threshold"]),
                    ),
                },
            }
            suffix = (
                "dermoscopic"
                if context == "milk10k_dermoscopic"
                else "clinical"
            )
            json_name = f"metrics_external_milk10k_{suffix}.json"
            npz_name = f"metrics_external_milk10k_{suffix}.npz"
            arrays.update(
                {
                    "broad_malignancy_targets": broad_targets,
                    "lesion_ids": groups,
                    "diagnosis_classes": reference[
                        "diagnosis_classes"
                    ],
                }
            )
        arrays["image_names"] = reference["image_names"]
        write_json_exclusive(output / json_name, report)
        write_npz_exclusive(output / npz_name, **arrays)
        output_artifacts.extend([output / json_name, output / npz_name])
        context_outputs[context] = report

    ensemble = {
        "schema_version": 1,
        "status": "complete",
        "candidate_type": "ensemble",
        "method": identity_payload["method"],
        "ensemble_component_set_sha256": component_set_sha256,
        "validation_selection": {
            "policy": [
                "validation PR-AUC descending",
                "validation ROC-AUC descending",
                "model count ascending",
                "candidate name descending",
            ],
            "selected": selected_name,
            "candidates": selection_results,
            "external_metrics_observed_during_selection": False,
        },
        "components": selected_summaries,
        "weights": weights,
        "validation_logit_normalization": {
            name: normalization[name] for name in selected_members
        },
        "inference_model_count": len(selected_members),
        "estimated_relative_model_compute": float(len(selected_members)),
        "deployment_status": "research_only_distillation_or_multi_engine_pending",
        "cxx_tensorrt_runtime_status": "not_implemented_for_ensemble",
        "distillation_recommended": True,
        "artifact_sha256": {
            path.name: sha256_file(path) for path in output_artifacts
        },
        "test_pr_auc": context_outputs["test"][
            "calibrated_low_threshold_metrics"
        ]["pr_auc"],
        "milk10k_dermoscopic_pr_auc": context_outputs[
            "milk10k_dermoscopic"
        ]["melanoma_vs_all"]["low_threshold"]["pr_auc"],
        "thresholds_validated": False,
        "deployment_authorized": False,
        "research_only": True,
    }
    write_json_exclusive(output / "ensemble.json", ensemble)
    print(ensemble)
    return ensemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", action="append", required=True)
    parser.add_argument("--candidate-set", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    components = dict(_parse_named_path(value) for value in args.component)
    candidate_sets = dict(
        _parse_candidate_set(value) for value in args.candidate_set
    )
    if len(components) != len(args.component):
        raise ValueError("Component names must be unique")
    if len(candidate_sets) != len(args.candidate_set):
        raise ValueError("Candidate-set names must be unique")
    build_ensemble_candidate(
        components=components,
        candidate_sets=candidate_sets,
        output_dir=Path(args.output_dir),
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
