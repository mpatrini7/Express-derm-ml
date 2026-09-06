from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .artifacts import write_json_exclusive
from .metrics import compute_binary_metrics, grouped_bootstrap_intervals


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_PROTOCOLS = {"within_source_test", "leave_one_source_out"}
HIGHER_IS_BETTER = {
    "roc_auc",
    "pr_auc",
    "pr_auc_lift",
    "sensitivity",
    "specificity",
    "precision",
    "negative_predictive_value",
    "f1_score",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    normalized = str(value).strip().lower()
    if not SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return normalized


def _load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Generalization config must use schema_version 1")
    return config


def _aligned_groups(
    dataset: dict[str, Any],
    *,
    arrays: Any,
    record_ids: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    group_key = dataset.get("group_id_key")
    if group_key is not None:
        if str(group_key) not in arrays.files:
            raise ValueError(
                f"Dataset {dataset['id']}: missing group key {group_key}"
            )
        groups = np.asarray(arrays[str(group_key)], dtype=str)
        evidence = {
            "method": "prediction_array",
            "group_id_key": str(group_key),
        }
    else:
        lookup = dataset.get("group_lookup")
        if not isinstance(lookup, dict):
            raise ValueError(
                f"Dataset {dataset['id']}: group_id_key or group_lookup is required"
            )
        manifest_path = Path(str(lookup["manifest_path"]))
        expected_hash = _require_sha256(
            lookup["manifest_file_sha256"],
            f"datasets[{dataset['id']}].group_lookup.manifest_file_sha256",
        )
        if sha256_file(manifest_path) != expected_hash:
            raise ValueError(
                f"Dataset {dataset['id']}: group lookup manifest hash mismatch"
            )
        id_column = str(lookup.get("record_id_column", "image_name"))
        group_column = str(lookup.get("group_id_column", "group_id"))
        target_column = str(lookup.get("target_column", "target"))
        frame = pd.read_csv(
            manifest_path,
            dtype={id_column: str, group_column: str},
        )
        missing = {id_column, group_column, target_column} - set(frame.columns)
        if missing:
            raise ValueError(
                f"Dataset {dataset['id']}: group lookup is missing {sorted(missing)}"
            )
        if frame[id_column].duplicated().any():
            raise ValueError(
                f"Dataset {dataset['id']}: group lookup record IDs are not unique"
            )
        indexed = frame.set_index(id_column)
        missing_ids = sorted(set(record_ids) - set(indexed.index.astype(str)))
        if missing_ids:
            raise ValueError(
                f"Dataset {dataset['id']}: prediction IDs are absent from lookup"
            )
        aligned = indexed.reindex(record_ids)
        lookup_targets = pd.to_numeric(
            aligned[target_column], errors="coerce"
        ).to_numpy()
        if not np.array_equal(lookup_targets, targets):
            raise ValueError(
                f"Dataset {dataset['id']}: prediction targets do not match lookup"
            )
        groups = aligned[group_column].astype(str).to_numpy()
        evidence = {
            "method": "manifest_lookup",
            "manifest_path": str(manifest_path),
            "manifest_file_sha256": expected_hash,
            "record_id_column": id_column,
            "group_id_column": group_column,
            "target_column": target_column,
        }

    if groups.shape != record_ids.shape or any(not value.strip() for value in groups):
        raise ValueError(f"Dataset {dataset['id']}: invalid or blank group IDs")
    return groups, evidence


def _evaluate_dataset(
    dataset: dict[str, Any],
    *,
    candidate: dict[str, Any],
    training_sources: set[str],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    dataset_id = str(dataset.get("id", "")).strip()
    source_id = str(dataset.get("source_id", "")).strip()
    if not dataset_id or not source_id:
        raise ValueError("Every evaluation dataset needs id and source_id")
    protocol = str(dataset.get("evaluation_protocol", ""))
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(
            f"Dataset {dataset_id}: unsupported evaluation_protocol {protocol!r}"
        )
    if dataset.get("training_authorized") is not False:
        raise ValueError(
            f"Dataset {dataset_id}: evaluation data must forbid training"
        )
    if protocol == "leave_one_source_out" and source_id in training_sources:
        raise ValueError(
            f"Dataset {dataset_id}: held-out source appears in training_source_ids"
        )
    if protocol == "within_source_test" and source_id not in training_sources:
        raise ValueError(
            f"Dataset {dataset_id}: within-source test source is absent from training"
        )

    predictions_path = Path(str(dataset["predictions_path"]))
    expected_hash = _require_sha256(
        dataset["predictions_sha256"],
        f"datasets[{dataset_id}].predictions_sha256",
    )
    actual_hash = sha256_file(predictions_path)
    if actual_hash != expected_hash:
        raise ValueError(f"Dataset {dataset_id}: predictions hash mismatch")
    probability_key = str(
        dataset.get("probability_key", "calibrated_probabilities")
    )
    target_key = str(dataset.get("target_key", "targets"))
    record_id_key = str(dataset.get("record_id_key", "image_names"))
    with np.load(predictions_path, allow_pickle=False) as arrays:
        missing_keys = {
            probability_key,
            target_key,
            record_id_key,
        } - set(arrays.files)
        if missing_keys:
            raise ValueError(
                f"Dataset {dataset_id}: predictions missing {sorted(missing_keys)}"
            )
        probabilities = np.asarray(arrays[probability_key], dtype=np.float64)
        targets = np.asarray(arrays[target_key], dtype=np.int64)
        record_ids = np.asarray(arrays[record_id_key], dtype=str)
        groups, group_evidence = _aligned_groups(
            dataset,
            arrays=arrays,
            record_ids=record_ids,
            targets=targets,
        )

    if probabilities.ndim != 1 or targets.ndim != 1 or record_ids.ndim != 1:
        raise ValueError(f"Dataset {dataset_id}: prediction arrays must be flat")
    if not (
        len(probabilities) == len(targets) == len(record_ids) == len(groups)
    ) or len(targets) == 0:
        raise ValueError(f"Dataset {dataset_id}: prediction arrays are unaligned")
    if len(np.unique(record_ids)) != len(record_ids):
        raise ValueError(f"Dataset {dataset_id}: record IDs must be unique")
    if any(not value.strip() for value in record_ids):
        raise ValueError(f"Dataset {dataset_id}: record IDs cannot be blank")
    if set(np.unique(targets)) != {0, 1}:
        raise ValueError(f"Dataset {dataset_id}: both binary targets are required")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0.0) | (probabilities > 1.0)
    ).any():
        raise ValueError(f"Dataset {dataset_id}: probabilities must be in [0, 1]")

    threshold = float(candidate["frozen_high_threshold"])
    metrics = compute_binary_metrics(
        targets,
        probabilities,
        threshold=threshold,
        ece_bins=int(bootstrap.get("ece_bins", 10)),
    )
    metrics["pr_auc_lift"] = float(metrics["pr_auc"] / metrics["prevalence"])
    metrics["false_positive_rate"] = float(1.0 - metrics["specificity"])
    metrics["false_negative_rate"] = float(1.0 - metrics["sensitivity"])
    intervals = grouped_bootstrap_intervals(
        targets,
        probabilities,
        groups,
        threshold=threshold,
        samples=int(bootstrap.get("samples", 500)),
        confidence_level=float(bootstrap.get("confidence_level", 0.95)),
        seed=int(bootstrap.get("seed", 2026)),
    )
    intervals["resampling_unit"] = "group_id"
    return {
        "id": dataset_id,
        "source_id": source_id,
        "modality": str(dataset.get("modality", "unspecified")),
        "evaluation_protocol": protocol,
        "training_authorized": False,
        "predictions_path": str(predictions_path),
        "predictions_sha256": actual_hash,
        "records": int(len(targets)),
        "groups": int(len(np.unique(groups))),
        "positive_records": int(targets.sum()),
        "group_evidence": group_evidence,
        "metrics_at_frozen_high_threshold": metrics,
        "confidence_intervals": intervals,
    }


def _gate_report(
    evaluations: list[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[dict[str, Any]] = []
    minimum_datasets = int(gate.get("minimum_datasets", 2))
    minimum_loso = int(gate.get("minimum_leave_one_source_out_datasets", 1))
    loso_count = sum(
        item["evaluation_protocol"] == "leave_one_source_out"
        for item in evaluations
    )
    if len(evaluations) < minimum_datasets:
        blockers.append(
            {
                "criterion": "minimum_datasets",
                "required": minimum_datasets,
                "observed": len(evaluations),
            }
        )
    if loso_count < minimum_loso:
        blockers.append(
            {
                "criterion": "minimum_leave_one_source_out_datasets",
                "required": minimum_loso,
                "observed": loso_count,
            }
        )

    minimums = gate.get("minimum_worst_source", {})
    maximum_gaps = gate.get("maximum_source_gap", {})
    if not isinstance(minimums, dict) or not isinstance(maximum_gaps, dict):
        raise ValueError("Research gate metric criteria must be mappings")
    values_by_metric: dict[str, list[float]] = {}
    for metric in sorted(set(minimums) | set(maximum_gaps)):
        if metric not in HIGHER_IS_BETTER:
            raise ValueError(f"Unsupported research gate metric: {metric}")
        values_by_metric[metric] = [
            float(item["metrics_at_frozen_high_threshold"][metric])
            for item in evaluations
        ]
    for metric, required_value in minimums.items():
        observed = min(values_by_metric[metric])
        required = float(required_value)
        if observed < required:
            blockers.append(
                {
                    "criterion": f"minimum_worst_source.{metric}",
                    "required": required,
                    "observed": observed,
                }
            )
    for metric, allowed_value in maximum_gaps.items():
        values = values_by_metric[metric]
        observed = max(values) - min(values)
        allowed = float(allowed_value)
        if observed > allowed:
            blockers.append(
                {
                    "criterion": f"maximum_source_gap.{metric}",
                    "required": allowed,
                    "observed": observed,
                }
            )

    summary_metrics = sorted(
        {
            "roc_auc",
            "pr_auc",
            "pr_auc_lift",
            "sensitivity",
            "specificity",
            "precision",
            *minimums.keys(),
            *maximum_gaps.keys(),
        }
    )
    cross_source = {}
    for metric in summary_metrics:
        values = [
            float(item["metrics_at_frozen_high_threshold"][metric])
            for item in evaluations
        ]
        cross_source[metric] = {
            "macro_mean": float(np.mean(values)),
            "worst": float(min(values)),
            "best": float(max(values)),
            "gap": float(max(values) - min(values)),
        }
    return {
        "status": "pass" if not blockers else "fail",
        "research_stage_only": True,
        "deployment_authorized": False,
        "leave_one_source_out_datasets": int(loso_count),
        "cross_source_summary": cross_source,
        "blockers": blockers,
    }


def evaluate_generalization_config(
    config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = _load_config(config_path)
    candidate = config.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("Generalization config is missing candidate")
    model_version = str(candidate.get("model_version", "")).strip()
    if not model_version:
        raise ValueError("candidate.model_version is required")
    checkpoint_sha256 = _require_sha256(
        candidate.get("checkpoint_sha256"), "candidate.checkpoint_sha256"
    )
    calibration_sha256 = _require_sha256(
        candidate.get("calibration_sha256"), "candidate.calibration_sha256"
    )
    high_threshold = float(candidate.get("frozen_high_threshold"))
    if not 0.0 < high_threshold < 1.0:
        raise ValueError("candidate.frozen_high_threshold must be in (0, 1)")
    training_sources = {
        str(value).strip() for value in candidate.get("training_source_ids", [])
    }
    if not training_sources or "" in training_sources:
        raise ValueError("candidate.training_source_ids cannot be empty or blank")
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Generalization config requires evaluation datasets")
    dataset_ids = [str(item.get("id", "")) for item in datasets]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("Evaluation dataset IDs must be unique")
    bootstrap = config.get("bootstrap", {})
    if not isinstance(bootstrap, dict):
        raise ValueError("bootstrap must be a mapping")
    evaluations = [
        _evaluate_dataset(
            item,
            candidate=candidate,
            training_sources=training_sources,
            bootstrap=bootstrap,
        )
        for item in datasets
    ]
    gate = config.get("research_gate", {})
    if not isinstance(gate, dict):
        raise ValueError("research_gate must be a mapping")
    report = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "multi_source_generalization_gate",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "candidate": {
            "model_version": model_version,
            "checkpoint_sha256": checkpoint_sha256,
            "calibration_sha256": calibration_sha256,
            "frozen_high_threshold": high_threshold,
            "training_source_ids": sorted(training_sources),
            "validation_status": "research_only",
        },
        "evaluations": evaluations,
        "gate": _gate_report(evaluations, gate),
    }
    write_json_exclusive(output_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one frozen candidate across within-source and "
            "leave-one-source-out datasets without changing thresholds."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--fail-on-gate",
        action="store_true",
        help="Return exit status 2 when the configured research gate fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_generalization_config(args.config, args.output)
    print(
        {
            "output": args.output,
            "gate": report["gate"]["status"],
            "blockers": report["gate"]["blockers"],
        }
    )
    if args.fail_on_gate and report["gate"]["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
