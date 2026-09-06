from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifacts import write_json_exclusive
from .common import sha256_file


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_candidate(name: str, run_dir: Path) -> dict[str, Any]:
    ensemble_path = run_dir / "ensemble.json"
    if ensemble_path.is_file():
        return summarize_ensemble_candidate(name, run_dir)
    run_path = run_dir / "run.json"
    calibration_path = run_dir / "calibration.json"
    test_path = run_dir / "metrics_test.json"
    external_path = run_dir / "metrics_external_milk10k_dermoscopic.json"
    clinical_path = run_dir / "metrics_external_milk10k_clinical.json"
    required = (
        run_path,
        calibration_path,
        test_path,
        external_path,
        clinical_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Candidate artifacts are missing: {missing}")
    run = _read_json(run_path)
    calibration = _read_json(calibration_path)
    test = _read_json(test_path)
    external = _read_json(external_path)
    clinical = _read_json(clinical_path)
    checkpoint_sha256 = str(run["checkpoint_sha256"])
    if calibration.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError(f"Calibration checkpoint mismatch: {name}")
    if test.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError(f"Test checkpoint mismatch: {name}")
    for label, report in (("external", external), ("clinical", clinical)):
        if report.get("run_checkpoint_sha256") != checkpoint_sha256:
            raise RuntimeError(f"{label.title()} checkpoint mismatch: {name}")
        if report.get("thresholds_validated") is not False:
            raise RuntimeError(f"{label.title()} thresholds must be unvalidated")
    test_metrics = test["calibrated_low_threshold_metrics"]
    external_metrics = external["melanoma_vs_all"]["low_threshold"]
    clinical_metrics = clinical["melanoma_vs_all"]["low_threshold"]
    candidate_type = str(run.get("candidate_type", "single_model"))
    metric_names = (
        "roc_auc",
        "pr_auc",
        "sensitivity",
        "specificity",
        "brier_score",
        "expected_calibration_error",
    )
    return {
        "name": name,
        "candidate_type": candidate_type,
        "run_dir": str(run_dir),
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": run["manifest_sha256"],
        "calibration_sha256": sha256_file(calibration_path),
        "test": {key: test_metrics[key] for key in metric_names},
        "milk10k_dermoscopic": {
            key: external_metrics[key] for key in metric_names
        },
        "milk10k_clinical_stress": {
            key: clinical_metrics[key]
            for key in ("roc_auc", "pr_auc", "sensitivity", "specificity")
        },
        "thresholds_validated": False,
        "research_only": True,
    }


def summarize_ensemble_candidate(name: str, run_dir: Path) -> dict[str, Any]:
    ensemble_path = run_dir / "ensemble.json"
    calibration_path = run_dir / "calibration.json"
    test_path = run_dir / "metrics_test.json"
    external_path = run_dir / "metrics_external_milk10k_dermoscopic.json"
    clinical_path = run_dir / "metrics_external_milk10k_clinical.json"
    required = (
        ensemble_path,
        calibration_path,
        test_path,
        external_path,
        clinical_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Ensemble candidate artifacts are missing: {missing}")
    ensemble = _read_json(ensemble_path)
    if ensemble.get("status") != "complete":
        raise RuntimeError(f"Ensemble candidate is incomplete: {name}")
    if ensemble.get("deployment_authorized") is not False:
        raise RuntimeError("Research ensemble cannot authorize deployment")
    for filename, expected in ensemble.get("artifact_sha256", {}).items():
        if sha256_file(run_dir / filename) != expected:
            raise RuntimeError(
                f"Ensemble artifact hash mismatch: {name}/{filename}"
            )
    component_set_sha256 = ensemble["ensemble_component_set_sha256"]
    calibration = _read_json(calibration_path)
    test = _read_json(test_path)
    external = _read_json(external_path)
    clinical = _read_json(clinical_path)
    for label, report in (
        ("calibration", calibration),
        ("test", test),
        ("external", external),
        ("clinical", clinical),
    ):
        if report.get("ensemble_component_set_sha256") != component_set_sha256:
            raise RuntimeError(f"Ensemble identity mismatch: {name}/{label}")
        if report.get("thresholds_validated") is not False:
            raise RuntimeError(f"Ensemble thresholds must be unvalidated: {name}")
    test_metrics = test["calibrated_low_threshold_metrics"]
    external_metrics = external["melanoma_vs_all"]["low_threshold"]
    clinical_metrics = clinical["melanoma_vs_all"]["low_threshold"]
    metric_names = (
        "roc_auc",
        "pr_auc",
        "sensitivity",
        "specificity",
        "brier_score",
        "expected_calibration_error",
    )
    return {
        "name": name,
        "candidate_type": "ensemble",
        "run_dir": str(run_dir),
        "ensemble_sha256": sha256_file(ensemble_path),
        "ensemble_component_set_sha256": component_set_sha256,
        "component_checkpoint_sha256": [
            component["checkpoint_sha256"]
            for component in ensemble["components"]
        ],
        "manifest_sha256": sorted(
            {component["manifest_sha256"] for component in ensemble["components"]}
        ),
        "calibration_sha256": sha256_file(calibration_path),
        "inference_model_count": ensemble["inference_model_count"],
        "test": {key: test_metrics[key] for key in metric_names},
        "milk10k_dermoscopic": {
            key: external_metrics[key] for key in metric_names
        },
        "milk10k_clinical_stress": {
            key: clinical_metrics[key]
            for key in ("roc_auc", "pr_auc", "sensitivity", "specificity")
        },
        "deployment_authorized": False,
        "thresholds_validated": False,
        "research_only": True,
    }


def selection_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    external = candidate["milk10k_dermoscopic"]
    test = candidate["test"]
    return (
        float(external["pr_auc"]),
        float(external["roc_auc"]),
        float(test["pr_auc"]),
        float(test["roc_auc"]),
        -float(external["brier_score"]),
    )


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if len(candidates) < 2:
        raise ValueError("At least two candidates are required")
    ordered = sorted(candidates, key=selection_key, reverse=True)
    return {
        "schema_version": 1,
        "status": "research_candidate_selection",
        "selection_policy": {
            "order": [
                "MILK10k dermoscopic PR-AUC descending",
                "MILK10k dermoscopic ROC-AUC descending",
                "ISIC test PR-AUC descending",
                "ISIC test ROC-AUC descending",
                "MILK10k dermoscopic Brier score ascending",
            ],
            "warning": (
                "ISIC test and MILK10k are observed development benchmarks; "
                "selection does not constitute independent or microscope-domain "
                "validation"
            ),
        },
        "selected_research_candidate": ordered[0]["name"],
        "ranking": [candidate["name"] for candidate in ordered],
        "candidates": ordered,
        "deployment_authorized": False,
        "thresholds_validated": False,
        "research_only": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate in NAME=RUN_DIR form; provide at least twice.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = []
    for value in args.candidate:
        name, separator, run_dir = value.partition("=")
        if not separator or not name.strip() or not run_dir.strip():
            raise ValueError("--candidate must use NAME=RUN_DIR")
        candidates.append(summarize_candidate(name.strip(), Path(run_dir)))
    report = select_candidate(candidates)
    write_json_exclusive(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
