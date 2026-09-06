from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
import onnxruntime as ort

from .artifacts import (
    create_new_directory,
    require_absent,
    write_json_exclusive,
    write_npz_exclusive,
)
from .common import sha256_file


CLASS_NAMES = ("benign", "atypical", "melanoma")
CLASS_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
DIAGNOSIS_FIELDS = (
    "diagnosis_1",
    "diagnosis_2",
    "diagnosis_3",
    "diagnosis_4",
    "diagnosis_5",
)
ATYPICAL_TOKENS = ("atyp", "dysplast", "clark")


class ThreeZoneThresholdError(RuntimeError):
    pass


@dataclass(frozen=True)
class CohortRecord:
    image_name: str
    patient_id: str
    lesion_group: str
    class_name: str
    image_sha256: str
    split: str


def reference_class(row: Mapping[str, str]) -> str | None:
    """Map histopathology-confirmed melanocytic lesions to three references."""
    if row.get("diagnosis_confirm_type", "").strip().lower() != "histopathology":
        return None
    hierarchy = " | ".join(row.get(field, "") for field in DIAGNOSIS_FIELDS)
    normalized = hierarchy.lower()
    if "melanoma" in normalized:
        return "melanoma"
    if any(token in normalized for token in ATYPICAL_TOKENS):
        return "atypical"
    if (
        row.get("diagnosis_1", "").strip().lower() == "benign"
        and ("nevus" in normalized or "melanocytic" in normalized)
    ):
        return "benign"
    return None


def patient_partition(
    patient_id: str,
    *,
    seed: int,
    calibration_fraction: float,
) -> str:
    if not patient_id.strip():
        raise ThreeZoneThresholdError("Every cohort record needs a patient ID")
    if not 0.0 < calibration_fraction < 1.0:
        raise ThreeZoneThresholdError("Calibration fraction must be in (0, 1)")
    digest = hashlib.sha256(f"{seed}:{patient_id}".encode("utf-8")).digest()
    unit_value = int.from_bytes(digest[:8], "big") / float(2**64)
    return "calibration" if unit_value < calibration_fraction else "test"


def patient_fold(patient_id: str, *, seed: int, folds: int) -> int:
    if not patient_id.strip():
        raise ThreeZoneThresholdError("Every cohort record needs a patient ID")
    if folds < 2:
        raise ThreeZoneThresholdError("Cross-validation needs at least two folds")
    digest = hashlib.sha256(f"{seed}:cv:{patient_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def build_cohort(
    rows: Iterable[Mapping[str, str]],
    image_receipts: Mapping[str, str],
    *,
    excluded_image_names: set[str],
    seed: int,
    calibration_fraction: float,
) -> list[CohortRecord]:
    grouped: dict[tuple[str, str], list[tuple[str, Mapping[str, str]]]] = {}
    for row in rows:
        class_name = reference_class(row)
        if class_name is None:
            continue
        image_name = row.get("image_name", "").strip()
        patient_id = row.get("patient_id", "").strip()
        lesion_id = row.get("lesion_id", "").strip()
        if not image_name or not patient_id:
            raise ThreeZoneThresholdError(
                "Eligible cohort rows need image_name and patient_id"
            )
        if image_name in excluded_image_names:
            continue
        lesion_group = lesion_id or image_name
        grouped.setdefault((patient_id, lesion_group), []).append(
            (class_name, row)
        )

    cohort: list[CohortRecord] = []
    for (patient_id, lesion_group), candidates in sorted(grouped.items()):
        classes = {class_name for class_name, _ in candidates}
        if len(classes) != 1:
            raise ThreeZoneThresholdError(
                f"Conflicting classes for lesion group {lesion_group}: {classes}"
            )
        class_name, selected = min(
            candidates,
            key=lambda item: item[1]["image_name"],
        )
        image_name = selected["image_name"].strip()
        image_sha256 = image_receipts.get(image_name, "")
        if len(image_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in image_sha256.lower()
        ):
            raise ThreeZoneThresholdError(
                f"Missing image SHA-256 receipt for {image_name}"
            )
        cohort.append(
            CohortRecord(
                image_name=image_name,
                patient_id=patient_id,
                lesion_group=lesion_group,
                class_name=class_name,
                image_sha256=image_sha256,
                split=patient_partition(
                    patient_id,
                    seed=seed,
                    calibration_fraction=calibration_fraction,
                ),
            )
        )

    if not cohort:
        raise ThreeZoneThresholdError("Three-class cohort is empty")
    for split in ("calibration", "test"):
        split_classes = {
            record.class_name for record in cohort if record.split == split
        }
        if split_classes != set(CLASS_NAMES):
            raise ThreeZoneThresholdError(
                f"Split {split} does not contain all reference classes"
            )
    calibration_patients = {
        record.patient_id for record in cohort if record.split == "calibration"
    }
    test_patients = {
        record.patient_id for record in cohort if record.split == "test"
    }
    if calibration_patients & test_patients:
        raise ThreeZoneThresholdError("Patient leakage across threshold splits")
    return cohort


def three_class_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
    *,
    low_threshold: float,
    high_threshold: float,
) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if targets.ndim != 1 or scores.ndim != 1 or len(targets) != len(scores):
        raise ThreeZoneThresholdError("Targets and scores must be equal vectors")
    if len(targets) == 0 or set(np.unique(targets)) != {0, 1, 2}:
        raise ThreeZoneThresholdError("All three target classes are required")
    if not np.isfinite(scores).all():
        raise ThreeZoneThresholdError("Scores must be finite")
    if not 0.0 <= low_threshold < high_threshold <= 1.0:
        raise ThreeZoneThresholdError("Thresholds must be ordered in [0, 1]")

    predictions = np.where(
        scores < low_threshold,
        0,
        np.where(scores < high_threshold, 1, 2),
    )
    result = classification_metrics(targets, predictions)
    result["low_threshold"] = float(low_threshold)
    result["high_threshold"] = float(high_threshold)
    return result


def classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if (
        targets.ndim != 1
        or predictions.ndim != 1
        or len(targets) != len(predictions)
    ):
        raise ThreeZoneThresholdError(
            "Targets and predictions must be equal vectors"
        )
    if len(targets) == 0 or set(np.unique(targets)) != {0, 1, 2}:
        raise ThreeZoneThresholdError("All three target classes are required")
    if np.any((predictions < 0) | (predictions > 2)):
        raise ThreeZoneThresholdError("Predictions must be class indexes 0, 1, or 2")

    confusion = np.zeros((3, 3), dtype=np.int64)
    for target, prediction in zip(targets, predictions, strict=True):
        confusion[int(target), int(prediction)] += 1
    recalls = np.diag(confusion) / confusion.sum(axis=1)
    predicted_counts = confusion.sum(axis=0)
    precisions = np.divide(
        np.diag(confusion),
        predicted_counts,
        out=np.zeros(3, dtype=np.float64),
        where=predicted_counts != 0,
    )
    return {
        "records": int(len(targets)),
        "confusion_matrix": confusion.tolist(),
        "class_support": {
            name: int(confusion[index].sum())
            for index, name in enumerate(CLASS_NAMES)
        },
        "class_recall": {
            name: float(recalls[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "class_error": {
            name: float(1.0 - recalls[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "class_precision": {
            name: float(precisions[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "predicted_distribution": {
            name: int(predicted_counts[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "accuracy": float(np.trace(confusion) / confusion.sum()),
        "macro_recall": float(recalls.mean()),
        "worst_class_recall": float(recalls.min()),
        "recall_range": float(recalls.max() - recalls.min()),
    }


def optimize_three_zone_thresholds(
    targets: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    """Maximize worst-class recall, then macro recall, without class weights."""
    targets = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if targets.ndim != 1 or scores.ndim != 1 or len(targets) != len(scores):
        raise ThreeZoneThresholdError("Targets and scores must be equal vectors")
    if len(targets) < 3 or set(np.unique(targets)) != {0, 1, 2}:
        raise ThreeZoneThresholdError("Threshold search needs all three classes")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ThreeZoneThresholdError("Threshold search scores must be in [0, 1]")

    order = np.argsort(scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_targets = targets[order]
    boundaries = np.flatnonzero(ordered_scores[:-1] < ordered_scores[1:]) + 1
    if len(boundaries) < 2:
        raise ThreeZoneThresholdError("Not enough distinct scores for two thresholds")

    prefix = np.zeros((len(targets) + 1, 3), dtype=np.int64)
    for index, target in enumerate(ordered_targets, start=1):
        prefix[index] = prefix[index - 1]
        prefix[index, int(target)] += 1
    totals = prefix[-1]

    best_key: tuple[float, ...] | None = None
    best_boundaries: tuple[int, int] | None = None
    candidates = 0
    for low_position_index, low_position in enumerate(boundaries[:-1]):
        low_correct = prefix[low_position, 0]
        for high_position in boundaries[low_position_index + 1 :]:
            candidates += 1
            recalls = np.asarray(
                [
                    low_correct / totals[0],
                    (
                        prefix[high_position, 1]
                        - prefix[low_position, 1]
                    )
                    / totals[1],
                    (totals[2] - prefix[high_position, 2]) / totals[2],
                ],
                dtype=np.float64,
            )
            key = (
                float(recalls.min()),
                float(recalls.mean()),
                float(-np.ptp(recalls)),
                float(recalls[0] + recalls[2]),
                float(-low_position),
                float(-high_position),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_boundaries = (int(low_position), int(high_position))

    if best_boundaries is None:
        raise ThreeZoneThresholdError("Threshold optimization found no candidate")
    low_position, high_position = best_boundaries
    low_threshold = float(
        (ordered_scores[low_position - 1] + ordered_scores[low_position]) / 2.0
    )
    high_threshold = float(
        (
            ordered_scores[high_position - 1]
            + ordered_scores[high_position]
        )
        / 2.0
    )
    result = three_class_metrics(
        targets,
        scores,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    result["selection_objective"] = (
        "lexicographic_maximin_class_recall_then_macro_recall"
    )
    result["candidate_pairs"] = candidates
    return result


def cross_validate_three_zone_thresholds(
    targets: np.ndarray,
    scores: np.ndarray,
    patient_ids: np.ndarray,
    *,
    seed: int,
    folds: int,
) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    patient_ids = np.asarray(patient_ids).astype(str)
    if not (len(targets) == len(scores) == len(patient_ids)):
        raise ThreeZoneThresholdError(
            "Targets, scores, and patient IDs must have equal length"
        )
    fold_indexes = np.asarray(
        [
            patient_fold(patient_id, seed=seed, folds=folds)
            for patient_id in patient_ids
        ],
        dtype=np.int64,
    )
    out_of_fold_predictions = np.full(len(targets), -1, dtype=np.int64)
    fold_reports: list[dict[str, Any]] = []
    for fold in range(folds):
        held_out = fold_indexes == fold
        selected = optimize_three_zone_thresholds(
            targets[~held_out],
            scores[~held_out],
        )
        evaluated = three_class_metrics(
            targets[held_out],
            scores[held_out],
            low_threshold=float(selected["low_threshold"]),
            high_threshold=float(selected["high_threshold"]),
        )
        out_of_fold_predictions[held_out] = np.where(
            scores[held_out] < selected["low_threshold"],
            0,
            np.where(scores[held_out] < selected["high_threshold"], 1, 2),
        )
        fold_reports.append(
            {
                "fold": fold,
                "selected_low_threshold": selected["low_threshold"],
                "selected_high_threshold": selected["high_threshold"],
                "held_out": evaluated,
            }
        )
    pooled = classification_metrics(targets, out_of_fold_predictions)
    return {
        "folds": folds,
        "split_unit": "patient_id",
        "selection": "all_non_held_out_folds",
        "fold_reports": fold_reports,
        "pooled_out_of_fold": pooled,
        "mean_fold_worst_class_recall": float(
            np.mean(
                [
                    report["held_out"]["worst_class_recall"]
                    for report in fold_reports
                ]
            )
        ),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def _image_receipts(path: Path) -> dict[str, str]:
    receipts: dict[str, str] = {}
    for row in _read_csv(path):
        name = row.get("image_name", "").strip()
        digest = row.get("sha256", "").strip().lower()
        if name in receipts and receipts[name] != digest:
            raise ThreeZoneThresholdError(f"Conflicting receipt for {name}")
        receipts[name] = digest
    return receipts


def _excluded_images(path: Path | None) -> set[str]:
    if path is None:
        return set()
    rows = _read_csv(path)
    return {
        (row.get("isic_id") or row.get("image_name") or "").strip()
        for row in rows
        if (row.get("isic_id") or row.get("image_name") or "").strip()
    }


def _calibrated_score(raw_output: float, manifest: Mapping[str, Any]) -> float:
    method = manifest.get("calibration_method")
    if method == "affine_logistic_scaling":
        calibrated_logit = (
            float(manifest["logit_scale"]) * raw_output
            + float(manifest["logit_bias"])
        )
    elif method == "temperature_scaling":
        calibrated_logit = raw_output / float(manifest["temperature"])
    else:
        raise ThreeZoneThresholdError(f"Unsupported calibration: {method}")
    return float(1.0 / (1.0 + math.exp(-np.clip(calibrated_logit, -50, 50))))


def score_cohort(
    cohort: list[CohortRecord],
    *,
    model_dir: Path,
    images_dir: Path,
) -> np.ndarray:
    manifest_path = model_dir / "manifest.json"
    model_path = model_dir / "model.onnx"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(model_path) != manifest.get("model_sha256"):
        raise ThreeZoneThresholdError("ONNX hash does not match manifest")
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    size = int(manifest["image_size"])
    mean = np.asarray(manifest["mean"], dtype=np.float32)
    std = np.asarray(manifest["std"], dtype=np.float32)
    scores = np.empty(len(cohort), dtype=np.float64)
    for index, record in enumerate(cohort):
        if index == 0 or (index + 1) % 250 == 0:
            print(f"Scoring image {index + 1:,}/{len(cohort):,}", flush=True)
        image_path = images_dir / f"{record.image_name}.jpg"
        if not image_path.is_file() or sha256_file(image_path) != record.image_sha256:
            raise ThreeZoneThresholdError(
                f"Image integrity check failed: {record.image_name}"
            )
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ThreeZoneThresholdError(
                f"Unable to decode image: {record.image_name}"
            )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - mean) / std
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        raw = session.run(
            [str(manifest["output_name"])],
            {str(manifest["input_name"]): tensor},
        )[0]
        scores[index] = _calibrated_score(
            float(np.asarray(raw).reshape(-1)[0]),
            manifest,
        )
    return scores


def _class_counts(records: list[CohortRecord]) -> dict[str, dict[str, int]]:
    return {
        split: {
            class_name: sum(
                record.split == split and record.class_name == class_name
                for record in records
            )
            for class_name in CLASS_NAMES
        }
        for split in ("calibration", "test")
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select fair low/intermediate/high thresholds on a patient-safe "
            "three-class dermoscopy cohort."
        )
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--metadata-csv", required=True, type=Path)
    parser.add_argument("--image-receipts-csv", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--exclude-manifest", type=Path)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--target-class-recall", type=float, default=0.9)
    parser.add_argument("--cross-validation-folds", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.target_class_recall <= 1.0:
        raise ValueError("Target class recall must be in (0, 1]")
    if args.cross_validation_folds < 2:
        raise ValueError("Cross-validation needs at least two folds")
    require_absent([args.output_dir])
    cohort = build_cohort(
        _read_csv(args.metadata_csv),
        _image_receipts(args.image_receipts_csv),
        excluded_image_names=_excluded_images(args.exclude_manifest),
        seed=args.seed,
        calibration_fraction=args.calibration_fraction,
    )
    scores = score_cohort(
        cohort,
        model_dir=args.model_dir,
        images_dir=args.images_dir,
    )
    targets = np.asarray(
        [CLASS_INDEX[record.class_name] for record in cohort],
        dtype=np.int64,
    )
    split_values = np.asarray([record.split for record in cohort])
    calibration_members = split_values == "calibration"
    test_members = split_values == "test"
    selected = optimize_three_zone_thresholds(
        targets[calibration_members],
        scores[calibration_members],
    )
    test_metrics = three_class_metrics(
        targets[test_members],
        scores[test_members],
        low_threshold=float(selected["low_threshold"]),
        high_threshold=float(selected["high_threshold"]),
    )
    manifest = json.loads(
        (args.model_dir / "manifest.json").read_text(encoding="utf-8")
    )
    current_test = three_class_metrics(
        targets[test_members],
        scores[test_members],
        low_threshold=float(manifest["low_threshold"]),
        high_threshold=float(manifest["high_threshold"]),
    )
    cross_validation = cross_validate_three_zone_thresholds(
        targets,
        scores,
        np.asarray([record.patient_id for record in cohort]),
        seed=args.seed,
        folds=args.cross_validation_folds,
    )
    target_achieved = all(
        value >= args.target_class_recall
        for value in test_metrics["class_recall"].values()
    )
    report = {
        "schema_version": 1,
        "status": "research_only",
        "selection_split": "calibration",
        "evaluation_split": "test",
        "split_unit": "patient_id",
        "one_image_per_lesion": True,
        "reference_standard": "histopathology",
        "reference_classes": list(CLASS_NAMES),
        "model_version": manifest["version"],
        "model_release": manifest.get("release_version", manifest["version"]),
        "model_sha256": manifest["model_sha256"],
        "model_manifest_sha256": sha256_file(args.model_dir / "manifest.json"),
        "metadata_sha256": sha256_file(args.metadata_csv),
        "image_receipts_sha256": sha256_file(args.image_receipts_csv),
        "excluded_manifest_sha256": (
            sha256_file(args.exclude_manifest)
            if args.exclude_manifest is not None
            else None
        ),
        "seed": args.seed,
        "calibration_fraction": args.calibration_fraction,
        "cohort_records": len(cohort),
        "cohort_counts": _class_counts(cohort),
        "selected_on_calibration": selected,
        "locked_test": test_metrics,
        "current_thresholds_locked_test": current_test,
        "patient_grouped_cross_validation": cross_validation,
        "target_class_recall": args.target_class_recall,
        "target_achieved_on_locked_test": target_achieved,
        "deployment_authorized": False,
        "limitation": (
            "Dermoscopy reference data do not validate USB-microscope "
            "performance; intermediate is a histopathology-derived atypical "
            "reference class, not a clinical follow-up instruction."
        ),
    }
    output_dir = create_new_directory(args.output_dir)
    write_npz_exclusive(
        output_dir / "scores.npz",
        image_names=np.asarray([record.image_name for record in cohort]),
        image_sha256s=np.asarray([record.image_sha256 for record in cohort]),
        patient_ids=np.asarray([record.patient_id for record in cohort]),
        lesion_groups=np.asarray([record.lesion_group for record in cohort]),
        class_names=np.asarray([record.class_name for record in cohort]),
        splits=split_values,
        targets=targets,
        scores=scores,
    )
    write_json_exclusive(output_dir / "report.json", report)
    print(json.dumps(report["locked_test"], indent=2, sort_keys=True))
    print(f"Target achieved: {target_achieved}")


if __name__ == "__main__":
    main()
