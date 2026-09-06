from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from .artifacts import create_new_directory, write_json_exclusive, write_npz_exclusive
from .build_ensemble_candidate import standardized_logit_average
from .calibration import apply_calibration, fit_development_calibration
from .common import sha256_file
from .dataset import DEPLOYMENT_PREPROCESSING_VERSION, LesionDataset
from .device import resolve_training_device, uses_cuda_transfer_optimizations
from .generate_teacher_targets import (
    infer_teacher_logits,
    load_teacher_models,
    validate_teacher_ensemble,
)
from .manifest import read_manifest
from .metrics import compute_binary_metrics, sigmoid
from .preflight import validate_training_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the frozen teacher ensemble on validation and evaluate "
            "it once on the internal test split."
        ),
    )
    parser.add_argument("--ensemble-dir", required=True)
    parser.add_argument("--teacher-targets-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--minimum-sensitivity", type=float, default=0.9)
    parser.add_argument("--minimum-specificity", type=float, default=0.9)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


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
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Batch size must be positive and workers non-negative")
    if args.ece_bins <= 0:
        raise ValueError("ECE bins must be positive")

    ensemble_dir = Path(args.ensemble_dir).resolve()
    teacher_targets_dir = Path(args.teacher_targets_dir).resolve()
    ensemble, components = validate_teacher_ensemble(ensemble_dir)
    receipt_path = teacher_targets_dir / "teacher_targets.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("evaluation_preprocessing") != (
        DEPLOYMENT_PREPROCESSING_VERSION
    ):
        raise RuntimeError("Teacher preprocessing contract mismatch")
    if receipt.get("normalization_split") != "validation":
        raise RuntimeError("Teacher normalization must use validation")
    if receipt.get("ensemble_json_sha256") != sha256_file(
        ensemble_dir / "ensemble.json"
    ):
        raise RuntimeError("Teacher ensemble receipt hash mismatch")

    preflight = validate_training_input(
        manifest_path=args.manifest,
        images_dir=args.images_dir,
        verify_image_hashes=True,
        image_hash_splits=("validation", "test"),
    )
    if preflight["manifest_sha256"] != receipt.get("manifest_sha256"):
        raise RuntimeError("Teacher and evaluation manifests do not match")
    frame = read_manifest(args.manifest)
    split_frames = {
        split: frame.loc[frame["split"] == split].reset_index(drop=True)
        for split in ("validation", "test")
    }
    if any(split_frame.empty for split_frame in split_frames.values()):
        raise RuntimeError("Validation and test splits must both be non-empty")

    component_names = [str(component["name"]) for component in components]
    normalization = receipt.get("validation_logit_normalization", {})
    if set(normalization) != set(component_names):
        raise RuntimeError("Teacher normalization components are incomplete")

    device = resolve_training_device(args.device)
    transfer_optimizations = uses_cuda_transfer_optimizations(device)
    models, image_size, model_provenance = load_teacher_models(
        ensemble_dir,
        components,
        device,
    )
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for split, split_frame in split_frames.items():
        dataset = LesionDataset(
            split_frame,
            args.images_dir,
            image_size,
            training=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=transfer_optimizations,
            persistent_workers=args.num_workers > 0,
        )
        component_logits = infer_teacher_logits(
            models,
            loader,
            device,
            non_blocking=transfer_optimizations,
            show_progress=True,
        )
        logits_by_name = {
            name: component_logits[:, index].astype(np.float64)
            for index, name in enumerate(component_names)
        }
        predictions[split] = {
            "component_logits": component_logits,
            "ensemble_logits": standardized_logit_average(
                logits_by_name,
                normalization,
                component_names,
            ),
            "targets": split_frame["target"].to_numpy(dtype=np.int64),
            "image_names": split_frame["image_name"].to_numpy(dtype=str),
        }

    validation = predictions["validation"]
    calibration = fit_development_calibration(
        validation["ensemble_logits"],
        validation["targets"],
        minimum_sensitivity=args.minimum_sensitivity,
        minimum_specificity=args.minimum_specificity,
        ece_bins=args.ece_bins,
        minimum_scale=0.05,
        maximum_scale=20.0,
        maximum_abs_bias=20.0,
    )
    calibration.update(
        {
            "candidate_type": "teacher_ensemble",
            "evaluation_preprocessing": DEPLOYMENT_PREPROCESSING_VERSION,
            "ensemble_json_sha256": sha256_file(ensemble_dir / "ensemble.json"),
            "teacher_targets_receipt_sha256": sha256_file(receipt_path),
            "validation_records": int(len(split_frames["validation"])),
            "device": str(device),
        }
    )

    output_dir = create_new_directory(args.output_dir)
    write_json_exclusive(output_dir / "calibration.json", calibration)
    report_splits = {}
    for split, values in predictions.items():
        uncalibrated = sigmoid(values["ensemble_logits"])
        calibrated = apply_calibration(values["ensemble_logits"], calibration)
        low_threshold = float(calibration["low_threshold"])
        high_threshold = float(calibration["high_threshold"])
        report_splits[split] = {
            "records": int(len(values["targets"])),
            "positive_records": int(values["targets"].sum()),
            "uncalibrated": compute_binary_metrics(
                values["targets"],
                uncalibrated,
                threshold=0.5,
                ece_bins=args.ece_bins,
            ),
            "low_threshold": compute_binary_metrics(
                values["targets"],
                calibrated,
                threshold=low_threshold,
                ece_bins=args.ece_bins,
            ),
            "high_threshold": compute_binary_metrics(
                values["targets"],
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
            component_logits=values["component_logits"],
            ensemble_logits=values["ensemble_logits"],
            uncalibrated_probabilities=uncalibrated,
            calibrated_probabilities=calibrated,
            targets=values["targets"],
            image_names=values["image_names"],
        )

    report = {
        "schema_version": 1,
        "status": "complete",
        "candidate_type": "teacher_ensemble",
        "selection_split": "validation",
        "evaluation_preprocessing": DEPLOYMENT_PREPROCESSING_VERSION,
        "manifest_sha256": preflight["manifest_sha256"],
        "ensemble_json_sha256": sha256_file(ensemble_dir / "ensemble.json"),
        "teacher_targets_receipt_sha256": sha256_file(receipt_path),
        "component_order": component_names,
        "components": model_provenance,
        "low_threshold": float(calibration["low_threshold"]),
        "high_threshold": float(calibration["high_threshold"]),
        "splits": report_splits,
        "thresholds_validated": False,
        "research_only": True,
        "device": str(device),
    }
    write_json_exclusive(output_dir / "metrics.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
