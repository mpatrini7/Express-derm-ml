from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .artifacts import (
    require_absent,
    write_json_exclusive,
    write_npz_exclusive,
)
from .common import load_yaml, sha256_file
from .calibration import apply_calibration
from .dataset import (
    DEPLOYMENT_PREPROCESSING_VERSION,
    LesionDataset,
    checkpoint_preprocessing,
    validate_preprocessing_version,
)
from .device import resolve_training_device, uses_cuda_transfer_optimizations
from .manifest import read_manifest
from .metrics import (
    compute_binary_metrics,
    grouped_bootstrap_intervals,
    sigmoid,
)
from .model import create_model
from .preflight import validate_training_input
from .train import evaluate_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen development calibration on the test split."
        ),
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Override DataLoader workers; use 0 for robust macOS execution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics_test.json"
    predictions_path = run_dir / "predictions_test.npz"
    require_absent([metrics_path, predictions_path])

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if run.get("status") != "complete":
        raise RuntimeError("Training run is not complete")
    preprocessing = validate_preprocessing_version(
        str(
            run.get(
                "evaluation_preprocessing",
                DEPLOYMENT_PREPROCESSING_VERSION,
            )
        )
    )
    config_path = run_dir / "config.yaml"
    if sha256_file(config_path) != run.get("config_sha256"):
        raise RuntimeError("Config hash does not match run.json")
    calibration_path = run_dir / "calibration.json"
    calibration = json.loads(
        calibration_path.read_text(encoding="utf-8")
    )
    if calibration.get("selection_split") != "validation":
        raise RuntimeError("Calibration was not selected on validation data")
    if calibration.get("evaluation_preprocessing") != preprocessing:
        raise RuntimeError("Calibration preprocessing contract mismatch")
    checkpoint_path = run_dir / "best.pt"
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != run.get("checkpoint_sha256"):
        raise RuntimeError("Checkpoint hash does not match run.json")
    if checkpoint_sha256 != calibration.get("checkpoint_sha256"):
        raise RuntimeError("Calibration checkpoint hash mismatch")
    preflight = validate_training_input(
        manifest_path=run_dir / "manifest.csv",
        images_dir=args.images_dir,
        verify_image_hashes=True,
        image_hash_splits=("test",),
    )
    if preflight["manifest_sha256"] != run.get("manifest_sha256"):
        raise RuntimeError("Run manifest hash does not match run.json")
    if preflight["manifest_sha256"] != calibration.get("manifest_sha256"):
        raise RuntimeError("Calibration manifest hash mismatch")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint_preprocessing(checkpoint) != preprocessing:
        raise RuntimeError("Run and checkpoint preprocessing do not match")
    config = load_yaml(config_path)
    frame = read_manifest(run_dir / "manifest.csv")
    test_frame = frame.loc[frame["split"] == "test"].copy()
    dataset = LesionDataset(
        test_frame,
        args.images_dir,
        int(checkpoint["image_size"]),
        training=False,
        preprocessing=preprocessing,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=(
            int(config["training"]["num_workers"])
            if args.num_workers is None
            else args.num_workers
        ),
    )
    if loader.num_workers < 0:
        raise ValueError("--num-workers must be zero or greater")

    device = resolve_training_device(args.device)
    transfer_optimizations = uses_cuda_transfer_optimizations(device)
    model = create_model(checkpoint["architecture"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    logits, targets = evaluate_loader(
        model,
        loader,
        device,
        non_blocking=transfer_optimizations,
    )
    uncalibrated_probabilities = sigmoid(logits)
    calibrated_probabilities = apply_calibration(logits, calibration)
    low_threshold = float(calibration["low_threshold"])
    high_threshold = float(calibration["high_threshold"])
    ece_bins = int(config["calibration"]["ece_bins"])

    attention_levels = np.full(len(targets), "inconclusive", dtype="<U12")
    attention_levels[calibrated_probabilities < low_threshold] = "low"
    attention_levels[calibrated_probabilities >= high_threshold] = "high"
    calibration_parameters = {}
    if calibration["method"] == "affine_logistic_scaling":
        calibration_parameters = {
            "logit_scale": float(calibration["logit_scale"]),
            "logit_bias": float(calibration["logit_bias"]),
        }
    elif calibration["method"] == "temperature_scaling":
        calibration_parameters = {
            "temperature": float(calibration["temperature"]),
        }
    else:
        raise RuntimeError(
            f"Unsupported calibration method: {calibration['method']}"
        )
    metrics = {
        "schema_version": 2,
        "split": "test",
        "selection_split": "validation",
        "manifest_sha256": run["manifest_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_sha256": sha256_file(calibration_path),
        "calibration_method": calibration["method"],
        "evaluation_preprocessing": preprocessing,
        **calibration_parameters,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "decision_policy_version": "binary-extremes-v1",
        "uncalibrated_metrics": compute_binary_metrics(
            targets,
            uncalibrated_probabilities,
            threshold=0.5,
            ece_bins=ece_bins,
        ),
        "calibrated_low_threshold_metrics": compute_binary_metrics(
            targets,
            calibrated_probabilities,
            threshold=low_threshold,
            ece_bins=ece_bins,
        ),
        "calibrated_high_threshold_metrics": compute_binary_metrics(
            targets,
            calibrated_probabilities,
            threshold=high_threshold,
            ece_bins=ece_bins,
        ),
        "attention_counts": {
            level: int((attention_levels == level).sum())
            for level in ("low", "inconclusive", "high")
        },
        "confidence_intervals": {
            "low_threshold": grouped_bootstrap_intervals(
                targets,
                calibrated_probabilities,
                test_frame["group_id"].to_numpy(dtype=str),
                threshold=low_threshold,
                samples=int(
                    config.get("evaluation", {}).get(
                        "bootstrap_samples",
                        2000,
                    )
                ),
                confidence_level=float(
                    config.get("evaluation", {}).get(
                        "confidence_level",
                        0.95,
                    )
                ),
                seed=int(config["seed"]),
            ),
            "high_threshold": grouped_bootstrap_intervals(
                targets,
                calibrated_probabilities,
                test_frame["group_id"].to_numpy(dtype=str),
                threshold=high_threshold,
                samples=int(
                    config.get("evaluation", {}).get(
                        "bootstrap_samples",
                        2000,
                    )
                ),
                confidence_level=float(
                    config.get("evaluation", {}).get(
                        "confidence_level",
                        0.95,
                    )
                ),
                seed=int(config["seed"]),
            ),
        },
        "threshold_status": "development_frozen",
        "thresholds_validated": False,
        "research_only": True,
        "device": str(device),
    }
    metrics["confidence_intervals"]["low_threshold"][
        "resampling_unit"
    ] = "patient"
    metrics["confidence_intervals"]["high_threshold"][
        "resampling_unit"
    ] = "patient"

    write_npz_exclusive(
        predictions_path,
        logits=logits,
        uncalibrated_probabilities=uncalibrated_probabilities,
        calibrated_probabilities=calibrated_probabilities,
        targets=targets,
        attention_levels=attention_levels,
        image_names=test_frame["image_name"].to_numpy(dtype=str),
    )
    write_json_exclusive(metrics_path, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
