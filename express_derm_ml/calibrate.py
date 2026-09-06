from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .artifacts import (
    copy_file_exclusive,
    create_new_directory,
    require_absent,
    write_json_exclusive,
    write_npz_exclusive,
)
from .calibration import apply_calibration, fit_development_calibration
from .common import load_yaml, sha256_file
from .dataset import (
    DEPLOYMENT_PREPROCESSING_VERSION,
    LesionDataset,
    checkpoint_preprocessing,
    validate_preprocessing_version,
)
from .device import resolve_training_device, uses_cuda_transfer_optimizations
from .manifest import read_manifest
from .metrics import sigmoid
from .model import create_model
from .preflight import validate_training_input
from .train import evaluate_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit development-only calibration and attention thresholds "
            "using the validation split."
        ),
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--output-dir",
        help=(
            "Create an immutable derived run with identical training artifacts "
            "and a new calibration. Omit to calibrate the source run."
        ),
    )
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


def clone_training_run_for_recalibration(
    source_dir: Path,
    output_dir: Path,
) -> Path:
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("Derived calibration output must differ from source")
    source_run = json.loads(
        (source_dir / "run.json").read_text(encoding="utf-8")
    )
    if source_run.get("status") != "complete":
        raise RuntimeError("Source training run is not complete")
    required_files = (
        "best.pt",
        "config.yaml",
        "history.json",
        "manifest.csv",
        "manifest.csv.sha256",
        "manifest.near_duplicates.csv",
        "manifest.report.json",
        "preflight.json",
    )
    missing = [name for name in required_files if not (source_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Source training artifacts are missing: {missing}")
    target = create_new_directory(output_dir)
    for name in required_files:
        copy_file_exclusive(source_dir / name, target / name)
    derived_run = {
        **source_run,
        "schema_version": max(int(source_run.get("schema_version", 1)), 2),
        "derived_calibration": True,
        "derived_from_checkpoint_sha256": source_run["checkpoint_sha256"],
        "training_artifacts_unchanged": True,
        "evaluation_preprocessing": source_run.get(
            "evaluation_preprocessing",
            DEPLOYMENT_PREPROCESSING_VERSION,
        ),
    }
    write_json_exclusive(target / "run.json", derived_run)
    return target


def main() -> None:
    args = parse_args()
    source_run_dir = Path(args.run_dir)
    run_dir = (
        clone_training_run_for_recalibration(
            source_run_dir,
            Path(args.output_dir),
        )
        if args.output_dir
        else source_run_dir
    )
    calibration_path = run_dir / "calibration.json"
    predictions_path = run_dir / "predictions_validation.npz"
    require_absent([calibration_path, predictions_path])

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
    checkpoint_path = run_dir / "best.pt"
    config_path = run_dir / "config.yaml"
    if sha256_file(config_path) != run.get("config_sha256"):
        raise RuntimeError("Config hash does not match run.json")
    if sha256_file(checkpoint_path) != run.get("checkpoint_sha256"):
        raise RuntimeError("Checkpoint hash does not match run.json")
    preflight = validate_training_input(
        manifest_path=run_dir / "manifest.csv",
        images_dir=args.images_dir,
        verify_image_hashes=True,
        image_hash_splits=("validation",),
    )
    if preflight["manifest_sha256"] != run.get("manifest_sha256"):
        raise RuntimeError("Run manifest hash does not match run.json")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint_preprocessing(checkpoint) != preprocessing:
        raise RuntimeError("Run and checkpoint preprocessing do not match")
    config = load_yaml(config_path)
    frame = read_manifest(run_dir / "manifest.csv")
    validation_frame = frame.loc[frame["split"] == "validation"].copy()
    dataset = LesionDataset(
        validation_frame,
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

    calibration_config = config["calibration"]
    calibration = fit_development_calibration(
        logits,
        targets,
        minimum_sensitivity=float(
            calibration_config["minimum_sensitivity"]
        ),
        minimum_specificity=float(
            calibration_config["minimum_specificity"]
        ),
        ece_bins=int(calibration_config["ece_bins"]),
        minimum_scale=float(calibration_config.get("minimum_scale", 0.05)),
        maximum_scale=float(calibration_config.get("maximum_scale", 20.0)),
        maximum_abs_bias=float(
            calibration_config.get("maximum_abs_bias", 20.0)
        ),
    )
    calibration.update(
        {
            "manifest_sha256": run["manifest_sha256"],
            "checkpoint_sha256": run["checkpoint_sha256"],
            "validation_records": int(len(validation_frame)),
            "device": str(device),
            "evaluation_preprocessing": preprocessing,
        }
    )
    uncalibrated_probabilities = sigmoid(logits)
    calibrated_probabilities = apply_calibration(logits, calibration)
    write_npz_exclusive(
        predictions_path,
        logits=logits,
        uncalibrated_probabilities=uncalibrated_probabilities,
        calibrated_probabilities=calibrated_probabilities,
        targets=targets,
        image_names=validation_frame["image_name"].to_numpy(dtype=str),
    )
    write_json_exclusive(calibration_path, calibration)
    print(calibration)


if __name__ == "__main__":
    main()
