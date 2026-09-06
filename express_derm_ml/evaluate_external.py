from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .artifacts import require_absent, write_json_exclusive, write_npz_exclusive
from .calibration import apply_calibration
from .common import load_yaml, sha256_file
from .dataset import (
    DEPLOYMENT_PREPROCESSING_VERSION,
    LesionDataset,
    checkpoint_preprocessing,
    validate_preprocessing_version,
)
from .device import resolve_training_device, uses_cuda_transfer_optimizations
from .manifest import canonical_manifest_sha256, read_manifest
from .metrics import compute_binary_metrics, grouped_bootstrap_intervals, sigmoid
from .model import create_model
from .train import evaluate_loader


def resolve_num_workers(configured: int, override: int | None) -> int:
    num_workers = int(configured if override is None else override)
    if num_workers < 0:
        raise ValueError("--num-workers must be zero or greater")
    return num_workers


def _load_external_dataset_card(
    manifest_path: Path,
    *,
    manifest_sha256: str,
    training_manifest_sha256: str,
    records: int,
    lesions: int,
) -> tuple[dict[str, object], Path]:
    card_path = manifest_path.with_name(
        f"{manifest_path.stem}.dataset.json"
    )
    card = json.loads(card_path.read_text(encoding="utf-8"))
    required_values = {
        "manifest_sha256": manifest_sha256,
        "reference_manifest_sha256": training_manifest_sha256,
        "purpose": "external_research_evaluation_only",
        "patient_identifier_status": (
            "unavailable_in_public_challenge_package"
        ),
        "records": records,
        "lesions": lesions,
        "cross_source_exact_duplicates": 0,
        "cross_source_near_duplicate_candidates": 0,
    }
    for key, expected in required_values.items():
        if card.get(key) != expected:
            raise RuntimeError(
                f"External dataset card mismatch for {key}: "
                f"expected {expected!r}, found {card.get(key)!r}"
            )
    if card.get("training_authorized") is not False:
        raise RuntimeError(
            "External dataset card must explicitly forbid training"
        )
    if not isinstance(card.get("dataset"), dict):
        raise RuntimeError("External dataset card is missing dataset identity")
    if not str(card.get("evaluation_image_type", "")).strip():
        raise RuntimeError(
            "External dataset card is missing evaluation image type"
        )
    return card, card_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen research run on an external manifest.",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--external-manifest", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help=(
            "Override the training-config DataLoader worker count; use 0 on "
            "hosts where multiprocessing shared memory is unavailable."
        ),
    )
    return parser.parse_args()


def _verify_external_images(frame, images_root: Path) -> None:
    for position, row in enumerate(frame.itertuples(index=False), start=1):
        if position == 1 or position % 500 == 0:
            print(
                f"Verifying external image {position:,}/{len(frame):,}",
                flush=True,
            )
        path = (images_root / str(row.image_path)).resolve()
        try:
            path.relative_to(images_root)
        except ValueError as error:
            raise RuntimeError("External image path escapes its root") from error
        if not path.is_file() or sha256_file(path) != str(row.sha256):
            raise RuntimeError(
                f"External image integrity check failed: {row.image_name}"
            )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    external_manifest_path = Path(args.external_manifest)
    output_path = Path(args.output)
    predictions_path = output_path.with_suffix(".npz")
    require_absent([output_path, predictions_path])

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    preprocessing = validate_preprocessing_version(
        str(
            run.get(
                "evaluation_preprocessing",
                DEPLOYMENT_PREPROCESSING_VERSION,
            )
        )
    )
    checkpoint_path = run_dir / "best.pt"
    calibration_path = run_dir / "calibration.json"
    config_path = run_dir / "config.yaml"
    if sha256_file(checkpoint_path) != run.get("checkpoint_sha256"):
        raise RuntimeError("Checkpoint hash does not match run.json")
    if sha256_file(config_path) != run.get("config_sha256"):
        raise RuntimeError("Config hash does not match run.json")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("checkpoint_sha256") != run.get("checkpoint_sha256"):
        raise RuntimeError("Calibration checkpoint hash mismatch")
    if calibration.get("evaluation_preprocessing") != preprocessing:
        raise RuntimeError("Calibration preprocessing contract mismatch")

    frame = read_manifest(
        external_manifest_path,
        additional_text_columns=(
            "diagnosis_class",
            "diagnosis_full",
            "diagnosis_confirm_type",
            "patient_identifier_status",
            "image_type",
        ),
    )
    manifest_sha256 = canonical_manifest_sha256(frame)
    recorded_manifest_sha256 = Path(
        f"{external_manifest_path}.sha256"
    ).read_text(encoding="ascii").strip()
    if manifest_sha256 != recorded_manifest_sha256:
        raise RuntimeError("External manifest hash mismatch")
    if frame["lesion_id"].duplicated().any():
        raise RuntimeError(
            "External evaluation requires exactly one image per lesion"
        )
    if set(frame["target_melanoma"].astype(int).unique()) != {0, 1}:
        raise RuntimeError("External melanoma target must be binary")
    external_card, external_card_path = _load_external_dataset_card(
        external_manifest_path,
        manifest_sha256=manifest_sha256,
        training_manifest_sha256=str(run["manifest_sha256"]),
        records=int(len(frame)),
        lesions=int(frame["lesion_id"].nunique()),
    )
    if not frame["image_type"].eq(
        str(external_card["evaluation_image_type"])
    ).all():
        raise RuntimeError(
            "External manifest contains an unexpected image modality"
        )
    images_root = Path(args.images_dir).resolve()
    _verify_external_images(frame, images_root)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint_preprocessing(checkpoint) != preprocessing:
        raise RuntimeError("Run and checkpoint preprocessing do not match")
    config = load_yaml(config_path)
    model_task = str(
        config.get("task", {}).get("name", "binary_melanoma_attention")
    )
    dataset_frame = frame.copy()
    dataset_frame["target"] = dataset_frame["target_melanoma"].astype(int)
    dataset = LesionDataset(
        dataset_frame,
        images_root,
        int(checkpoint["image_size"]),
        training=False,
        preprocessing=preprocessing,
    )
    device = resolve_training_device(args.device)
    transfer_optimizations = uses_cuda_transfer_optimizations(device)
    num_workers = resolve_num_workers(
        int(config["training"]["num_workers"]),
        args.num_workers,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=transfer_optimizations,
    )
    model = create_model(checkpoint["architecture"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    logits, melanoma_targets = evaluate_loader(
        model,
        loader,
        device,
        non_blocking=transfer_optimizations,
    )
    uncalibrated = sigmoid(logits)
    calibrated = apply_calibration(logits, calibration)
    low_threshold = float(calibration["low_threshold"])
    high_threshold = float(calibration["high_threshold"])
    broad_targets = frame["target_broad_malignancy"].to_numpy(dtype=np.int64)
    groups = frame["lesion_id"].to_numpy(dtype=str)
    ece_bins = int(config["calibration"]["ece_bins"])
    bootstrap_samples = int(config.get("evaluation", {}).get(
        "bootstrap_samples",
        2000,
    ))
    confidence_level = float(config.get("evaluation", {}).get(
        "confidence_level",
        0.95,
    ))

    melanoma_low = compute_binary_metrics(
        melanoma_targets,
        calibrated,
        threshold=low_threshold,
        ece_bins=ece_bins,
    )
    melanoma_high = compute_binary_metrics(
        melanoma_targets,
        calibrated,
        threshold=high_threshold,
        ece_bins=ece_bins,
    )
    low_intervals = grouped_bootstrap_intervals(
        melanoma_targets,
        calibrated,
        groups,
        threshold=low_threshold,
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=int(config["seed"]),
    )
    high_intervals = grouped_bootstrap_intervals(
        melanoma_targets,
        calibrated,
        groups,
        threshold=high_threshold,
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=int(config["seed"]),
    )
    for intervals in (low_intervals, high_intervals):
        intervals["resampling_unit"] = "lesion"
        intervals["limitation"] = (
            "MILK10k public package has no patient identifier; confidence "
            "intervals cannot account for within-patient correlation"
        )

    class_breakdown = {}
    classes = sorted(frame["diagnosis_class"].astype(str).unique())
    for class_name in classes:
        members = frame["diagnosis_class"].eq(class_name).to_numpy()
        class_breakdown[class_name] = {
            "n": int(members.sum()),
            "mean_score": float(calibrated[members].mean()),
            "low_or_higher": int((calibrated[members] >= low_threshold).sum()),
            "high": int((calibrated[members] >= high_threshold).sum()),
        }

    if model_task == "binary_broad_malignancy_attention":
        broad_warning = (
            "This run explicitly targets broad malignancy attention, but its "
            "frozen thresholds were calibrated on the available internal "
            "melanoma-versus-benign validation cohort; no internal BCC/SCC/AK "
            "validation cohort was available"
        )
    else:
        broad_warning = (
            "The frozen model was trained for melanoma attention; this "
            "secondary analysis does not redefine its target"
        )

    report = {
        "schema_version": 2,
        "evaluation_preprocessing": preprocessing,
        "status": "external_research_evaluation",
        "run_checkpoint_sha256": run["checkpoint_sha256"],
        "training_manifest_sha256": run["manifest_sha256"],
        "external_manifest_sha256": manifest_sha256,
        "external_dataset_card_sha256": sha256_file(external_card_path),
        "external_dataset": external_card["dataset"],
        "external_image_type": external_card["evaluation_image_type"],
        "model_task": model_task,
        "external_records": int(len(frame)),
        "external_lesions": int(frame["lesion_id"].nunique()),
        "patient_identifier_status": external_card[
            "patient_identifier_status"
        ],
        "calibration_method": calibration["method"],
        "calibration_sha256": sha256_file(calibration_path),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "melanoma_vs_all": {
            "uncalibrated": compute_binary_metrics(
                melanoma_targets,
                uncalibrated,
                threshold=0.5,
                ece_bins=ece_bins,
            ),
            "low_threshold": melanoma_low,
            "high_threshold": melanoma_high,
            "confidence_intervals": {
                "low_threshold": low_intervals,
                "high_threshold": high_intervals,
            },
        },
        "broad_malignancy_exploratory": {
            "warning": broad_warning,
            "low_threshold": compute_binary_metrics(
                broad_targets,
                calibrated,
                threshold=low_threshold,
                ece_bins=ece_bins,
            ),
            "high_threshold": compute_binary_metrics(
                broad_targets,
                calibrated,
                threshold=high_threshold,
                ece_bins=ece_bins,
            ),
        },
        "diagnosis_class_breakdown": class_breakdown,
        "thresholds_validated": False,
        "training_authorized_from_external_set": False,
        "research_only": True,
        "device": str(device),
        "num_workers": num_workers,
    }
    write_npz_exclusive(
        predictions_path,
        logits=logits,
        uncalibrated_probabilities=uncalibrated,
        calibrated_probabilities=calibrated,
        melanoma_targets=melanoma_targets,
        broad_malignancy_targets=broad_targets,
        image_names=frame["image_name"].to_numpy(dtype=str),
        lesion_ids=groups,
        diagnosis_classes=frame["diagnosis_class"].to_numpy(dtype=str),
    )
    write_json_exclusive(output_path, report)
    print(report)


if __name__ == "__main__":
    main()
