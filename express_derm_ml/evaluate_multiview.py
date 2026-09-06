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
from .dataset import checkpoint_preprocessing
from .device import resolve_training_device, uses_cuda_transfer_optimizations
from .manifest import canonical_manifest_sha256, read_manifest
from .metrics import compute_binary_metrics
from .model import create_model
from .path_safety import resolve_manifest_image_path
from .multiview import (
    MULTIVIEW_PROTOCOL_VERSION,
    POLICY_VIEW_NAMES,
    VIEW_NAMES,
    MultiViewDataset,
    attention_from_consensus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate deterministic multi-view consensus without changing "
            "model weights, calibration, or frozen thresholds."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument(
        "--split",
        choices=("validation", "test", "external"),
        required=True,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-predictions")
    parser.add_argument(
        "--runtime-manifest",
        help="Optional published model manifest whose identity must match.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--external-target",
        choices=("melanoma", "broad_malignancy"),
        default="melanoma",
        help="Binary target used only when --split external.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--parity-atol",
        type=float,
        default=2e-3,
        help=(
            "Maximum original-view logit drift allowed across different MPS "
            "batch shapes; defaults to 0.002."
        ),
    )
    return parser.parse_args()


def _verify_images(frame, images_dir: Path) -> None:
    root = images_dir.resolve()
    for position, row in enumerate(frame.itertuples(index=False), start=1):
        if position == 1 or position % 1000 == 0:
            print(f"Verified image {position:,}/{len(frame):,}", flush=True)
        try:
            image_path = resolve_manifest_image_path(
                root,
                str(row.image_path),
            )
        except ValueError as error:
            raise RuntimeError("Manifest image path escapes its root") from error
        if not image_path.is_file():
            raise RuntimeError(f"Missing image: {row.image_name}")
        if sha256_file(image_path) != str(row.sha256):
            raise RuntimeError(f"Image hash mismatch: {row.image_name}")


@torch.inference_mode()
def _evaluate(model, loader, device, *, non_blocking: bool):
    model.eval()
    collected_logits: list[np.ndarray] = []
    collected_targets: list[np.ndarray] = []
    collected_names: list[str] = []
    for batch_index, (views, targets, names) in enumerate(loader, start=1):
        if batch_index == 1 or batch_index % 100 == 0:
            print(
                f"Evaluating multi-view batch {batch_index:,}/{len(loader):,}",
                flush=True,
            )
        batch_size, view_count, channels, height, width = views.shape
        flattened = views.reshape(
            batch_size * view_count,
            channels,
            height,
            width,
        )
        output = model(
            flattened.to(device, non_blocking=non_blocking)
        ).reshape(batch_size, view_count)
        collected_logits.append(output.cpu().numpy().astype(np.float64))
        collected_targets.append(targets.numpy().astype(np.int64))
        collected_names.extend(str(name) for name in names)
    return (
        np.concatenate(collected_logits, axis=0),
        np.concatenate(collected_targets, axis=0),
        np.asarray(collected_names, dtype=str),
    )


def _validate_reference_predictions(
    reference_path: Path,
    *,
    logits: np.ndarray,
    targets: np.ndarray,
    image_names: np.ndarray,
    atol: float,
) -> dict[str, object]:
    reference_sha256 = sha256_file(reference_path)
    with np.load(reference_path, allow_pickle=False) as reference:
        reference_names = np.asarray(reference["image_names"], dtype=str)
        reference_target_key = (
            "targets" if "targets" in reference.files else "melanoma_targets"
        )
        if reference_target_key not in reference.files:
            raise RuntimeError("Reference predictions have no melanoma target")
        reference_targets = np.asarray(
            reference[reference_target_key],
            dtype=np.int64,
        )
        reference_logits = np.asarray(reference["logits"], dtype=np.float64)
    if not np.array_equal(image_names, reference_names):
        raise RuntimeError("Reference prediction image order does not match")
    if not np.array_equal(targets, reference_targets):
        raise RuntimeError("Reference prediction targets do not match")
    maximum_absolute_error = float(
        np.max(np.abs(logits[:, 0] - reference_logits))
    )
    if maximum_absolute_error > atol:
        raise RuntimeError(
            "Original-view parity failed: "
            f"maximum error {maximum_absolute_error} exceeds {atol}"
        )
    return {
        "reference_predictions_sha256": reference_sha256,
        "reference_target_key": reference_target_key,
        "maximum_absolute_logit_error": maximum_absolute_error,
        "absolute_tolerance": atol,
        "status": "pass",
    }


def _policy_report(
    targets: np.ndarray,
    calibrated_probabilities: np.ndarray,
    *,
    policy_name: str,
    low_threshold: float,
    high_threshold: float,
    ece_bins: int,
    baseline_high: dict[str, float | int],
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    levels, minimum, maximum = attention_from_consensus(
        calibrated_probabilities,
        policy_name=policy_name,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    high_metrics = compute_binary_metrics(
        targets,
        minimum,
        threshold=high_threshold,
        ece_bins=ece_bins,
    )
    low_metrics = compute_binary_metrics(
        targets,
        maximum,
        threshold=low_threshold,
        ece_bins=ece_bins,
    )
    baseline_false_positive = int(baseline_high["false_positive"])
    baseline_true_positive = int(baseline_high["true_positive"])
    return (
        {
            "view_names": list(POLICY_VIEW_NAMES[policy_name]),
            "high_consensus_metrics": high_metrics,
            "low_consensus_metrics": low_metrics,
            "attention_counts": {
                level: int(np.sum(levels == level))
                for level in ("low", "inconclusive", "high")
            },
            "change_from_original_high": {
                "false_positive_delta": int(high_metrics["false_positive"])
                - baseline_false_positive,
                "false_positive_reduction_fraction": (
                    float(
                        baseline_false_positive
                        - int(high_metrics["false_positive"])
                    )
                    / max(baseline_false_positive, 1)
                ),
                "true_positive_delta": int(high_metrics["true_positive"])
                - baseline_true_positive,
                "sensitivity_delta": float(high_metrics["sensitivity"])
                - float(baseline_high["sensitivity"]),
                "specificity_delta": float(high_metrics["specificity"])
                - float(baseline_high["specificity"]),
                "precision_delta": float(high_metrics["precision"])
                - float(baseline_high["precision"]),
            },
        },
        levels,
        minimum,
        maximum,
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be zero or greater")
    if args.parity_atol < 0:
        raise ValueError("--parity-atol cannot be negative")
    if args.split != "external" and args.external_target != "melanoma":
        raise ValueError("--external-target is valid only for external data")

    run_dir = Path(args.run_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    predictions_path = output_path.with_suffix(".npz")
    require_absent([output_path, predictions_path])

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    checkpoint_path = run_dir / "best.pt"
    config_path = run_dir / "config.yaml"
    calibration_path = run_dir / "calibration.json"
    if sha256_file(checkpoint_path) != str(run["checkpoint_sha256"]):
        raise RuntimeError("Checkpoint hash does not match run.json")
    if sha256_file(config_path) != str(run["config_sha256"]):
        raise RuntimeError("Config hash does not match run.json")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("checkpoint_sha256") != run["checkpoint_sha256"]:
        raise RuntimeError("Calibration checkpoint hash mismatch")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    preprocessing = checkpoint_preprocessing(checkpoint)
    runtime_evidence = None
    if args.runtime_manifest:
        runtime_manifest_path = Path(args.runtime_manifest).resolve()
        runtime_manifest = json.loads(
            runtime_manifest_path.read_text(encoding="utf-8")
        )
        runtime_preprocessing = runtime_manifest.get("preprocessing", {})
        expected_values = {
            "checkpoint_sha256": run["checkpoint_sha256"],
            "calibration_sha256": sha256_file(calibration_path),
            "architecture": checkpoint["architecture"],
            "image_size": int(checkpoint["image_size"]),
        }
        for key, expected in expected_values.items():
            if runtime_manifest.get(key) != expected:
                raise RuntimeError(
                    f"Published runtime manifest mismatch for {key}"
                )
        if runtime_preprocessing.get("version") != preprocessing:
            raise RuntimeError("Published runtime preprocessing mismatch")
        runtime_evidence = {
            "manifest_sha256": sha256_file(runtime_manifest_path),
            "version": runtime_manifest.get("version"),
            "release_version": runtime_manifest.get("release_version"),
            "model_sha256": runtime_manifest.get("model_sha256"),
        }
    complete_frame = read_manifest(
        manifest_path,
        additional_text_columns=("image_type",),
    )
    manifest_sha256 = canonical_manifest_sha256(complete_frame)
    if args.split == "external":
        recorded = Path(f"{manifest_path}.sha256").read_text(
            encoding="ascii"
        ).strip()
        if recorded != manifest_sha256:
            raise RuntimeError("External manifest hash mismatch")
        card_path = manifest_path.with_name(f"{manifest_path.stem}.dataset.json")
        card = json.loads(card_path.read_text(encoding="utf-8"))
        if card.get("reference_manifest_sha256") != run["manifest_sha256"]:
            raise RuntimeError("External dataset was not checked against this run")
        if card.get("training_authorized") is not False:
            raise RuntimeError("External dataset must forbid training")
        frame = complete_frame.copy()
        target_column = (
            "target_melanoma"
            if args.external_target == "melanoma"
            else "target_broad_malignancy"
        )
        frame["target"] = frame[target_column].astype(int)
        external_evidence: dict[str, object] | None = {
            "dataset_card_sha256": sha256_file(card_path),
            "training_authorized": False,
            "evaluation_image_type": card["evaluation_image_type"],
            "target_column": target_column,
        }
    else:
        if manifest_sha256 != run["manifest_sha256"]:
            raise RuntimeError("Internal manifest hash does not match run.json")
        frame = complete_frame.loc[
            complete_frame["split"].astype(str) == args.split
        ].copy()
        external_evidence = None
    if frame.empty or set(frame["target"].astype(int).unique()) != {0, 1}:
        raise RuntimeError("Evaluation split must contain both binary targets")

    _verify_images(frame, Path(args.images_dir))
    dataset = MultiViewDataset(
        frame,
        args.images_dir,
        int(checkpoint["image_size"]),
        preprocessing=preprocessing,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    device = resolve_training_device(args.device)
    model = create_model(str(checkpoint["architecture"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    logits, targets, image_names = _evaluate(
        model,
        loader,
        device,
        non_blocking=uses_cuda_transfer_optimizations(device),
    )
    calibrated = apply_calibration(logits, calibration)
    low_threshold = float(calibration["low_threshold"])
    high_threshold = float(calibration["high_threshold"])
    config = load_yaml(config_path)
    ece_bins = int(config["calibration"]["ece_bins"])
    baseline_low = compute_binary_metrics(
        targets,
        calibrated[:, 0],
        threshold=low_threshold,
        ece_bins=ece_bins,
    )
    baseline_high = compute_binary_metrics(
        targets,
        calibrated[:, 0],
        threshold=high_threshold,
        ece_bins=ece_bins,
    )

    parity = None
    if args.reference_predictions:
        parity = _validate_reference_predictions(
            Path(args.reference_predictions),
            logits=logits,
            targets=targets,
            image_names=image_names,
            atol=args.parity_atol,
        )

    policies: dict[str, object] = {}
    policy_arrays: dict[str, np.ndarray] = {}
    for policy_name in POLICY_VIEW_NAMES:
        report, levels, minimum, maximum = _policy_report(
            targets,
            calibrated,
            policy_name=policy_name,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            ece_bins=ece_bins,
            baseline_high=baseline_high,
        )
        policies[policy_name] = report
        policy_arrays[f"{policy_name}_attention_levels"] = levels
        policy_arrays[f"{policy_name}_minimum_probability"] = minimum
        policy_arrays[f"{policy_name}_maximum_probability"] = maximum

    report = {
        "schema_version": 1,
        "status": "exploratory_research_evaluation",
        "protocol_version": MULTIVIEW_PROTOCOL_VERSION,
        "split": args.split,
        "evaluation_target": (
            args.external_target if args.split == "external" else "target"
        ),
        "view_names": list(VIEW_NAMES),
        "checkpoint_sha256": run["checkpoint_sha256"],
        "calibration_sha256": sha256_file(calibration_path),
        "manifest_sha256": manifest_sha256,
        "preprocessing": preprocessing,
        "image_size": int(checkpoint["image_size"]),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "threshold_policy": "original_validation_frozen_per_view",
        "original_view": {
            "low_threshold_metrics": baseline_low,
            "high_threshold_metrics": baseline_high,
        },
        "policies": policies,
        "parity": parity,
        "published_runtime": runtime_evidence,
        "external_evidence": external_evidence,
        "thresholds_validated": False,
        "deployment_authorized": False,
        "research_only": True,
        "device": str(device),
        "records": int(len(targets)),
    }
    write_npz_exclusive(
        predictions_path,
        logits=logits,
        calibrated_probabilities=calibrated,
        targets=targets,
        image_names=image_names,
        view_names=np.asarray(VIEW_NAMES, dtype=str),
        **policy_arrays,
    )
    write_json_exclusive(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
