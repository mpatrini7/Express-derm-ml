from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .artifacts import create_new_directory, write_json_exclusive, write_npz_exclusive
from .calibration import apply_calibration
from .common import load_yaml, sha256_file
from .dataset import DEPLOYMENT_PREPROCESSING_VERSION, LesionDataset
from .device import resolve_training_device, uses_cuda_transfer_optimizations
from .generate_teacher_targets import infer_teacher_logits
from .manifest import read_manifest
from .metrics import compute_binary_metrics, sigmoid
from .model import create_model
from .preflight import validate_training_input


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _probability_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        np.asarray(probabilities, dtype=np.float64),
        1e-8,
        1.0 - 1e-8,
    )
    return np.log(clipped / (1.0 - clipped))


def select_two_component_weights(
    component_logits: dict[str, np.ndarray],
    targets: np.ndarray,
    *,
    generalist_name: str,
    grid_step: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    names = sorted(component_logits)
    if len(names) != 2 or generalist_name not in names:
        raise ValueError("Weight selection requires exactly two components")
    if not 0.0 < grid_step <= 1.0 or not np.isclose(
        round(1.0 / grid_step) * grid_step,
        1.0,
    ):
        raise ValueError("Weight grid step must divide one exactly")
    specialist_name = next(name for name in names if name != generalist_name)
    arrays = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in component_logits.items()
    }
    targets = np.asarray(targets, dtype=np.int64)
    if any(values.shape != targets.shape for values in arrays.values()):
        raise ValueError("Component logits and targets must align")
    if set(np.unique(targets)) != {0, 1}:
        raise ValueError("Weight selection requires binary validation targets")

    results: list[dict[str, float]] = []
    steps = int(round(1.0 / grid_step))
    for index in range(steps + 1):
        generalist_weight = float(index / steps)
        specialist_weight = float(1.0 - generalist_weight)
        blended = (
            generalist_weight * arrays[generalist_name]
            + specialist_weight * arrays[specialist_name]
        )
        metrics = compute_binary_metrics(targets, sigmoid(blended))
        results.append(
            {
                "generalist_weight": generalist_weight,
                "specialist_weight": specialist_weight,
                "pr_auc": float(metrics["pr_auc"]),
                "roc_auc": float(metrics["roc_auc"]),
            }
        )
    selected = max(
        results,
        key=lambda row: (
            row["pr_auc"],
            row["roc_auc"],
            row["generalist_weight"],
        ),
    )
    return (
        {
            generalist_name: selected["generalist_weight"],
            specialist_name: selected["specialist_weight"],
        },
        results,
    )


def _load_component(
    name: str,
    run_dir: Path,
    validation_names: np.ndarray,
    validation_targets: np.ndarray,
) -> tuple[dict[str, Any], torch.nn.Module, np.ndarray]:
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    calibration_path = run_dir / "calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    checkpoint_path = run_dir / "best.pt"
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if run.get("status") != "complete":
        raise RuntimeError(f"Teacher component is incomplete: {name}")
    if run.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError(f"Teacher checkpoint hash mismatch: {name}")
    if calibration.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError(f"Teacher calibration hash mismatch: {name}")
    if run.get("evaluation_preprocessing") != DEPLOYMENT_PREPROCESSING_VERSION:
        raise RuntimeError(f"Teacher preprocessing mismatch: {name}")
    if calibration.get("evaluation_preprocessing") != (
        DEPLOYMENT_PREPROCESSING_VERSION
    ):
        raise RuntimeError(f"Teacher calibration preprocessing mismatch: {name}")

    with np.load(run_dir / "predictions_validation.npz") as predictions:
        if not np.array_equal(
            predictions["image_names"].astype(str),
            validation_names,
        ):
            raise RuntimeError(f"Teacher validation image mismatch: {name}")
        if not np.array_equal(
            predictions["targets"].astype(np.int64),
            validation_targets,
        ):
            raise RuntimeError(f"Teacher validation target mismatch: {name}")
        raw_logits = predictions["logits"].astype(np.float64)
        recorded_probabilities = predictions[
            "calibrated_probabilities"
        ].astype(np.float64)
    recalculated = apply_calibration(raw_logits, calibration)
    if not np.allclose(recalculated, recorded_probabilities, atol=1e-7):
        raise RuntimeError(f"Teacher validation calibration mismatch: {name}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model = create_model(checkpoint["architecture"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    provenance = {
        "name": name,
        "run_dir": str(run_dir),
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_sha256": sha256_file(calibration_path),
        "architecture": str(checkpoint["architecture"]),
        "image_size": int(checkpoint["image_size"]),
        "logit_scale": float(calibration["logit_scale"]),
        "logit_bias": float(calibration["logit_bias"]),
    }
    return provenance, model.eval(), _probability_logit(recalculated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create train-only targets from a validation-selected weighted "
            "specialist/generalist teacher."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    components_config = config.get("components")
    if not isinstance(components_config, list) or len(components_config) != 2:
        raise ValueError("Teacher config requires exactly two components")
    names = [str(component["name"]) for component in components_config]
    if len(set(names)) != 2:
        raise ValueError("Teacher component names must be unique")
    generalist_name = str(config["selection"]["generalist_name"])
    if generalist_name not in names:
        raise ValueError("Configured generalist component is unavailable")
    if config["selection"].get("split") != "validation":
        raise ValueError("Teacher weights must be selected on validation")
    if config["selection"].get("metric") != "pr_auc":
        raise ValueError("Teacher weight selection metric must be PR-AUC")
    grid_step = float(config["selection"]["grid_step"])
    batch_size = int(config["inference"]["batch_size"])
    num_workers = int(config["inference"]["num_workers"])
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("Invalid teacher inference configuration")

    preflight = validate_training_input(
        manifest_path=args.manifest,
        images_dir=args.images_dir,
        verify_image_hashes=True,
        image_hash_splits=("train", "validation"),
    )
    frame = read_manifest(args.manifest)
    training_frame = frame.loc[frame["split"] == "train"].reset_index(
        drop=True
    )
    validation_frame = frame.loc[
        frame["split"] == "validation"
    ].reset_index(drop=True)
    if training_frame.empty or validation_frame.empty:
        raise RuntimeError("Teacher generation requires train and validation")
    validation_names = validation_frame["image_name"].to_numpy(dtype=str)
    validation_targets = validation_frame["target"].to_numpy(dtype=np.int64)

    provenance: list[dict[str, Any]] = []
    models: list[torch.nn.Module] = []
    validation_logits: dict[str, np.ndarray] = {}
    for component in components_config:
        name = str(component["name"])
        run_dir = Path(str(component["run_dir"])).resolve()
        details, model, calibrated_logits = _load_component(
            name,
            run_dir,
            validation_names,
            validation_targets,
        )
        provenance.append(details)
        models.append(model)
        validation_logits[name] = calibrated_logits
    image_sizes = {int(component["image_size"]) for component in provenance}
    if len(image_sizes) != 1:
        raise RuntimeError("Teacher components use different image sizes")
    image_size = image_sizes.pop()

    weights, selection_results = select_two_component_weights(
        validation_logits,
        validation_targets,
        generalist_name=generalist_name,
        grid_step=grid_step,
    )
    component_set = {
        "target_kind": "validation_selected_weighted_calibrated_logit",
        "components": [
            {
                "name": component["name"],
                "checkpoint_sha256": component["checkpoint_sha256"],
                "calibration_sha256": component["calibration_sha256"],
                "weight": weights[str(component["name"])],
            }
            for component in provenance
        ],
    }
    component_set_sha256 = _canonical_sha256(component_set)

    device = resolve_training_device(
        args.device or str(config["inference"].get("device", "auto"))
    )
    transfer_optimizations = uses_cuda_transfer_optimizations(device)
    models = [model.to(device) for model in models]
    dataset = LesionDataset(
        training_frame,
        args.images_dir,
        image_size,
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=transfer_optimizations,
        persistent_workers=False,
    )
    raw_component_logits = infer_teacher_logits(
        models,
        loader,
        device,
        non_blocking=transfer_optimizations,
        show_progress=bool(config["inference"].get("show_progress", False)),
    )
    calibrated_component_logits = np.empty_like(
        raw_component_logits,
        dtype=np.float64,
    )
    for index, component in enumerate(provenance):
        calibrated_component_logits[:, index] = (
            float(component["logit_scale"]) * raw_component_logits[:, index]
            + float(component["logit_bias"])
        )
    teacher_scores = sum(
        weights[str(component["name"])]
        * calibrated_component_logits[:, index]
        for index, component in enumerate(provenance)
    ).astype(np.float32)
    if not np.isfinite(teacher_scores).all():
        raise RuntimeError("Weighted teacher produced non-finite scores")

    output_dir = create_new_directory(args.output_dir)
    targets_path = output_dir / "teacher_targets.npz"
    write_npz_exclusive(
        targets_path,
        image_names=training_frame["image_name"].to_numpy(dtype=str),
        hard_targets=training_frame["target"].to_numpy(dtype=np.int64),
        teacher_scores=teacher_scores,
        component_names=np.asarray(names, dtype=str),
        component_logits=calibrated_component_logits.astype(np.float32),
    )
    selected_row = next(
        row
        for row in selection_results
        if np.isclose(
            row["generalist_weight"],
            weights[generalist_name],
        )
    )
    receipt = {
        "schema_version": 2,
        "status": "complete",
        "target_split": "train",
        "target_kind": component_set["target_kind"],
        "record_count": int(len(training_frame)),
        "positive_count": int(training_frame["target"].sum()),
        "normalization_split": "validation",
        "normalization_records": int(len(validation_frame)),
        "selection_metric": "pr_auc",
        "selection_grid_step": grid_step,
        "selection_results": selection_results,
        "selected_validation_metrics": {
            "pr_auc": selected_row["pr_auc"],
            "roc_auc": selected_row["roc_auc"],
        },
        "weights": weights,
        "component_order": names,
        "components": provenance,
        "component_value_kind": "affine_calibrated_logit",
        "ensemble_component_set_sha256": component_set_sha256,
        "manifest_sha256": preflight["manifest_sha256"],
        "teacher_targets_sha256": sha256_file(targets_path),
        "image_size": image_size,
        "evaluation_preprocessing": DEPLOYMENT_PREPROCESSING_VERSION,
        "device": str(device),
        "thresholds_validated": False,
        "research_only": True,
    }
    write_json_exclusive(output_dir / "teacher_targets.json", receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
