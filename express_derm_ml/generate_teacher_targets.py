from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .artifacts import create_new_directory, write_json_exclusive, write_npz_exclusive
from .build_ensemble_candidate import standardized_logit_average
from .common import sha256_file
from .dataset import DEPLOYMENT_PREPROCESSING_VERSION, LesionDataset
from .device import resolve_training_device, uses_cuda_transfer_optimizations
from .manifest import read_manifest
from .model import create_model
from .preflight import validate_training_input


ENSEMBLE_METHOD = "equal_weight_validation_standardized_logit_average"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_component_run(ensemble_dir: Path, value: str) -> Path:
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [
        Path.cwd() / raw,
        ensemble_dir.parent.parent / raw,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise RuntimeError(f"Ensemble component run is unavailable: {value}")


def validate_teacher_ensemble(
    ensemble_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ensemble_path = ensemble_dir / "ensemble.json"
    ensemble = _read_json(ensemble_path)
    if ensemble.get("status") != "complete":
        raise RuntimeError("Teacher ensemble is not complete")
    if ensemble.get("method") != ENSEMBLE_METHOD:
        raise RuntimeError("Unsupported teacher ensemble method")
    if ensemble.get("candidate_type") != "ensemble":
        raise RuntimeError("Teacher artifact is not an ensemble")
    if ensemble.get("thresholds_validated") is not False:
        raise RuntimeError("Teacher thresholds must remain unvalidated")

    components = ensemble.get("components")
    if not isinstance(components, list) or len(components) < 2:
        raise RuntimeError("Teacher ensemble components are incomplete")
    names = [str(component["name"]) for component in components]
    if len(names) != len(set(names)):
        raise RuntimeError("Teacher ensemble component names are not unique")
    normalization = ensemble.get("validation_logit_normalization", {})
    weights = ensemble.get("weights", {})
    if set(normalization) != set(names) or set(weights) != set(names):
        raise RuntimeError("Teacher normalization or weights are incomplete")
    expected_weight = 1.0 / len(names)
    if any(
        not np.isclose(float(weights[name]), expected_weight)
        for name in names
    ):
        raise RuntimeError("Teacher target generation requires equal weights")

    for filename, expected_hash in ensemble.get("artifact_sha256", {}).items():
        artifact = ensemble_dir / filename
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            raise RuntimeError(f"Teacher ensemble artifact mismatch: {filename}")
    return ensemble, components


def load_teacher_models(
    ensemble_dir: Path,
    components: list[dict[str, Any]],
    device: torch.device,
) -> tuple[list[torch.nn.Module], int, list[dict[str, str]]]:
    models: list[torch.nn.Module] = []
    image_size: int | None = None
    provenance = []
    for component in components:
        name = str(component["name"])
        run_dir = _resolve_component_run(
            ensemble_dir,
            str(component["run_dir"]),
        )
        checkpoint_path = run_dir / "best.pt"
        checkpoint_hash = sha256_file(checkpoint_path)
        if checkpoint_hash != component.get("checkpoint_sha256"):
            raise RuntimeError(f"Teacher checkpoint hash mismatch: {name}")
        run = _read_json(run_dir / "run.json")
        if run.get("status") != "complete":
            raise RuntimeError(f"Teacher component is incomplete: {name}")
        if run.get("checkpoint_sha256") != checkpoint_hash:
            raise RuntimeError(f"Teacher run checkpoint mismatch: {name}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        component_size = int(checkpoint["image_size"])
        if image_size is None:
            image_size = component_size
        elif component_size != image_size:
            raise RuntimeError("Teacher components use different image sizes")
        model = create_model(checkpoint["architecture"], pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        models.append(model.eval().to(device))
        provenance.append(
            {
                "name": name,
                "checkpoint_sha256": checkpoint_hash,
                "run_sha256": sha256_file(run_dir / "run.json"),
            }
        )
    if image_size is None:
        raise RuntimeError("Teacher ensemble has no loadable components")
    return models, image_size, provenance


@torch.inference_mode()
def infer_teacher_logits(
    models: list[torch.nn.Module],
    loader: DataLoader,
    device: torch.device,
    *,
    non_blocking: bool,
    show_progress: bool,
) -> np.ndarray:
    component_batches: list[list[np.ndarray]] = [[] for _ in models]
    progress = tqdm(loader, desc="teacher targets", disable=not show_progress)
    for images, _, _ in progress:
        images = images.to(device, non_blocking=non_blocking)
        for index, model in enumerate(models):
            values = model(images).flatten().cpu().numpy().astype(np.float32)
            component_batches[index].append(values)
    return np.stack(
        [np.concatenate(batches) for batches in component_batches],
        axis=1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze train-only targets from an immutable ensemble.",
    )
    parser.add_argument("--ensemble-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Batch size must be positive and workers non-negative")
    ensemble_dir = Path(args.ensemble_dir).resolve()
    ensemble, components = validate_teacher_ensemble(ensemble_dir)
    preflight = validate_training_input(
        manifest_path=args.manifest,
        images_dir=args.images_dir,
        verify_image_hashes=True,
        image_hash_splits=("train", "validation"),
    )
    frame = read_manifest(args.manifest)
    training_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    validation_frame = frame.loc[
        frame["split"] == "validation"
    ].reset_index(drop=True)
    if training_frame.empty or training_frame["image_name"].duplicated().any():
        raise RuntimeError("Teacher target manifest train split is invalid")
    if validation_frame.empty:
        raise RuntimeError("Teacher normalization requires validation records")

    device = resolve_training_device(args.device)
    transfer_optimizations = uses_cuda_transfer_optimizations(device)
    models, image_size, model_provenance = load_teacher_models(
        ensemble_dir,
        components,
        device,
    )
    validation_dataset = LesionDataset(
        validation_frame,
        args.images_dir,
        image_size,
        training=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=transfer_optimizations,
        persistent_workers=args.num_workers > 0,
    )
    validation_component_logits = infer_teacher_logits(
        models,
        validation_loader,
        device,
        non_blocking=transfer_optimizations,
        show_progress=args.show_progress,
    )
    del validation_loader, validation_dataset
    component_names = [str(component["name"]) for component in components]
    validation_normalization = {}
    for index, name in enumerate(component_names):
        values = validation_component_logits[:, index].astype(np.float64)
        standard_deviation = float(values.std())
        if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
            raise RuntimeError(
                f"Teacher validation logits have invalid spread: {name}"
            )
        validation_normalization[name] = {
            "mean": float(values.mean()),
            "standard_deviation": standard_deviation,
        }

    dataset = LesionDataset(
        training_frame,
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
        show_progress=args.show_progress,
    )
    logits_by_name = {
        name: component_logits[:, index].astype(np.float64)
        for index, name in enumerate(component_names)
    }
    teacher_scores = standardized_logit_average(
        logits_by_name,
        validation_normalization,
        component_names,
    ).astype(np.float32)
    if not np.isfinite(teacher_scores).all():
        raise RuntimeError("Teacher ensemble produced non-finite scores")

    output_dir = create_new_directory(args.output_dir)
    targets_path = output_dir / "teacher_targets.npz"
    write_npz_exclusive(
        targets_path,
        image_names=training_frame["image_name"].to_numpy(dtype=str),
        hard_targets=training_frame["target"].to_numpy(dtype=np.int64),
        teacher_scores=teacher_scores,
        component_names=np.asarray(component_names, dtype=str),
        component_logits=component_logits,
    )
    receipt = {
        "schema_version": 2,
        "status": "complete",
        "target_split": "train",
        "target_kind": "validation_standardized_ensemble_logit",
        "record_count": int(len(training_frame)),
        "normalization_records": int(len(validation_frame)),
        "normalization_split": "validation",
        "validation_logit_normalization": validation_normalization,
        "evaluation_preprocessing": DEPLOYMENT_PREPROCESSING_VERSION,
        "positive_count": int(training_frame["target"].sum()),
        "image_size": image_size,
        "manifest_sha256": preflight["manifest_sha256"],
        "ensemble_json_sha256": sha256_file(ensemble_dir / "ensemble.json"),
        "ensemble_component_set_sha256": ensemble[
            "ensemble_component_set_sha256"
        ],
        "teacher_targets_sha256": sha256_file(targets_path),
        "component_order": component_names,
        "components": model_provenance,
        "device": str(device),
        "thresholds_validated": False,
        "research_only": True,
    }
    write_json_exclusive(output_dir / "teacher_targets.json", receipt)
    print(receipt)


if __name__ == "__main__":
    main()
