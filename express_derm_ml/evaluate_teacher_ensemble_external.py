from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from .artifacts import require_absent, write_json_exclusive, write_npz_exclusive
from .build_ensemble_candidate import standardized_logit_average
from .common import sha256_file
from .dataset import DEPLOYMENT_PREPROCESSING_VERSION, LesionDataset
from .device import resolve_training_device, uses_cuda_transfer_optimizations
from .evaluate_external import _verify_external_images
from .generate_teacher_targets import (
    infer_teacher_logits,
    load_teacher_models,
    validate_teacher_ensemble,
)
from .manifest import canonical_manifest_sha256, read_manifest
from .metrics import compute_binary_metrics, sigmoid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen teacher ensemble with deployment preprocessing."
    )
    parser.add_argument("--ensemble-dir", required=True)
    parser.add_argument("--teacher-targets-dir", required=True)
    parser.add_argument("--external-manifest", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Batch size must be positive and workers non-negative")
    output_path = Path(args.output)
    predictions_path = output_path.with_suffix(".npz")
    require_absent([output_path, predictions_path])

    ensemble_dir = Path(args.ensemble_dir).resolve()
    ensemble, components = validate_teacher_ensemble(ensemble_dir)
    receipt_path = Path(args.teacher_targets_dir) / "teacher_targets.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("evaluation_preprocessing") != (
        DEPLOYMENT_PREPROCESSING_VERSION
    ) or receipt.get("normalization_split") != "validation":
        raise RuntimeError("Teacher normalization contract mismatch")
    component_names = [str(component["name"]) for component in components]
    normalization = receipt.get("validation_logit_normalization", {})
    if set(normalization) != set(component_names):
        raise RuntimeError("Teacher normalization components are incomplete")

    manifest_path = Path(args.external_manifest)
    frame = read_manifest(
        manifest_path,
        additional_text_columns=(
            "diagnosis_class",
            "diagnosis_full",
            "diagnosis_confirm_type",
            "patient_identifier_status",
            "image_type",
        ),
    )
    manifest_sha256 = canonical_manifest_sha256(frame)
    if Path(f"{manifest_path}.sha256").read_text().strip() != manifest_sha256:
        raise RuntimeError("External manifest hash mismatch")
    images_root = Path(args.images_dir).resolve()
    _verify_external_images(frame, images_root)

    device = resolve_training_device(args.device)
    transfer_optimizations = uses_cuda_transfer_optimizations(device)
    models, image_size, model_provenance = load_teacher_models(
        ensemble_dir,
        components,
        device,
    )
    dataset_frame = frame.copy()
    dataset_frame["target"] = dataset_frame["target_melanoma"].astype(int)
    dataset = LesionDataset(
        dataset_frame,
        images_root,
        image_size,
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    component_logits = infer_teacher_logits(
        models,
        loader,
        device,
        non_blocking=transfer_optimizations,
        show_progress=False,
    )
    logits_by_name = {
        name: component_logits[:, index].astype(np.float64)
        for index, name in enumerate(component_names)
    }
    ensemble_logits = standardized_logit_average(
        logits_by_name,
        normalization,
        component_names,
    )
    targets = frame["target_melanoma"].to_numpy(dtype=np.int64)
    probabilities = sigmoid(ensemble_logits)
    metrics = compute_binary_metrics(
        targets,
        probabilities,
        threshold=0.5,
        ece_bins=10,
    )
    report = {
        "schema_version": 1,
        "status": "external_research_evaluation",
        "candidate_type": "teacher_ensemble",
        "evaluation_preprocessing": DEPLOYMENT_PREPROCESSING_VERSION,
        "external_manifest_sha256": manifest_sha256,
        "ensemble_json_sha256": sha256_file(ensemble_dir / "ensemble.json"),
        "teacher_targets_receipt_sha256": sha256_file(receipt_path),
        "component_order": component_names,
        "components": model_provenance,
        "melanoma_vs_all": metrics,
        "training_authorized_from_external_set": False,
        "thresholds_validated": False,
        "research_only": True,
        "device": str(device),
    }
    write_npz_exclusive(
        predictions_path,
        component_logits=component_logits,
        ensemble_logits=ensemble_logits,
        probabilities=probabilities,
        targets=targets,
        image_names=frame["image_name"].to_numpy(dtype=str),
    )
    write_json_exclusive(output_path, report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
