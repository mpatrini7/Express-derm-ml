from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .artifacts import copy_file_exclusive, create_new_directory, write_json_exclusive
from .common import load_yaml, set_seed, sha256_file
from .dataset import (
    DEPLOYMENT_PREPROCESSING_VERSION,
    DistillationDataset,
    LesionDataset,
)
from .device import resolve_training_device, uses_cuda_transfer_optimizations
from .manifest import read_manifest
from .metrics import compute_binary_metrics, sigmoid
from .model import create_model
from .preflight import validate_training_input
from .train import (
    EpochBalancedSampler,
    _copy_manifest_bundle,
    evaluate_loader,
    positive_class_weight,
)


def load_distillation_targets(
    targets_path: Path,
    receipt_path: Path,
    training_frame,
    *,
    manifest_sha256: str,
    expected_component_set_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "complete":
        raise RuntimeError("Teacher target receipt is incomplete")
    if receipt.get("target_split") != "train":
        raise RuntimeError("Teacher targets must be train-only")
    if receipt.get("normalization_split") != "validation":
        raise RuntimeError("Teacher normalization must use validation only")
    if receipt.get("evaluation_preprocessing") != (
        DEPLOYMENT_PREPROCESSING_VERSION
    ):
        raise RuntimeError("Teacher preprocessing contract mismatch")
    if receipt.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("Teacher target manifest hash mismatch")
    if (
        receipt.get("ensemble_component_set_sha256")
        != expected_component_set_sha256
    ):
        raise RuntimeError("Unexpected teacher ensemble component set")
    if sha256_file(targets_path) != receipt.get("teacher_targets_sha256"):
        raise RuntimeError("Teacher target artifact hash mismatch")

    with np.load(targets_path, allow_pickle=False) as bundle:
        required = {
            "image_names",
            "hard_targets",
            "teacher_scores",
            "component_names",
            "component_logits",
        }
        if not required.issubset(bundle.files):
            raise RuntimeError("Teacher target artifact is incomplete")
        image_names = bundle["image_names"].astype(str)
        hard_targets = bundle["hard_targets"].astype(np.int64)
        teacher_scores = bundle["teacher_scores"].astype(np.float32)
        component_logits = bundle["component_logits"]
        component_names = bundle["component_names"].astype(str)

    manifest_names = training_frame["image_name"].to_numpy(dtype=str)
    manifest_targets = training_frame["target"].to_numpy(dtype=np.int64)
    if not np.array_equal(image_names, manifest_names):
        raise RuntimeError("Teacher target image order does not match train split")
    if not np.array_equal(hard_targets, manifest_targets):
        raise RuntimeError("Teacher hard targets do not match train split")
    if teacher_scores.shape != (len(training_frame),):
        raise RuntimeError("Teacher score shape does not match train split")
    if component_logits.shape != (len(training_frame), len(component_names)):
        raise RuntimeError("Teacher component logit shape is invalid")
    if not np.isfinite(teacher_scores).all() or not np.isfinite(
        component_logits
    ).all():
        raise RuntimeError("Teacher targets contain non-finite values")
    if int(receipt.get("record_count", -1)) != len(training_frame):
        raise RuntimeError("Teacher target record count mismatch")
    return teacher_scores, receipt


def teacher_run_provenance(receipt: dict[str, Any]) -> dict[str, Any]:
    component_set_sha256 = receipt.get("ensemble_component_set_sha256")
    if not isinstance(component_set_sha256, str) or not component_set_sha256:
        raise RuntimeError("Teacher component-set provenance is unavailable")
    return {
        "teacher_target_kind": receipt.get("target_kind"),
        "teacher_ensemble_json_sha256": receipt.get(
            "ensemble_json_sha256"
        ),
        "teacher_ensemble_component_set_sha256": component_set_sha256,
        "teacher_score_scale": receipt.get("teacher_score_scale"),
        "teacher_source_receipt_sha256": receipt.get(
            "source_teacher_receipt_sha256"
        ),
    }


def distillation_loss(
    student_logits: torch.Tensor,
    hard_targets: torch.Tensor,
    teacher_scores: torch.Tensor,
    hard_criterion: nn.Module,
    *,
    temperature: float,
    hard_weight: float,
    soft_weight: float,
    soft_loss_kind: str = "temperature_bce",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if temperature <= 0.0:
        raise ValueError("Distillation temperature must be positive")
    if hard_weight < 0.0 or soft_weight < 0.0:
        raise ValueError("Distillation weights cannot be negative")
    if not np.isclose(hard_weight + soft_weight, 1.0):
        raise ValueError("Distillation weights must sum to one")
    hard_loss = hard_criterion(student_logits, hard_targets)
    if soft_loss_kind == "temperature_bce":
        teacher_probabilities = torch.sigmoid(teacher_scores / temperature)
        soft_loss = F.binary_cross_entropy_with_logits(
            student_logits / temperature,
            teacher_probabilities,
        ) * (temperature**2)
    elif soft_loss_kind == "logit_mse":
        soft_loss = F.mse_loss(student_logits, teacher_scores)
    else:
        raise ValueError(f"Unsupported distillation soft loss: {soft_loss_kind}")
    total = hard_weight * hard_loss + soft_weight * soft_loss
    return total, hard_loss, soft_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill an immutable ensemble into one edge model.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--teacher-targets-dir", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if config.get("deployment", {}).get("preprocessing") != (
        DEPLOYMENT_PREPROCESSING_VERSION
    ):
        raise ValueError("Config deployment preprocessing contract mismatch")
    set_seed(int(config["seed"]))
    distillation = config["distillation"]
    temperature = float(distillation["temperature"])
    hard_weight = float(distillation["hard_loss_weight"])
    soft_weight = float(distillation["soft_loss_weight"])
    soft_loss_kind = str(
        distillation.get("soft_loss_kind", "temperature_bce")
    )
    if soft_loss_kind not in {"temperature_bce", "logit_mse"}:
        raise ValueError("Unsupported distillation soft loss")
    if temperature <= 0.0 or not np.isclose(hard_weight + soft_weight, 1.0):
        raise ValueError("Invalid distillation temperature or loss weights")

    requested_device = args.device or str(
        config.get("training", {}).get("device", "auto")
    )
    device = resolve_training_device(requested_device)
    transfer_optimizations = uses_cuda_transfer_optimizations(device)
    preflight = validate_training_input(
        manifest_path=args.manifest,
        images_dir=args.images_dir,
        verify_image_hashes=True,
    )
    manifest = read_manifest(args.manifest)
    training_frame = manifest.loc[manifest["split"] == "train"].reset_index(
        drop=True
    )
    validation_frame = manifest.loc[
        manifest["split"] == "validation"
    ].reset_index(drop=True)
    if training_frame.empty or validation_frame.empty:
        raise ValueError("Train and validation splits are required")

    teacher_dir = Path(args.teacher_targets_dir)
    teacher_targets_path = teacher_dir / "teacher_targets.npz"
    teacher_receipt_path = teacher_dir / "teacher_targets.json"
    teacher_scores, teacher_receipt = load_distillation_targets(
        teacher_targets_path,
        teacher_receipt_path,
        training_frame,
        manifest_sha256=preflight["manifest_sha256"],
        expected_component_set_sha256=str(
            distillation["teacher_ensemble_component_set_sha256"]
        ),
    )

    initial_checkpoint_path = Path(args.initial_checkpoint)
    initial_checkpoint_sha256 = sha256_file(initial_checkpoint_path)
    if (
        initial_checkpoint_sha256
        != distillation["initial_checkpoint_sha256"]
    ):
        raise RuntimeError("Initial student checkpoint hash mismatch")
    initial_checkpoint = torch.load(
        initial_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    architecture = str(config["model"]["architecture"])
    image_size = int(config["model"]["image_size"])
    if initial_checkpoint["architecture"] != architecture:
        raise RuntimeError("Initial checkpoint architecture mismatch")
    if int(initial_checkpoint["image_size"]) != image_size:
        raise RuntimeError("Initial checkpoint image size mismatch")

    output_dir = create_new_directory(args.output_dir)
    copy_file_exclusive(args.config, output_dir / "config.yaml")
    _copy_manifest_bundle(Path(args.manifest), output_dir)
    copy_file_exclusive(
        teacher_targets_path,
        output_dir / "teacher_targets.npz",
    )
    copy_file_exclusive(
        teacher_receipt_path,
        output_dir / "teacher_targets.json",
    )
    copy_file_exclusive(
        initial_checkpoint_path,
        output_dir / "initial_student.pt",
    )
    write_json_exclusive(output_dir / "preflight.json", preflight)

    train_dataset = DistillationDataset(
        training_frame,
        args.images_dir,
        image_size,
        training=True,
        teacher_scores=teacher_scores,
    )
    validation_dataset = LesionDataset(
        validation_frame,
        args.images_dir,
        image_size,
        training=False,
    )
    labels = training_frame["target"].astype(int).to_numpy()
    sampling_ratio = config["training"].get(
        "negative_to_positive_ratio_per_epoch"
    )
    priority_collections = config["training"].get(
        "always_include_negative_collections",
        [],
    )
    if not isinstance(priority_collections, list) or any(
        not str(value).strip() for value in priority_collections
    ):
        raise ValueError(
            "always_include_negative_collections must be a list of names"
        )
    priority_source_column = str(
        config["training"].get(
            "priority_negative_source_column",
            "collection_id",
        )
    )
    required_negative_indices: np.ndarray | None = None
    priority_negative_repeats = int(
        config["training"].get("priority_negative_repeats", 1)
    )
    if priority_negative_repeats < 1:
        raise ValueError("priority_negative_repeats must be positive")
    if priority_collections:
        if sampling_ratio is None:
            raise ValueError(
                "Priority negatives require epoch-balanced sampling"
            )
        if priority_source_column not in training_frame.columns:
            raise ValueError(
                "Priority-negative source column is missing: "
                f"{priority_source_column}"
            )
        priority_mask = training_frame[priority_source_column].astype(str).isin(
            {str(value) for value in priority_collections}
        ).to_numpy()
        required_negative_indices = np.flatnonzero(priority_mask)
        if len(required_negative_indices) == 0:
            raise ValueError("No configured priority negatives were found")
        if np.any(labels[required_negative_indices] != 0):
            raise ValueError(
                "Priority-negative collections must contain only negatives"
            )
    train_sampler = (
        EpochBalancedSampler(
            labels,
            negative_to_positive_ratio=int(sampling_ratio),
            seed=int(config["seed"]),
            always_include_negative_indices=required_negative_indices,
            always_include_negative_repeats=priority_negative_repeats,
        )
        if sampling_ratio is not None
        else None
    )
    num_workers = int(config["training"]["num_workers"])
    persistent_workers = bool(
        config["training"].get("persistent_workers", False)
    ) and num_workers > 0
    cache_validation_images = bool(
        config["training"].get("cache_validation_images", False)
    )
    if cache_validation_images:
        print("Caching exact OpenCV validation pixels", flush=True)
        validation_dataset.cache_evaluation_images(workers=num_workers)
        print("Validation pixel cache complete", flush=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=transfer_optimizations,
        persistent_workers=persistent_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0 if cache_validation_images else num_workers,
        pin_memory=transfer_optimizations,
        persistent_workers=(
            False if cache_validation_images else persistent_workers
        ),
    )

    model = create_model(architecture, pretrained=False)
    model.load_state_dict(initial_checkpoint["model_state"])
    model.to(device)
    class_weight_mode = str(
        config["training"].get("positive_class_weight", "sqrt_balanced")
    )
    positive_weight = torch.tensor(
        [positive_class_weight(labels, class_weight_mode)],
        device=device,
        dtype=torch.float32,
    )
    hard_criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler_name = str(config["training"].get("scheduler", "none"))
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["training"]["epochs"]),
            eta_min=float(config["training"].get("minimum_learning_rate", 0.0)),
        )
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported learning-rate scheduler: {scheduler_name}")
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(config["training"]["amp"]) and device.type == "cuda",
    )
    selection_metric = str(config["training"].get("selection_metric", "pr_auc"))
    if selection_metric not in {"roc_auc", "pr_auc"}:
        raise ValueError("Checkpoint selection metric must be roc_auc or pr_auc")

    def save_candidate_checkpoint() -> None:
        temporary_checkpoint = output_dir / "best.pt.tmp"
        torch.save(
            {
                "model_state": model.state_dict(),
                "architecture": architecture,
                "image_size": image_size,
                "config": config,
                "manifest_sha256": preflight["manifest_sha256"],
                "candidate_type": "distilled_student",
                "teacher_ensemble_component_set_sha256": teacher_receipt[
                    "ensemble_component_set_sha256"
                ],
                "teacher_targets_sha256": teacher_receipt[
                    "teacher_targets_sha256"
                ],
            },
            temporary_checkpoint,
        )
        temporary_checkpoint.replace(output_dir / "best.pt")

    initial_logits, initial_targets = evaluate_loader(
        model,
        validation_loader,
        device,
        non_blocking=transfer_optimizations,
    )
    initial_metrics = compute_binary_metrics(
        initial_targets,
        sigmoid(initial_logits),
        ece_bins=int(config["calibration"]["ece_bins"]),
    )
    initial_metrics.update(
        {
            "epoch": 0,
            "checkpoint_role": "initial_student",
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
    )
    print(initial_metrics, flush=True)
    best_metric = float(initial_metrics[selection_metric])
    best_epoch = 0
    stale_epochs = 0
    history = [initial_metrics]
    save_candidate_checkpoint()
    started_at = time.time()
    show_progress = bool(config["training"].get("show_progress", True))
    for epoch in range(int(config["training"]["epochs"])):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        total_sum = 0.0
        hard_sum = 0.0
        soft_sum = 0.0
        progress = tqdm(
            train_loader,
            desc=f"distill epoch {epoch + 1}",
            disable=not show_progress,
        )
        for images, hard_targets, teacher_batch, _ in progress:
            images = images.to(device, non_blocking=transfer_optimizations)
            hard_targets = hard_targets.to(
                device,
                non_blocking=transfer_optimizations,
            )
            teacher_batch = teacher_batch.to(
                device,
                non_blocking=transfer_optimizations,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=scaler.is_enabled(),
            ):
                logits = model(images).flatten()
                loss, hard_loss, soft_loss = distillation_loss(
                    logits,
                    hard_targets,
                    teacher_batch,
                    hard_criterion,
                    temperature=temperature,
                    hard_weight=hard_weight,
                    soft_weight=soft_weight,
                    soft_loss_kind=soft_loss_kind,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = images.size(0)
            total_sum += float(loss.item()) * batch_size
            hard_sum += float(hard_loss.item()) * batch_size
            soft_sum += float(soft_loss.item()) * batch_size

        validation_logits, validation_targets = evaluate_loader(
            model,
            validation_loader,
            device,
            non_blocking=transfer_optimizations,
        )
        metrics = compute_binary_metrics(
            validation_targets,
            sigmoid(validation_logits),
            ece_bins=int(config["calibration"]["ece_bins"]),
        )
        records_per_epoch = len(train_loader.sampler)
        metrics.update(
            {
                "epoch": epoch + 1,
                "train_loss": total_sum / records_per_epoch,
                "train_hard_loss": hard_sum / records_per_epoch,
                "train_soft_loss": soft_sum / records_per_epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        history.append(metrics)
        print(metrics, flush=True)

        selected_value = float(metrics[selection_metric])
        if selected_value > best_metric:
            best_metric = selected_value
            best_epoch = epoch + 1
            stale_epochs = 0
            save_candidate_checkpoint()
        else:
            stale_epochs += 1
        if scheduler is not None:
            scheduler.step()
        if stale_epochs >= int(config["training"]["early_stopping_patience"]):
            break

    write_json_exclusive(output_dir / "history.json", {"epochs": history})
    checkpoint_path = output_dir / "best.pt"
    write_json_exclusive(
        output_dir / "run.json",
        {
            "schema_version": 2,
            "status": "complete",
            "candidate_type": "distilled_student",
            "architecture": architecture,
            "image_size": image_size,
            "selection_metric": selection_metric,
            "best_validation_metric": best_metric,
            "best_epoch": best_epoch,
            "manifest_sha256": preflight["manifest_sha256"],
            "config_sha256": sha256_file(args.config),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "initial_checkpoint_sha256": initial_checkpoint_sha256,
            "teacher_targets_sha256": teacher_receipt[
                "teacher_targets_sha256"
            ],
            "teacher_targets_receipt_sha256": sha256_file(
                output_dir / "teacher_targets.json"
            ),
            **teacher_run_provenance(teacher_receipt),
            "distillation_temperature": temperature,
            "hard_loss_weight": hard_weight,
            "soft_loss_weight": soft_weight,
            "soft_loss_kind": soft_loss_kind,
            "training_seconds": time.time() - started_at,
            "device": str(device),
            "device_requested": requested_device,
            "seed": int(config["seed"]),
            "positive_class_weight": float(positive_weight.item()),
            "positive_class_weight_mode": class_weight_mode,
            "negative_to_positive_ratio_per_epoch": (
                int(sampling_ratio) if sampling_ratio is not None else None
            ),
            "training_records_per_epoch": int(len(train_loader.sampler)),
            "always_include_negative_collections": [
                str(value) for value in priority_collections
            ],
            "priority_negative_source_column": (
                priority_source_column if priority_collections else None
            ),
            "priority_negative_records_per_epoch": (
                int(len(required_negative_indices))
                if required_negative_indices is not None
                else 0
            ),
            "priority_negative_repeats": (
                priority_negative_repeats if priority_collections else 0
            ),
            "priority_negative_presentations_per_epoch": (
                int(len(required_negative_indices))
                * priority_negative_repeats
                if required_negative_indices is not None
                else 0
            ),
            "validation_pixel_cache": (
                "exact_opencv_resized_rgb_uint8_ram"
                if cache_validation_images
                else "disabled"
            ),
            "scheduler": scheduler_name,
            "torch_version": str(torch.__version__),
            "numpy_version": str(np.__version__),
            "validation_status": "research_only",
            "evaluation_preprocessing": (
                DEPLOYMENT_PREPROCESSING_VERSION
            ),
            "thresholds_validated": False,
            "deployment_target": "linux_arm64_tensorrt_cpp",
            "research_only": True,
        },
    )


if __name__ == "__main__":
    main()
