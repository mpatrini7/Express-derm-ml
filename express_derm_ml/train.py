from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from .artifacts import (
    copy_file_exclusive,
    create_new_directory,
    write_json_exclusive,
)
from .common import load_yaml, set_seed, sha256_file
from .dataset import LesionDataset
from .dataset import DEPLOYMENT_PREPROCESSING_VERSION
from .dataset import validate_preprocessing_version
from .device import resolve_training_device, uses_cuda_transfer_optimizations
from .manifest import artifact_path, read_manifest
from .metrics import compute_binary_metrics, sigmoid
from .model import create_model
from .preflight import validate_training_input


def positive_class_weight(labels: np.ndarray, mode: str) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Class weighting requires both binary targets")
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=labels,
    )
    balanced_ratio = float(class_weights[1] / class_weights[0])
    if mode == "none":
        return 1.0
    if mode == "sqrt_balanced":
        return math.sqrt(balanced_ratio)
    if mode == "balanced":
        return balanced_ratio
    raise ValueError(f"Unsupported positive class weighting mode: {mode}")


class EpochBalancedSampler(Sampler[int]):
    def __init__(
        self,
        labels: np.ndarray,
        *,
        negative_to_positive_ratio: int,
        seed: int,
        always_include_negative_indices: np.ndarray | None = None,
        always_include_negative_repeats: int = 1,
        always_include_positive_indices: np.ndarray | None = None,
        always_include_positive_repeats: int = 1,
    ) -> None:
        labels = np.asarray(labels, dtype=np.int64)
        if labels.ndim != 1 or set(np.unique(labels)) != {0, 1}:
            raise ValueError("Balanced sampling requires both binary targets")
        if negative_to_positive_ratio < 1:
            raise ValueError("Negative-to-positive ratio must be positive")
        all_positive_indices = np.flatnonzero(labels == 1)
        all_negative_indices = np.flatnonzero(labels == 0)
        self.negative_count = min(
            len(all_negative_indices),
            negative_to_positive_ratio * len(all_positive_indices),
        )
        required = np.asarray(
            (
                []
                if always_include_negative_indices is None
                else always_include_negative_indices
            ),
            dtype=np.int64,
        )
        if required.ndim != 1 or len(np.unique(required)) != len(required):
            raise ValueError("Required negative indices must be unique and flat")
        if len(required) and (
            required.min() < 0
            or required.max() >= len(labels)
            or np.any(labels[required] != 0)
        ):
            raise ValueError("Required sampler indices must all be negatives")
        if len(required) > self.negative_count:
            raise ValueError(
                "Required negatives exceed the configured epoch capacity"
            )
        if always_include_negative_repeats < 1:
            raise ValueError("Required negative repeats must be positive")
        required_slots = len(required) * always_include_negative_repeats
        if required_slots > self.negative_count:
            raise ValueError(
                "Repeated required negatives exceed the epoch capacity"
            )
        self.required_negative_indices = np.sort(required)
        self.required_negative_repeats = int(always_include_negative_repeats)
        self.negative_indices = np.setdiff1d(
            all_negative_indices,
            self.required_negative_indices,
            assume_unique=True,
        )
        self.sampled_negative_count = (
            self.negative_count - required_slots
        )
        required_positive = np.asarray(
            (
                []
                if always_include_positive_indices is None
                else always_include_positive_indices
            ),
            dtype=np.int64,
        )
        if (
            required_positive.ndim != 1
            or len(np.unique(required_positive)) != len(required_positive)
        ):
            raise ValueError("Required positive indices must be unique and flat")
        if len(required_positive) and (
            required_positive.min() < 0
            or required_positive.max() >= len(labels)
            or np.any(labels[required_positive] != 1)
        ):
            raise ValueError("Required sampler indices must all be positives")
        if always_include_positive_repeats < 1:
            raise ValueError("Required positive repeats must be positive")
        positive_slots = len(required_positive) * always_include_positive_repeats
        if positive_slots > len(all_positive_indices):
            raise ValueError("Repeated required positives exceed the epoch capacity")
        self.required_positive_indices = np.sort(required_positive)
        self.required_positive_repeats = int(always_include_positive_repeats)
        self.positive_indices = np.setdiff1d(
            all_positive_indices,
            self.required_positive_indices,
            assume_unique=True,
        )
        self.sampled_positive_count = len(all_positive_indices) - positive_slots
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        sampled_negatives = rng.choice(
            self.negative_indices,
            size=self.sampled_negative_count,
            replace=False,
        )
        indices = np.concatenate(
            [
                rng.choice(
                    self.positive_indices,
                    size=self.sampled_positive_count,
                    replace=False,
                ),
                np.tile(
                    self.required_positive_indices,
                    self.required_positive_repeats,
                ),
                np.tile(
                    self.required_negative_indices,
                    self.required_negative_repeats,
                ),
                sampled_negatives,
            ]
        )
        rng.shuffle(indices)
        return iter(indices.tolist())

    def __len__(self) -> int:
        priority_positive_slots = (
            len(self.required_positive_indices)
            * self.required_positive_repeats
        )
        return (
            self.sampled_positive_count
            + priority_positive_slots
            + self.negative_count
        )


class SourceClassBalancedSampler(Sampler[int]):
    def __init__(
        self,
        labels: np.ndarray,
        sources: np.ndarray,
        *,
        negative_to_positive_ratio: int,
        seed: int,
        always_include_negative_indices: np.ndarray | None = None,
        always_include_negative_repeats: int = 1,
        always_include_positive_indices: np.ndarray | None = None,
        always_include_positive_repeats: int = 1,
    ) -> None:
        labels = np.asarray(labels, dtype=np.int64)
        sources = np.asarray(sources, dtype=str)
        if labels.ndim != 1 or set(np.unique(labels)) != {0, 1}:
            raise ValueError("Source-balanced sampling requires binary targets")
        if sources.ndim != 1 or len(sources) != len(labels):
            raise ValueError("Sampling sources must align with binary targets")
        if any(not source.strip() for source in sources):
            raise ValueError("Sampling sources cannot be blank")
        if negative_to_positive_ratio < 1:
            raise ValueError("Negative-to-positive ratio must be positive")

        self.source_indices: dict[str, dict[int, np.ndarray]] = {}
        for source in sorted(np.unique(sources)):
            source_mask = sources == source
            by_target = {
                target: np.flatnonzero(source_mask & (labels == target))
                for target in (0, 1)
            }
            if any(len(indices) == 0 for indices in by_target.values()):
                raise ValueError(
                    f"Sampling source {source!r} must contain both targets"
                )
            self.source_indices[source] = by_target

        self.positive_count_per_source = min(
            len(by_target[1]) for by_target in self.source_indices.values()
        )
        self.negative_count_per_source = min(
            negative_to_positive_ratio * self.positive_count_per_source,
            *(len(by_target[0]) for by_target in self.source_indices.values()),
        )
        required = np.asarray(
            (
                []
                if always_include_negative_indices is None
                else always_include_negative_indices
            ),
            dtype=np.int64,
        )
        if required.ndim != 1 or len(np.unique(required)) != len(required):
            raise ValueError("Required negative indices must be unique and flat")
        if len(required) and (
            required.min() < 0
            or required.max() >= len(labels)
            or np.any(labels[required] != 0)
        ):
            raise ValueError("Required sampler indices must all be negatives")
        if always_include_negative_repeats < 1:
            raise ValueError("Required negative repeats must be positive")
        self.required_negative_repeats = int(always_include_negative_repeats)
        self.required_negative_indices_by_source: dict[str, np.ndarray] = {}
        for source, by_target in self.source_indices.items():
            required_for_source = np.intersect1d(
                required,
                by_target[0],
                assume_unique=True,
            )
            required_slots = (
                len(required_for_source) * self.required_negative_repeats
            )
            if required_slots > self.negative_count_per_source:
                raise ValueError(
                    "Repeated required negatives exceed the source epoch capacity: "
                    f"{source}"
                )
            self.required_negative_indices_by_source[source] = required_for_source
            by_target[0] = np.setdiff1d(
                by_target[0], required_for_source, assume_unique=True
            )
        assigned_required = sum(
            len(indices)
            for indices in self.required_negative_indices_by_source.values()
        )
        if assigned_required != len(required):
            raise RuntimeError("Required negatives were not assigned to one source")
        required_positive = np.asarray(
            (
                []
                if always_include_positive_indices is None
                else always_include_positive_indices
            ),
            dtype=np.int64,
        )
        if (
            required_positive.ndim != 1
            or len(np.unique(required_positive)) != len(required_positive)
        ):
            raise ValueError("Required positive indices must be unique and flat")
        if len(required_positive) and (
            required_positive.min() < 0
            or required_positive.max() >= len(labels)
            or np.any(labels[required_positive] != 1)
        ):
            raise ValueError("Required sampler indices must all be positives")
        if always_include_positive_repeats < 1:
            raise ValueError("Required positive repeats must be positive")
        self.required_positive_repeats = int(always_include_positive_repeats)
        self.required_positive_indices_by_source: dict[str, np.ndarray] = {}
        for source, by_target in self.source_indices.items():
            required_for_source = np.intersect1d(
                required_positive,
                by_target[1],
                assume_unique=True,
            )
            required_slots = (
                len(required_for_source) * self.required_positive_repeats
            )
            if required_slots > self.positive_count_per_source:
                raise ValueError(
                    "Repeated required positives exceed the source epoch capacity: "
                    f"{source}"
                )
            self.required_positive_indices_by_source[source] = required_for_source
            by_target[1] = np.setdiff1d(
                by_target[1], required_for_source, assume_unique=True
            )
        assigned_required_positive = sum(
            len(indices)
            for indices in self.required_positive_indices_by_source.values()
        )
        if assigned_required_positive != len(required_positive):
            raise RuntimeError("Required positives were not assigned to one source")
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        selected: list[np.ndarray] = []
        for source, by_target in self.source_indices.items():
            required_positive = self.required_positive_indices_by_source[source]
            sampled_positive_count = self.positive_count_per_source - (
                len(required_positive) * self.required_positive_repeats
            )
            required_negative = self.required_negative_indices_by_source[source]
            sampled_negative_count = self.negative_count_per_source - (
                len(required_negative) * self.required_negative_repeats
            )
            selected.extend(
                [
                    rng.choice(
                        by_target[1],
                        size=sampled_positive_count,
                        replace=False,
                    ),
                    np.tile(
                        required_positive,
                        self.required_positive_repeats,
                    ),
                    rng.choice(
                        by_target[0],
                        size=sampled_negative_count,
                        replace=False,
                    ),
                    np.tile(
                        required_negative,
                        self.required_negative_repeats,
                    ),
                ]
            )
        indices = np.concatenate(selected)
        rng.shuffle(indices)
        return iter(indices.tolist())

    def __len__(self) -> int:
        records_per_source = (
            self.positive_count_per_source + self.negative_count_per_source
        )
        return len(self.source_indices) * records_per_source


def collapse_sampling_sources(
    sources: np.ndarray,
    source_groups: dict[str, list[str]] | None,
) -> np.ndarray:
    sources = np.asarray(sources, dtype=str)
    if source_groups is None:
        return sources
    if not source_groups:
        raise ValueError("Sampling source groups cannot be empty")

    source_to_group: dict[str, str] = {}
    for group, members in source_groups.items():
        group_name = str(group).strip()
        if not group_name or not members:
            raise ValueError("Sampling source groups require names and members")
        for member in members:
            source = str(member).strip()
            if not source:
                raise ValueError("Sampling source group members cannot be blank")
            if source in source_to_group:
                raise ValueError(
                    f"Sampling source belongs to multiple groups: {source}"
                )
            source_to_group[source] = group_name

    unknown = sorted(set(sources) - set(source_to_group))
    if unknown:
        raise ValueError(f"Sampling sources are not grouped: {unknown}")
    return np.asarray([source_to_group[source] for source in sources])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--initial-checkpoint",
        help=(
            "Optional compatible checkpoint for progressive-resolution "
            "training. Architecture and preprocessing must match; image size "
            "may differ."
        ),
    )
    parser.add_argument(
        "--priority-records",
        help=(
            "Optional OOF hard-error CSV. Every record must match a training "
            "manifest row by image name, hash, target, group, and collection."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default=None,
        help=(
            "Training device. Defaults to training.device from the config, "
            "then CUDA, Apple Metal, or CPU."
        ),
    )
    return parser.parse_args()


def load_priority_record_indices(
    priority_records_path: str | Path,
    training_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    path = Path(priority_records_path)
    digest_path = Path(f"{path}.sha256")
    report_path = artifact_path(path, "report.json")
    if not path.is_file() or not digest_path.is_file() or not report_path.is_file():
        raise ValueError(
            "Priority records require CSV, .sha256, and .report.json artifacts"
        )
    digest = sha256_file(path)
    if digest_path.read_text(encoding="ascii").strip() != digest:
        raise ValueError("Priority-record digest mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_report = {
        "purpose": "out_of_fold_hard_error_mining",
        "evaluation_role": "out_of_fold_training",
        "output_sha256": digest,
        "training_authorized": True,
        "test_or_external_records_used": 0,
    }
    for field, expected in expected_report.items():
        if report.get(field) != expected:
            raise ValueError(f"Priority-record report mismatch: {field}")

    priority = pd.read_csv(path, dtype=str)
    required_columns = {
        "image_name",
        "sha256",
        "group_id",
        "collection_id",
        "target",
        "error_type",
    }
    missing = required_columns - set(priority.columns)
    if missing:
        raise ValueError(f"Priority records are missing columns: {sorted(missing)}")
    if priority.empty or priority["image_name"].duplicated().any():
        raise ValueError("Priority records must be non-empty and unique by image")
    allowed_errors = {"hard_false_positive", "hard_false_negative"}
    if set(priority["error_type"].astype(str)) - allowed_errors:
        raise ValueError("Priority records contain unsupported error types")

    training = training_frame.reset_index(drop=True).copy()
    if training["image_name"].astype(str).duplicated().any():
        raise ValueError("Training manifest image names must be unique")
    lookup = training.reset_index(names="training_index")
    aligned = priority.merge(
        lookup[
            [
                "training_index",
                "image_name",
                "sha256",
                "group_id",
                "collection_id",
                "target",
            ]
        ],
        on="image_name",
        how="left",
        suffixes=("_priority", "_manifest"),
        validate="one_to_one",
    )
    if aligned["training_index"].isna().any():
        raise ValueError("Every priority record must belong to the training split")
    for field in ("sha256", "group_id", "collection_id", "target"):
        if not aligned[f"{field}_priority"].astype(str).eq(
            aligned[f"{field}_manifest"].astype(str)
        ).all():
            raise ValueError(f"Priority records do not match manifest field: {field}")
    expected_target = aligned["error_type"].map(
        {"hard_false_positive": "0", "hard_false_negative": "1"}
    )
    if not aligned["target_priority"].astype(str).eq(expected_target).all():
        raise ValueError("Priority error type does not match target")

    indices = aligned["training_index"].astype(int).to_numpy()
    targets = aligned["target_priority"].astype(int).to_numpy()
    evidence: dict[str, object] = {
        "sha256": digest,
        "report_sha256": sha256_file(report_path),
        "records": int(len(aligned)),
        "hard_false_positive_records": int(np.sum(targets == 0)),
        "hard_false_negative_records": int(np.sum(targets == 1)),
    }
    return indices[targets == 0], indices[targets == 1], evidence


@torch.inference_mode()
def evaluate_loader(model, loader, device, *, non_blocking: bool = False):
    model.eval()
    logits, targets = [], []
    for images, labels, _ in loader:
        output = model(
            images.to(device, non_blocking=non_blocking)
        ).flatten()
        logits.extend(output.cpu().numpy().tolist())
        targets.extend(labels.numpy().tolist())
    logits_array = np.asarray(logits, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.int64)
    return logits_array, target_array


def _copy_manifest_bundle(manifest_path: Path, output_dir: Path) -> None:
    sources = {
        manifest_path: output_dir / "manifest.csv",
        Path(f"{manifest_path}.sha256"): output_dir / "manifest.csv.sha256",
        artifact_path(manifest_path, "report.json"): (
            output_dir / "manifest.report.json"
        ),
        artifact_path(manifest_path, "near_duplicates.csv"): (
            output_dir / "manifest.near_duplicates.csv"
        ),
    }
    for source, destination in sources.items():
        copy_file_exclusive(source, destination)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    preprocessing = validate_preprocessing_version(
        str(
            config.get("deployment", {}).get(
                "preprocessing",
                DEPLOYMENT_PREPROCESSING_VERSION,
            )
        )
    )
    set_seed(int(config["seed"]))

    requested_device = args.device or str(
        config.get("training", {}).get("device", "auto")
    )
    device = resolve_training_device(requested_device)
    cuda_transfer_optimizations = uses_cuda_transfer_optimizations(device)

    preflight = validate_training_input(
        manifest_path=args.manifest,
        images_dir=args.images_dir,
        verify_image_hashes=True,
    )
    output_dir = create_new_directory(args.output_dir)
    copy_file_exclusive(args.config, output_dir / "config.yaml")
    _copy_manifest_bundle(Path(args.manifest), output_dir)
    write_json_exclusive(output_dir / "preflight.json", preflight)
    manifest = read_manifest(output_dir / "manifest.csv")

    image_size = int(config["model"]["image_size"])
    center_crop_scales = tuple(
        float(value)
        for value in config.get("augmentation", {}).get(
            "center_crop_scales",
            [1.0],
        )
    )
    training_frame = manifest.loc[manifest["split"] == "train"].copy()
    validation_frame = manifest.loc[manifest["split"] == "validation"].copy()
    if training_frame.empty or validation_frame.empty:
        raise ValueError("Train and validation splits are required")

    train_dataset = LesionDataset(
        training_frame,
        args.images_dir,
        image_size,
        training=True,
        preprocessing=preprocessing,
        center_crop_scales=center_crop_scales,
    )
    validation_dataset = LesionDataset(
        validation_frame,
        args.images_dir,
        image_size,
        training=False,
        preprocessing=preprocessing,
    )
    num_workers = int(config["training"]["num_workers"])
    persistent_workers = bool(
        config["training"].get("persistent_workers", False)
    ) and num_workers > 0
    cache_validation_images = bool(
        config["training"].get("cache_validation_images", False)
    )
    if cache_validation_images:
        if num_workers <= 0:
            raise ValueError(
                "Validation pixel caching requires at least one worker"
            )
        print("Caching exact OpenCV validation pixels", flush=True)
        validation_dataset.cache_evaluation_images(workers=num_workers)
        print("Validation pixel cache complete", flush=True)
    labels = training_frame["target"].astype(int).to_numpy()
    sampling_ratio = config["training"].get(
        "negative_to_positive_ratio_per_epoch"
    )
    source_sampling = config["training"].get("source_balanced_sampling")
    if sampling_ratio is not None and source_sampling is not None:
        raise ValueError(
            "Configure either epoch-balanced or source-balanced sampling"
        )
    priority_collections = config["training"].get(
        "always_include_negative_collections", []
    )
    if not isinstance(priority_collections, list) or any(
        not str(value).strip() for value in priority_collections
    ):
        raise ValueError(
            "always_include_negative_collections must be a list of names"
        )
    priority_source_column = str(
        config["training"].get(
            "priority_negative_source_column", "collection_id"
        )
    )
    priority_negative_repeats = int(
        config["training"].get("priority_negative_repeats", 1)
    )
    if priority_negative_repeats < 1:
        raise ValueError("priority_negative_repeats must be positive")
    priority_positive_repeats = int(
        config["training"].get("priority_positive_repeats", 1)
    )
    if priority_positive_repeats < 1:
        raise ValueError("priority_positive_repeats must be positive")
    required_negative_indices: np.ndarray | None = None
    required_positive_indices: np.ndarray | None = None
    priority_records_evidence: dict[str, object] | None = None
    if priority_collections:
        if sampling_ratio is None and source_sampling is None:
            raise ValueError("Priority negatives require a balanced sampler")
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
    if args.priority_records:
        if sampling_ratio is None and source_sampling is None:
            raise ValueError("Priority records require a balanced sampler")
        hard_negative_indices, hard_positive_indices, priority_records_evidence = (
            load_priority_record_indices(args.priority_records, training_frame)
        )
        required_negative_indices = np.unique(
            np.concatenate(
                [
                    np.asarray(
                        []
                        if required_negative_indices is None
                        else required_negative_indices,
                        dtype=np.int64,
                    ),
                    hard_negative_indices,
                ]
            )
        )
        required_positive_indices = np.unique(hard_positive_indices)
        copy_file_exclusive(
            args.priority_records, output_dir / "priority_records.csv"
        )
        copy_file_exclusive(
            Path(f"{args.priority_records}.sha256"),
            output_dir / "priority_records.csv.sha256",
        )
        copy_file_exclusive(
            artifact_path(args.priority_records, "report.json"),
            output_dir / "priority_records.report.json",
        )
    if source_sampling is not None:
        source_column = str(source_sampling["source_column"])
        if source_column not in training_frame.columns:
            raise ValueError(
                f"Sampling source column is missing: {source_column}"
            )
        sources = collapse_sampling_sources(
            training_frame[source_column].fillna("").to_numpy(),
            source_sampling.get("source_groups"),
        )
        train_sampler = SourceClassBalancedSampler(
            labels,
            sources,
            negative_to_positive_ratio=int(
                source_sampling["negative_to_positive_ratio_per_source"]
            ),
            seed=int(config["seed"]),
            always_include_negative_indices=required_negative_indices,
            always_include_negative_repeats=priority_negative_repeats,
            always_include_positive_indices=required_positive_indices,
            always_include_positive_repeats=priority_positive_repeats,
        )
    elif sampling_ratio is not None:
        train_sampler = EpochBalancedSampler(
            labels,
            negative_to_positive_ratio=int(sampling_ratio),
            seed=int(config["seed"]),
            always_include_negative_indices=required_negative_indices,
            always_include_negative_repeats=priority_negative_repeats,
            always_include_positive_indices=required_positive_indices,
            always_include_positive_repeats=priority_positive_repeats,
        )
    else:
        train_sampler = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=cuda_transfer_optimizations,
        persistent_workers=persistent_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0 if cache_validation_images else num_workers,
        pin_memory=cuda_transfer_optimizations,
        persistent_workers=(
            False if cache_validation_images else persistent_workers
        ),
    )

    initial_checkpoint_sha256: str | None = None
    initial_checkpoint_image_size: int | None = None
    initial_checkpoint: dict[str, object] | None = None
    if args.initial_checkpoint:
        initial_checkpoint_path = Path(args.initial_checkpoint)
        initial_checkpoint = torch.load(
            initial_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if initial_checkpoint.get("architecture") != config["model"][
            "architecture"
        ]:
            raise ValueError("Initial checkpoint architecture mismatch")
        initial_preprocessing = validate_preprocessing_version(
            str(
                initial_checkpoint.get(
                    "preprocessing",
                    DEPLOYMENT_PREPROCESSING_VERSION,
                )
            )
        )
        if initial_preprocessing != preprocessing:
            raise ValueError("Initial checkpoint preprocessing mismatch")
        initial_checkpoint_sha256 = sha256_file(initial_checkpoint_path)
        initial_checkpoint_image_size = int(initial_checkpoint["image_size"])

    model = create_model(
        config["model"]["architecture"],
        bool(config["model"]["pretrained"]) and initial_checkpoint is None,
    )
    if initial_checkpoint is not None:
        model.load_state_dict(initial_checkpoint["model_state"])
    model.to(device)

    class_weight_mode = str(
        config["training"].get("positive_class_weight", "balanced")
    )
    positive_weight = torch.tensor(
        [positive_class_weight(labels, class_weight_mode)],
        device=device,
        dtype=torch.float32,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
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

    selection_metric = str(
        config["training"].get("selection_metric", "roc_auc")
    )
    if selection_metric not in {"roc_auc", "pr_auc"}:
        raise ValueError("Checkpoint selection metric must be roc_auc or pr_auc")
    best_metric = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    started_at = time.time()
    show_progress = bool(config["training"].get("show_progress", True))
    gradient_accumulation_steps = int(
        config["training"].get("gradient_accumulation_steps", 1)
    )
    if gradient_accumulation_steps <= 0:
        raise ValueError("Gradient accumulation steps must be positive")

    for epoch in range(int(config["training"]["epochs"])):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}",
            disable=not show_progress,
        )
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (images, targets, _) in enumerate(progress):
            images = images.to(
                device,
                non_blocking=cuda_transfer_optimizations,
            )
            targets = targets.to(
                device,
                non_blocking=cuda_transfer_optimizations,
            )
            with torch.autocast(
                device_type=device.type,
                enabled=scaler.is_enabled(),
            ):
                logits = model(images).flatten()
                unscaled_loss = criterion(logits, targets)
                loss = unscaled_loss / gradient_accumulation_steps

            scaler.scale(loss).backward()
            accumulation_boundary = (
                (batch_index + 1) % gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if accumulation_boundary:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(unscaled_loss.item()) * images.size(0)
            if show_progress:
                progress.set_postfix(loss=float(unscaled_loss.item()))

        validation_logits, validation_targets = evaluate_loader(
            model,
            validation_loader,
            device,
            non_blocking=cuda_transfer_optimizations,
        )
        validation_probabilities = sigmoid(validation_logits)
        metrics = compute_binary_metrics(
            validation_targets,
            validation_probabilities,
            ece_bins=int(config["calibration"]["ece_bins"]),
        )
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = running_loss / len(train_loader.sampler)
        metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        history.append(metrics)
        print(metrics, flush=True)

        selected_value = float(metrics[selection_metric])
        if selected_value > best_metric:
            best_metric = selected_value
            best_epoch = epoch + 1
            stale_epochs = 0
            temporary_checkpoint = output_dir / "best.pt.tmp"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "architecture": config["model"]["architecture"],
                    "image_size": image_size,
                    "preprocessing": preprocessing,
                    "config": config,
                    "manifest_sha256": preflight["manifest_sha256"],
                },
                temporary_checkpoint,
            )
            temporary_checkpoint.replace(output_dir / "best.pt")
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
            "schema_version": 1,
            "status": "complete",
            "architecture": config["model"]["architecture"],
            "image_size": image_size,
            "preprocessing": preprocessing,
            "selection_metric": selection_metric,
            "best_validation_metric": best_metric,
            "best_epoch": best_epoch,
            "manifest_sha256": preflight["manifest_sha256"],
            "config_sha256": sha256_file(args.config),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "training_seconds": time.time() - started_at,
            "device": str(device),
            "device_requested": requested_device,
            "seed": int(config["seed"]),
            "positive_class_weight": float(positive_weight.item()),
            "positive_class_weight_mode": class_weight_mode,
            "negative_to_positive_ratio_per_epoch": (
                int(sampling_ratio) if sampling_ratio is not None else None
            ),
            "source_balanced_sampling": (
                {
                    "source_column": str(source_sampling["source_column"]),
                    "negative_to_positive_ratio_per_source": int(
                        source_sampling[
                            "negative_to_positive_ratio_per_source"
                        ]
                    ),
                    "positive_records_per_source": int(
                        train_sampler.positive_count_per_source
                    ),
                    "negative_records_per_source": int(
                        train_sampler.negative_count_per_source
                    ),
                    "source_count": len(train_sampler.source_indices),
                    "source_groups": source_sampling.get("source_groups"),
                }
                if source_sampling is not None
                else None
            ),
            "always_include_negative_collections": [
                str(value) for value in priority_collections
            ],
            "priority_negative_source_column": priority_source_column,
            "priority_negative_repeats": priority_negative_repeats,
            "priority_positive_repeats": priority_positive_repeats,
            "priority_negative_records": int(
                0
                if required_negative_indices is None
                else len(required_negative_indices)
            ),
            "priority_positive_records": int(
                0
                if required_positive_indices is None
                else len(required_positive_indices)
            ),
            "priority_records": priority_records_evidence,
            "training_records_per_epoch": int(len(train_loader.sampler)),
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "training_center_crop_scales": list(center_crop_scales),
            "effective_batch_size": int(
                config["training"]["batch_size"]
            )
            * gradient_accumulation_steps,
            "initial_checkpoint_sha256": initial_checkpoint_sha256,
            "initial_checkpoint_image_size": initial_checkpoint_image_size,
            "validation_pixel_cache": (
                "exact_opencv_resized_rgb_uint8_ram"
                if cache_validation_images
                else "disabled"
            ),
            "scheduler": scheduler_name,
            "torch_version": str(torch.__version__),
            "numpy_version": str(np.__version__),
            "validation_status": "research_only",
            "evaluation_preprocessing": preprocessing,
            "thresholds_validated": False,
            "research_only": True,
        },
    )


if __name__ == "__main__":
    main()
