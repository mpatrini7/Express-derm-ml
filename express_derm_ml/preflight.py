from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .manifest import (
    ManifestValidationError,
    artifact_path,
    canonical_manifest_sha256,
    read_manifest,
    sha256_file,
)
from .path_safety import resolve_manifest_image_path


def _resolve_image(images_root: Path, raw_path: str) -> Path:
    try:
        return resolve_manifest_image_path(images_root, raw_path)
    except ValueError as error:
        raise ManifestValidationError(
            f"Manifest image path escapes the image root: {raw_path}"
        ) from error


def validate_training_input(
    *,
    manifest_path: str | Path,
    images_dir: str | Path,
    verify_image_hashes: bool,
    image_hash_splits: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    manifest = Path(manifest_path)
    images_root = Path(images_dir).resolve()
    if not images_root.is_dir():
        raise ManifestValidationError(
            f"Images directory does not exist: {images_root}"
        )

    frame = read_manifest(manifest)
    required = {
        "image_name",
        "image_path",
        "target",
        "split",
        "group_id",
        "sha256",
        "perceptual_hash",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ManifestValidationError(
            f"Training manifest is missing columns: {sorted(missing)}"
        )
    if frame.empty:
        raise ManifestValidationError("Training manifest is empty")

    manifest_sha256 = canonical_manifest_sha256(frame)
    digest_path = Path(f"{manifest}.sha256")
    if not digest_path.is_file():
        raise ManifestValidationError(
            f"Manifest digest file is missing: {digest_path}"
        )
    recorded_digest = digest_path.read_text(encoding="ascii").strip()
    if recorded_digest != manifest_sha256:
        raise ManifestValidationError(
            "Manifest digest does not match the canonical manifest"
        )

    report_path = artifact_path(manifest, "report.json")
    if not report_path.is_file():
        raise ManifestValidationError(
            f"Split report is missing: {report_path}"
        )
    split_report = json.loads(report_path.read_text(encoding="utf-8"))
    if split_report.get("split_manifest_sha256") != manifest_sha256:
        raise ManifestValidationError(
            "Split report does not match the training manifest"
        )

    expected_splits = ("train", "validation", "test")
    if image_hash_splits is not None:
        unknown_hash_splits = set(image_hash_splits) - set(expected_splits)
        if unknown_hash_splits:
            raise ManifestValidationError(
                "Unknown image-hash verification splits: "
                f"{sorted(unknown_hash_splits)}"
            )
    actual_splits = set(frame["split"].dropna().astype(str))
    if actual_splits != set(expected_splits):
        raise ManifestValidationError(
            f"Expected train/validation/test splits; found {sorted(actual_splits)}"
        )
    targets = pd.to_numeric(frame["target"], errors="coerce")
    if targets.isna().any() or not targets.isin([0, 1]).all():
        raise ManifestValidationError("Training targets must be binary 0 or 1")
    frame["target"] = targets.astype(int)
    split_target_counts = frame.groupby("split")["target"].nunique()
    if not split_target_counts.reindex(expected_splits).eq(2).all():
        raise ManifestValidationError(
            "Every training split must contain both binary targets"
        )
    if (frame.groupby("group_id")["split"].nunique() > 1).any():
        raise ManifestValidationError("Patient-group split leakage detected")
    if frame["image_name"].duplicated().any():
        raise ManifestValidationError("Duplicate image names detected")
    if frame["sha256"].duplicated().any():
        raise ManifestValidationError(
            "Exact duplicate images remain in the training manifest"
        )
    for report_field in (
        "group_leakage_records",
        "hash_leakage_records",
        "near_duplicate_split_leakage_records",
    ):
        if int(split_report.get(report_field, -1)) != 0:
            raise ManifestValidationError(
                f"Split report has unresolved leakage: {report_field}"
            )

    near_duplicate_path = artifact_path(manifest, "near_duplicates.csv")
    if not near_duplicate_path.is_file():
        raise ManifestValidationError(
            f"Near-duplicate report is missing: {near_duplicate_path}"
        )
    near_duplicates = pd.read_csv(near_duplicate_path)
    if not near_duplicates.empty:
        required_near_columns = {
            "left_split",
            "right_split",
            "same_group",
        }
        missing_near_columns = required_near_columns - set(
            near_duplicates.columns
        )
        if missing_near_columns:
            raise ManifestValidationError(
                "Near-duplicate report is missing split evidence"
            )
        if (
            near_duplicates["left_split"]
            != near_duplicates["right_split"]
        ).any():
            raise ManifestValidationError(
                "Perceptual near-duplicate split leakage detected"
            )
        same_group = (
            near_duplicates["same_group"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("true")
        )
        cross_group_candidates = int((~same_group).sum())
        reported_cross_group = split_report.get(
            "cross_group_near_duplicate_candidates"
        )
        if (
            reported_cross_group is not None
            and int(reported_cross_group) != cross_group_candidates
        ):
            raise ManifestValidationError(
                "Split report cross-group candidate count does not match"
            )
    else:
        cross_group_candidates = 0

    verified_images = 0
    total_rows = len(frame)
    for position, row in enumerate(frame.itertuples(index=False), start=1):
        image_path = _resolve_image(images_root, str(row.image_path))
        if not image_path.is_file():
            raise ManifestValidationError(
                f"Manifest image is missing: {row.image_path}"
            )
        should_verify_hash = (
            verify_image_hashes
            and (
                image_hash_splits is None
                or str(row.split) in image_hash_splits
            )
        )
        if should_verify_hash:
            actual_sha256 = sha256_file(image_path)
            if actual_sha256 != str(row.sha256):
                raise ManifestValidationError(
                    f"Image SHA-256 mismatch: {row.image_name}"
                )
            verified_images += 1
            if verified_images == 1 or verified_images % 1000 == 0:
                print(
                    "Verified image SHA-256 "
                    f"{verified_images:,}/{total_rows:,}",
                    flush=True,
                )

    return {
        "schema_version": 1,
        "manifest_sha256": manifest_sha256,
        "records": int(len(frame)),
        "groups": int(frame["group_id"].nunique()),
        "split_counts": {
            str(split): int(count)
            for split, count in frame["split"].value_counts().sort_index().items()
        },
        "near_duplicate_pairs": int(len(near_duplicates)),
        "cross_group_near_duplicate_candidates": cross_group_candidates,
        "image_hashes_verified": bool(verify_image_hashes),
        "image_hash_splits": (
            list(image_hash_splits)
            if image_hash_splits is not None
            else None
        ),
        "verified_images": verified_images,
    }
