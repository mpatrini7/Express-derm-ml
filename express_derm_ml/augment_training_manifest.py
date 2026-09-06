from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from .artifacts import require_absent, write_json_exclusive
from .integrity import NEAR_DUPLICATE_COLUMNS, find_near_duplicate_pairs
from .manifest import (
    ManifestValidationError,
    artifact_path,
    canonical_manifest_sha256,
    read_manifest,
    write_manifest,
)


def _verified_manifest(path: Path) -> pd.DataFrame:
    frame = read_manifest(path)
    digest_path = Path(f"{path}.sha256")
    if not digest_path.is_file():
        raise ManifestValidationError(f"Manifest digest is missing: {path}")
    expected = digest_path.read_text(encoding="ascii").strip()
    if canonical_manifest_sha256(frame) != expected:
        raise ManifestValidationError(f"Manifest digest mismatch: {path}")
    return frame


def _prefix_paths(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    normalized_prefix = PurePosixPath(prefix)
    if (
        not prefix.strip()
        or normalized_prefix.is_absolute()
        or ".." in normalized_prefix.parts
    ):
        raise ManifestValidationError(f"Invalid image prefix: {prefix}")
    result = frame.copy()
    prefixed: list[str] = []
    for value in result["image_path"]:
        relative = PurePosixPath(str(value))
        if relative.is_absolute() or ".." in relative.parts:
            raise ManifestValidationError(
                f"Invalid source image path: {value}"
            )
        prefixed.append((normalized_prefix / relative).as_posix())
    result["image_path"] = prefixed
    return result


def _split_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        split: {
            "images": int(len(split_frame)),
            "groups": int(split_frame["group_id"].nunique()),
            "target_counts": {
                str(int(target)): int(count)
                for target, count in split_frame["target"]
                .value_counts()
                .sort_index()
                .items()
            },
        }
        for split in ("train", "validation", "test")
        for split_frame in [frame.loc[frame["split"].eq(split)]]
    }


def _known_identifiers(frame: pd.DataFrame, column: str) -> set[str]:
    return {
        str(value).strip()
        for value in frame[column].dropna()
        if str(value).strip()
    }


def augment_training_manifest(
    *,
    base_manifest_path: str | Path,
    additional_manifest_path: str | Path,
    output_path: str | Path,
    base_image_prefix: str,
    additional_image_prefix: str,
) -> dict[str, Any]:
    base_path = Path(base_manifest_path)
    additional_path = Path(additional_manifest_path)
    output = Path(output_path)
    report_path = artifact_path(output, "report.json")
    near_duplicate_path = artifact_path(output, "near_duplicates.csv")
    require_absent(
        [output, Path(f"{output}.sha256"), report_path, near_duplicate_path]
    )
    base = _prefix_paths(
        _verified_manifest(base_path),
        base_image_prefix,
    )
    additional = _prefix_paths(
        _verified_manifest(additional_path),
        additional_image_prefix,
    )
    required = {
        "image_name",
        "image_path",
        "target",
        "group_id",
        "patient_id",
        "sha256",
        "perceptual_hash",
        "perceptual_hash_bits",
        "near_duplicate_hamming_threshold",
    }
    for name, frame in (("base", base), ("additional", additional)):
        missing = required - set(frame.columns)
        if missing:
            raise ManifestValidationError(
                f"{name} manifest is missing columns: {sorted(missing)}"
            )
    if set(base["split"].astype(str)) != {"train", "validation", "test"}:
        raise ManifestValidationError("Base manifest must contain three splits")
    if set(base["group_id"]) & set(additional["group_id"]):
        raise ManifestValidationError("Additional patient groups overlap base")
    if _known_identifiers(base, "patient_id") & _known_identifiers(
        additional,
        "patient_id",
    ):
        raise ManifestValidationError("Additional patient IDs overlap base")
    blank_patient_ids = additional["patient_id"].isna() | additional[
        "patient_id"
    ].fillna("").str.strip().eq("")
    if blank_patient_ids.any():
        if "patient_identifier_status" not in additional.columns:
            raise ManifestValidationError(
                "Additional records with blank patient IDs require an "
                "explicit patient_identifier_status"
            )
        statuses = additional.loc[
            blank_patient_ids,
            "patient_identifier_status",
        ].fillna("").str.strip().str.lower()
        if not statuses.eq("unavailable").all():
            raise ManifestValidationError(
                "Blank additional patient IDs must be marked unavailable"
            )
    if set(base["image_name"]) & set(additional["image_name"]):
        raise ManifestValidationError("Additional image IDs overlap base")
    if set(base["sha256"]) & set(additional["sha256"]):
        raise ManifestValidationError("Additional image hashes overlap base")
    settings_columns = (
        "perceptual_hash_bits",
        "near_duplicate_hamming_threshold",
        "perceptual_hash_algorithm",
        "perceptual_hash_implementation",
        "perceptual_hash_implementation_version",
    )
    for column in settings_columns:
        values = set(base[column].astype(str)) | set(
            additional[column].astype(str)
        )
        if len(values) != 1:
            raise ManifestValidationError(
                f"Manifest integrity setting differs: {column}"
            )
    additional["split"] = "train"
    additional["fold"] = -1
    columns = sorted(set(base.columns) | set(additional.columns))
    combined = pd.concat(
        [
            base.reindex(columns=columns),
            additional.reindex(columns=columns),
        ],
        ignore_index=True,
    ).sort_values("image_name", kind="mergesort")
    if combined["sha256"].duplicated().any():
        raise ManifestValidationError("Exact duplicates remain after merge")
    threshold = int(combined["near_duplicate_hamming_threshold"].iloc[0])
    near_duplicates = find_near_duplicate_pairs(
        combined,
        threshold=threshold,
    )
    split_by_image = combined.set_index("image_name")["split"]
    if not near_duplicates.empty:
        near_duplicates["left_split"] = near_duplicates[
            "left_image_name"
        ].map(split_by_image)
        near_duplicates["right_split"] = near_duplicates[
            "right_image_name"
        ].map(split_by_image)
    else:
        near_duplicates["left_split"] = pd.Series(dtype="string")
        near_duplicates["right_split"] = pd.Series(dtype="string")
    output.parent.mkdir(parents=True, exist_ok=True)
    near_duplicates.reindex(
        columns=[
            *NEAR_DUPLICATE_COLUMNS,
            "left_split",
            "right_split",
        ]
    ).to_csv(
        near_duplicate_path,
        index=False,
        lineterminator="\n",
    )
    if not near_duplicates.empty:
        split_leakage = (
            near_duplicates["left_split"]
            != near_duplicates["right_split"]
        )
        if split_leakage.any():
            raise ManifestValidationError(
                "Perceptual candidates cross training splits"
            )
    else:
        split_leakage = pd.Series(dtype="bool")
    cross_group_candidates = near_duplicates.loc[
        ~near_duplicates["same_group"]
    ]
    digest = write_manifest(combined, output)
    report = {
        "schema_version": 2,
        "source_manifest_sha256": Path(
            f"{base_path}.sha256"
        ).read_text(encoding="ascii").strip(),
        "additional_manifest_sha256": Path(
            f"{additional_path}.sha256"
        ).read_text(encoding="ascii").strip(),
        "split_manifest_sha256": digest,
        "additional_assignment": "train_only",
        "splits": _split_summary(combined),
        "group_leakage_records": 0,
        "hash_leakage_records": 0,
        "near_duplicate_hamming_threshold": threshold,
        "near_duplicate_pairs": int(len(near_duplicates)),
        "cross_group_near_duplicate_candidates": int(
            len(cross_group_candidates)
        ),
        "near_duplicate_split_leakage_records": int(split_leakage.sum()),
        "additional_patient_identifier_status": (
            "unavailable"
            if blank_patient_ids.any()
            else "available"
        ),
        "research_only": True,
    }
    write_json_exclusive(report_path, report)
    return {
        "manifest": output,
        "manifest_sha256": digest,
        "report": report_path,
        "near_duplicates": near_duplicate_path,
        "summary": report["splits"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--additional-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-image-prefix", required=True)
    parser.add_argument("--additional-image-prefix", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = augment_training_manifest(
        base_manifest_path=args.base_manifest,
        additional_manifest_path=args.additional_manifest,
        output_path=args.output,
        base_image_prefix=args.base_image_prefix,
        additional_image_prefix=args.additional_image_prefix,
    )
    print(result)


if __name__ == "__main__":
    main()
