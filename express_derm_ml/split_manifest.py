from __future__ import annotations

import argparse
import re
from pathlib import Path, PureWindowsPath
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .integrity import (
    NEAR_DUPLICATE_COLUMNS,
    find_near_duplicate_pairs,
)
from .manifest import (
    ManifestValidationError,
    artifact_path,
    canonical_manifest_sha256,
    read_manifest,
    save_json,
    write_manifest,
)


def _validate_relative_image_paths(frame: pd.DataFrame) -> None:
    invalid: list[str] = []
    for value in frame["image_path"].drop_duplicates():
        raw = str(value).strip()
        path = Path(raw)
        if (
            not raw
            or path.is_absolute()
            or PureWindowsPath(raw).is_absolute()
            or ".." in path.parts
        ):
            invalid.append(raw)
    if invalid:
        raise ManifestValidationError(
            "Manifest image paths must be relative to the image root: "
            f"{invalid[:5]}"
        )


def _validate_manifest(
    frame: pd.DataFrame,
    *,
    group_column: str,
    target_column: str,
    folds: int,
) -> tuple[pd.DataFrame, int]:
    required = {
        "image_name",
        "image_path",
        "sha256",
        "perceptual_hash",
        "perceptual_hash_algorithm",
        "perceptual_hash_implementation",
        "perceptual_hash_implementation_version",
        "perceptual_hash_bits",
        "near_duplicate_hamming_threshold",
        group_column,
        target_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ManifestValidationError(
            f"Missing manifest columns: {sorted(missing)}"
        )
    if folds < 3:
        raise ManifestValidationError(
            "At least 3 folds are required for train, validation and test"
        )
    if frame.empty:
        raise ManifestValidationError("Manifest is empty")

    normalized = frame.copy()
    normalized[group_column] = normalized[group_column].fillna("").astype(str)
    normalized[group_column] = normalized[group_column].str.strip()
    if (normalized[group_column] == "").any():
        raise ManifestValidationError("Every record must have a group identifier")

    numeric_targets = pd.to_numeric(
        normalized[target_column],
        errors="coerce",
    )
    if numeric_targets.isna().any() or not numeric_targets.isin([0, 1]).all():
        raise ManifestValidationError("Targets must be numeric 0 or 1")
    normalized[target_column] = numeric_targets.astype(int)
    targets = set(normalized[target_column].unique())
    if targets != {0, 1}:
        raise ManifestValidationError(
            f"Both binary targets 0 and 1 are required; found {sorted(targets)}"
        )

    _validate_relative_image_paths(normalized)
    if normalized["image_name"].duplicated().any():
        raise ManifestValidationError("Duplicate image names detected")
    if normalized["sha256"].duplicated().any():
        raise ManifestValidationError(
            "Exact duplicate image hashes detected before splitting"
        )
    valid_hashes = normalized["sha256"].map(
        lambda value: bool(re.fullmatch(r"[0-9a-f]{64}", str(value)))
    )
    if not valid_hashes.all():
        raise ManifestValidationError(
            "Every image SHA-256 must be 64 lowercase hexadecimal characters"
        )

    algorithms = set(normalized["perceptual_hash_algorithm"].astype(str))
    if algorithms != {"phash"}:
        raise ManifestValidationError(
            f"One phash algorithm is required; found {sorted(algorithms)}"
        )
    implementations = set(
        normalized["perceptual_hash_implementation"].astype(str)
    )
    implementation_versions = set(
        normalized["perceptual_hash_implementation_version"].astype(str)
    )
    if implementations != {"ImageHash"} or len(implementation_versions) != 1:
        raise ManifestValidationError(
            "Manifest must contain one pinned ImageHash implementation"
        )
    bit_widths = pd.to_numeric(
        normalized["perceptual_hash_bits"],
        errors="coerce",
    )
    thresholds = pd.to_numeric(
        normalized["near_duplicate_hamming_threshold"],
        errors="coerce",
    )
    if bit_widths.isna().any() or len(bit_widths.unique()) != 1:
        raise ManifestValidationError(
            "Manifest must contain one perceptual hash bit width"
        )
    if thresholds.isna().any() or len(thresholds.unique()) != 1:
        raise ManifestValidationError(
            "Manifest must contain one near-duplicate threshold"
        )
    bit_width = int(bit_widths.iloc[0])
    threshold = int(thresholds.iloc[0])
    expected_hex_length = bit_width // 4
    hash_lengths = normalized["perceptual_hash"].astype(str).str.len()
    if (
        bit_width % 4
        or not 16 <= bit_width <= 256
        or not hash_lengths.eq(expected_hex_length).all()
    ):
        raise ManifestValidationError(
            "Perceptual hash width does not match the stored hashes"
        )
    if not 0 <= threshold < bit_width:
        raise ManifestValidationError(
            "Near-duplicate threshold must fit the perceptual hash bit width"
        )
    if normalized[group_column].nunique() < folds:
        raise ManifestValidationError(
            f"At least {folds} distinct groups are required"
        )

    groups_per_target = normalized.groupby(target_column)[group_column].nunique()
    sparse_targets = groups_per_target.loc[lambda values: values < folds]
    if not sparse_targets.empty:
        details = {
            str(int(target)): int(count)
            for target, count in sparse_targets.items()
        }
        raise ManifestValidationError(
            f"Each target needs at least {folds} groups; found {details}"
        )

    return (
        normalized.sort_values(
            ["image_name", "sha256"],
            kind="mergesort",
        ).reset_index(drop=True),
        threshold,
    )


def _split_summary(
    frame: pd.DataFrame,
    *,
    group_column: str,
    target_column: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        split_frame = frame.loc[frame["split"] == split]
        summary[split] = {
            "images": int(len(split_frame)),
            "groups": int(split_frame[group_column].nunique()),
            "target_counts": {
                str(int(target)): int(count)
                for target, count in split_frame[target_column]
                .value_counts()
                .sort_index()
                .items()
            },
        }
    return summary


def split_manifest(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    group_column: str = "group_id",
    target_column: str = "target",
    folds: int = 5,
    seed: int = 2026,
) -> dict[str, Any]:
    source_frame = read_manifest(
        manifest_path,
        additional_text_columns=(group_column,),
    )
    source_sha256 = canonical_manifest_sha256(source_frame)
    frame, near_duplicate_threshold = _validate_manifest(
        source_frame,
        group_column=group_column,
        target_column=target_column,
        folds=folds,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    near_duplicate_path = artifact_path(output, "near_duplicates.csv")
    near_duplicate_frame = find_near_duplicate_pairs(
        frame,
        threshold=near_duplicate_threshold,
        group_column=group_column,
    )
    if near_duplicate_frame.empty:
        pd.DataFrame(columns=NEAR_DUPLICATE_COLUMNS).to_csv(
            near_duplicate_path,
            index=False,
            lineterminator="\n",
        )
    else:
        near_duplicate_frame.to_csv(
            near_duplicate_path,
            index=False,
            lineterminator="\n",
        )
    if (
        not near_duplicate_frame.empty
        and (~near_duplicate_frame["same_group"]).any()
    ):
        raise ManifestValidationError(
            "Near-duplicate images detected across patient groups; "
            f"review {near_duplicate_path} before splitting"
        )

    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    frame["fold"] = -1
    for fold, (_, validation_indices) in enumerate(
        splitter.split(
            frame,
            y=frame[target_column],
            groups=frame[group_column],
        )
    ):
        frame.loc[validation_indices, "fold"] = fold
    if (frame["fold"] < 0).any():
        raise ManifestValidationError("Some records were not assigned to a fold")

    frame["split"] = "train"
    frame.loc[frame["fold"] == 0, "split"] = "test"
    frame.loc[frame["fold"] == 1, "split"] = "validation"

    group_leakage = frame.groupby(group_column)["split"].nunique()
    if (group_leakage > 1).any():
        raise ManifestValidationError("Grouped split leakage detected")
    hash_leakage = frame.groupby("sha256")["split"].nunique()
    if (hash_leakage > 1).any():
        raise ManifestValidationError("Image-hash split leakage detected")
    if not near_duplicate_frame.empty:
        split_by_image = frame.set_index("image_name")["split"]
        near_duplicate_frame["left_split"] = near_duplicate_frame[
            "left_image_name"
        ].map(split_by_image)
        near_duplicate_frame["right_split"] = near_duplicate_frame[
            "right_image_name"
        ].map(split_by_image)
        cross_split = (
            near_duplicate_frame["left_split"]
            != near_duplicate_frame["right_split"]
        )
        if cross_split.any():
            near_duplicate_frame.to_csv(
                near_duplicate_path,
                index=False,
                lineterminator="\n",
            )
            raise ManifestValidationError(
                "Perceptual near-duplicate split leakage detected"
            )
        near_duplicate_frame.to_csv(
            near_duplicate_path,
            index=False,
            lineterminator="\n",
        )

    split_targets = frame.groupby("split")[target_column].nunique()
    for split in ("train", "validation", "test"):
        if int(split_targets.get(split, 0)) != 2:
            raise ManifestValidationError(
                f"Split {split!r} does not contain both targets"
            )

    split_sha256 = write_manifest(frame, output)
    report = {
        "schema_version": 1,
        "source_manifest_sha256": source_sha256,
        "split_manifest_sha256": split_sha256,
        "group_column": group_column,
        "target_column": target_column,
        "folds": folds,
        "seed": seed,
        "splits": _split_summary(
            frame,
            group_column=group_column,
            target_column=target_column,
        ),
        "group_leakage_records": 0,
        "hash_leakage_records": 0,
        "near_duplicate_hamming_threshold": near_duplicate_threshold,
        "near_duplicate_pairs": int(len(near_duplicate_frame)),
        "near_duplicate_split_leakage_records": 0,
    }
    report_path = artifact_path(output, "report.json")
    save_json(report_path, report)
    return {
        "manifest": output,
        "manifest_sha256": split_sha256,
        "report": report_path,
        "near_duplicates": near_duplicate_path,
        "summary": report["splits"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic patient-grouped dataset splits.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group-column", default="group_id")
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = split_manifest(
        manifest_path=args.manifest,
        output_path=args.output,
        group_column=args.group_column,
        target_column=args.target_column,
        folds=args.folds,
        seed=args.seed,
    )
    print(pd.DataFrame.from_dict(result["summary"], orient="index").to_string())
    print(
        f"Wrote grouped split manifest to {result['manifest']} "
        f"({result['manifest_sha256']})"
    )


if __name__ == "__main__":
    main()
