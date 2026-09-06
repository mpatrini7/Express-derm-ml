from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from .integrity import (
    NEAR_DUPLICATE_COLUMNS,
    find_near_duplicate_pairs,
    perceptual_hash_file,
    validate_perceptual_hash_runtime,
)
from .manifest import (
    ManifestValidationError,
    artifact_path,
    load_curation_config,
    save_json,
    sha256_file,
    write_manifest,
)

SUPPORTED_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def _safe_candidate(images_dir: Path, candidate: Path) -> Path | None:
    path = (images_dir / candidate).resolve()
    try:
        path.relative_to(images_dir)
    except ValueError:
        return None
    return path if path.is_file() else None


def find_image(images_dir: Path, image_name: str) -> Path | None:
    candidate = Path(image_name)
    if candidate.suffix:
        return _safe_candidate(images_dir, candidate)
    for suffix in SUPPORTED_SUFFIXES:
        path = _safe_candidate(images_dir, Path(f"{image_name}{suffix}"))
        if path is not None:
            return path
    return None


def _normalized_label(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _required_metadata_columns(config: dict[str, Any]) -> set[str]:
    columns = config["columns"]
    required = {
        str(columns["image"]),
        str(columns["group"]),
        str(config["label_mapping"]["source_column"]),
    }
    required.update(
        str(column)
        for key, column in columns.items()
        if key
        in {
            "patient",
            "patient_identifier_status",
            "selection_role",
            "lesion",
            "diagnosis",
            "source_url",
            "license",
            "license_url",
            "attribution",
        }
        and column
    )
    return required


def _mapped_labels(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        _normalized_label(source): mapped
        for source, mapped in config["label_mapping"]["values"].items()
    }


def _source_or_dataset_value(
    row: pd.Series,
    columns: dict[str, Any],
    dataset: dict[str, Any],
    key: str,
) -> str:
    source_column = columns.get(key)
    value = row[source_column] if source_column else dataset[key]
    normalized = "" if pd.isna(value) else str(value).strip()
    if not normalized:
        raise ManifestValidationError(
            f"Blank per-image provenance value for {key}"
        )
    return normalized


def _empty_csv(path: Path, columns: list[str]) -> None:
    pd.DataFrame(columns=columns).to_csv(
        path,
        index=False,
        lineterminator="\n",
    )


def _write_exclusions(path: Path, exclusions: list[dict[str, Any]]) -> None:
    if not exclusions:
        _empty_csv(path, ["row_number", "image_name", "reason"])
        return
    pd.DataFrame.from_records(exclusions).sort_values(
        ["row_number", "image_name"],
        kind="mergesort",
    ).to_csv(path, index=False, lineterminator="\n")


def _load_duplicate_exclusions(
    metadata: pd.DataFrame,
    config: dict[str, Any],
    duplicate_pairs_path: str | Path | None,
) -> tuple[dict[str, str], dict[str, Any] | None]:
    if duplicate_pairs_path is None:
        return {}, None

    source = Path(duplicate_pairs_path)
    pairs = pd.read_csv(source, dtype="string")
    required_columns = {"image_name_1", "image_name_2"}
    missing_columns = required_columns - set(pairs.columns)
    if missing_columns:
        raise ManifestValidationError(
            "Duplicate-pair file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    pairs = pairs.loc[:, ["image_name_1", "image_name_2"]].copy()
    for column in required_columns:
        pairs[column] = pairs[column].fillna("").str.strip()
    if pairs.empty or pairs.eq("").any(axis=None):
        raise ManifestValidationError(
            "Duplicate-pair file must contain non-blank pairs"
        )
    if pairs["image_name_1"].duplicated().any() or pairs[
        "image_name_2"
    ].duplicated().any():
        raise ManifestValidationError(
            "Duplicate-pair file must contain disjoint one-to-one pairs"
        )
    if set(pairs["image_name_1"]) & set(pairs["image_name_2"]):
        raise ManifestValidationError(
            "Duplicate-pair file must not contain chained pairs"
        )

    columns = config["columns"]
    image_column = str(columns["image"])
    metadata_names = metadata[image_column].fillna("").str.strip()
    if metadata_names.duplicated().any():
        raise ManifestValidationError(
            "Metadata image identifiers must be unique"
        )
    indexed = metadata.assign(_image_name=metadata_names).set_index(
        "_image_name",
        drop=True,
    )
    pair_names = set(pairs["image_name_1"]) | set(pairs["image_name_2"])
    unknown_names = sorted(pair_names - set(indexed.index))
    if unknown_names:
        raise ManifestValidationError(
            "Duplicate-pair file references images absent from metadata: "
            f"{unknown_names[:5]}"
        )

    group_column = str(columns["group"])
    label_column = str(config["label_mapping"]["source_column"])
    exclusions: dict[str, str] = {}
    for pair in pairs.itertuples(index=False):
        canonical = str(pair.image_name_1)
        excluded = str(pair.image_name_2)
        canonical_row = indexed.loc[canonical]
        excluded_row = indexed.loc[excluded]
        canonical_group = str(canonical_row[group_column]).strip()
        excluded_group = str(excluded_row[group_column]).strip()
        if canonical_group != excluded_group:
            raise ManifestValidationError(
                "Duplicate pair spans different patient groups: "
                f"{canonical}, {excluded}"
            )
        canonical_label = _normalized_label(canonical_row[label_column])
        excluded_label = _normalized_label(excluded_row[label_column])
        if canonical_label != excluded_label:
            raise ManifestValidationError(
                "Duplicate pair has conflicting source labels: "
                f"{canonical}, {excluded}"
            )
        exclusions[excluded] = canonical

    receipt = {
        "file": source.name,
        "sha256": sha256_file(source),
        "pairs": int(len(pairs)),
        "excluded_records": int(len(exclusions)),
        "policy": "retain_image_name_1_exclude_image_name_2",
    }
    return exclusions, receipt


def prepare_manifest(
    *,
    metadata_path: str | Path,
    images_dir: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    allow_exclusions: bool = False,
    duplicate_pairs_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_curation_config(config_path)
    required = _required_metadata_columns(config)
    metadata_columns = set(pd.read_csv(metadata_path, nrows=0).columns)
    missing = required - metadata_columns
    if missing:
        raise ManifestValidationError(
            f"Missing metadata columns: {sorted(missing)}"
        )
    metadata = pd.read_csv(
        metadata_path,
        dtype={
            column: "string"
            for column in required
        },
    )
    duplicate_exclusions, duplicate_receipt = _load_duplicate_exclusions(
        metadata,
        config,
        duplicate_pairs_path,
    )

    images_root = Path(images_dir).resolve()
    if not images_root.is_dir():
        raise ManifestValidationError(
            f"Images directory does not exist: {images_root}"
        )

    dataset = config["dataset"]
    columns = config["columns"]
    labels = _mapped_labels(config)
    integrity = config["integrity"]
    perceptual_config = integrity["perceptual_hash"]
    validate_perceptual_hash_runtime(perceptual_config)
    hash_size = int(perceptual_config["hash_size"])
    highfreq_factor = int(perceptual_config["highfreq_factor"])
    near_duplicate_threshold = int(
        integrity["near_duplicate_hamming_threshold"]
    )
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    row_number_by_image: dict[str, int] = {}

    total_rows = len(metadata)
    for position, (row_number, row) in enumerate(
        metadata.iterrows(),
        start=1,
    ):
        if position == 1 or position % 500 == 0:
            print(
                f"Scanning source image {position:,}/{total_rows:,}",
                flush=True,
            )
        image_value = row[columns["image"]]
        image_name = "" if pd.isna(image_value) else str(image_value).strip()
        if image_name:
            row_number_by_image[image_name] = int(row_number)
        exclusion = {
            "row_number": int(row_number),
            "image_name": image_name,
            "reason": "",
        }
        if not image_name:
            exclusion["reason"] = "missing_image_name"
            exclusions.append(exclusion)
            continue
        path = find_image(images_root, image_name)
        if path is None:
            exclusion["reason"] = "image_not_found"
            exclusions.append(exclusion)
            continue
        duplicate_of = duplicate_exclusions.get(image_name)
        if duplicate_of is not None:
            exclusion["reason"] = f"source_duplicate_of:{duplicate_of}"
            exclusions.append(exclusion)
            continue

        group_value = row[columns["group"]]
        if pd.isna(group_value) or not str(group_value).strip():
            exclusion["reason"] = "missing_group_id"
            exclusions.append(exclusion)
            continue

        source_label = _normalized_label(
            row[config["label_mapping"]["source_column"]]
        )
        mapped = labels.get(source_label)
        if mapped is None:
            exclusion["reason"] = f"unmapped_label:{source_label or 'blank'}"
            exclusions.append(exclusion)
            continue

        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
            perceptual_hash = perceptual_hash_file(
                path,
                hash_size=hash_size,
                highfreq_factor=highfreq_factor,
            )
        except Exception:
            exclusion["reason"] = "invalid_image"
            exclusions.append(exclusion)
            continue

        relative_path = path.relative_to(images_root).as_posix()
        record: dict[str, Any] = {
            "image_name": image_name,
            "image_path": relative_path,
            "target": int(mapped["target"]),
            "curated_label": str(mapped["label"]),
            "source_label": source_label,
            "group_id": str(group_value).strip(),
            "collection_id": str(dataset["id"]),
            "dataset_version": str(dataset["version"]),
            "source_url": _source_or_dataset_value(
                row, columns, dataset, "source_url"
            ),
            "license": _source_or_dataset_value(
                row, columns, dataset, "license"
            ),
            "license_url": _source_or_dataset_value(
                row, columns, dataset, "license_url"
            ),
            "attribution": _source_or_dataset_value(
                row, columns, dataset, "attribution"
            ),
            "width": int(width),
            "height": int(height),
            "sha256": sha256_file(path),
            "perceptual_hash": perceptual_hash,
            "perceptual_hash_algorithm": str(
                perceptual_config["algorithm"]
            ),
            "perceptual_hash_implementation": str(
                perceptual_config["implementation"]
            ),
            "perceptual_hash_implementation_version": str(
                perceptual_config["implementation_version"]
            ),
            "perceptual_hash_bits": hash_size * hash_size,
            "near_duplicate_hamming_threshold": near_duplicate_threshold,
        }
        for output_column, config_key in (
            ("patient_id", "patient"),
            ("patient_identifier_status", "patient_identifier_status"),
            ("selection_role", "selection_role"),
            ("lesion_id", "lesion"),
            ("diagnosis", "diagnosis"),
        ):
            source_column = columns.get(config_key)
            if source_column:
                value = row[source_column]
                record[output_column] = (
                    "" if pd.isna(value) else str(value).strip()
                )
        records.append(record)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    exclusion_path = artifact_path(output, "exclusions.csv")
    _write_exclusions(exclusion_path, exclusions)

    unexpected_exclusions = [
        exclusion
        for exclusion in exclusions
        if not exclusion["reason"].startswith("source_duplicate_of:")
    ]
    if unexpected_exclusions and not allow_exclusions:
        if len(unexpected_exclusions) == len(exclusions):
            message = f"Excluded {len(exclusions)} records"
        else:
            message = (
                f"Excluded {len(unexpected_exclusions)} unexpected records"
            )
        raise ManifestValidationError(
            f"{message}; review {exclusion_path}"
        )
    if not records:
        raise ManifestValidationError("No valid images were found")

    frame = (
        pd.DataFrame.from_records(records)
        .sort_values("image_name", kind="mergesort")
        .reset_index(drop=True)
    )
    expected_images = dataset.get("expected_images")
    if expected_images is not None and len(metadata) != int(expected_images):
        raise ManifestValidationError(
            "Metadata row count does not match configured collection: "
            f"expected {expected_images}, found {len(metadata)}"
        )

    duplicate_frame = frame.loc[
        frame.duplicated("sha256", keep=False)
    ].sort_values(["sha256", "image_name"], kind="mergesort")
    duplicate_path = artifact_path(output, "duplicates.csv")
    if duplicate_frame.empty:
        _empty_csv(duplicate_path, list(frame.columns))
    else:
        duplicate_frame.to_csv(
            duplicate_path,
            index=False,
            lineterminator="\n",
        )
        policy = integrity.get("exact_duplicate_policy", "block")
        if policy != "exclude_within_group_and_label":
            raise ManifestValidationError(
                "Exact duplicate images detected; "
                f"review {duplicate_path} before splitting"
            )
        conflicting_groups = duplicate_frame.groupby("sha256").filter(
            lambda group: (
                group["group_id"].nunique() != 1
                or group["source_label"].nunique() != 1
                or group["target"].nunique() != 1
            )
        )
        if not conflicting_groups.empty:
            raise ManifestValidationError(
                "Exact duplicate images span patient groups or labels; "
                f"review {duplicate_path} before splitting"
            )
        exact_duplicate_exclusions: dict[str, str] = {}
        for _, group in duplicate_frame.groupby("sha256", sort=True):
            names = sorted(group["image_name"].astype(str))
            canonical = names[0]
            for excluded in names[1:]:
                exact_duplicate_exclusions[excluded] = canonical
                exclusions.append(
                    {
                        "row_number": row_number_by_image[excluded],
                        "image_name": excluded,
                        "reason": f"exact_duplicate_of:{canonical}",
                    }
                )
        frame = frame.loc[
            ~frame["image_name"].isin(exact_duplicate_exclusions)
        ].reset_index(drop=True)
        _write_exclusions(exclusion_path, exclusions)

    exact_duplicate_records = int(len(duplicate_frame))
    exact_duplicate_groups = int(duplicate_frame["sha256"].nunique())
    exact_duplicate_exclusion_count = max(
        exact_duplicate_records - exact_duplicate_groups,
        0,
    )

    near_duplicate_frame = find_near_duplicate_pairs(
        frame,
        threshold=near_duplicate_threshold,
    )
    near_duplicate_path = artifact_path(output, "near_duplicates.csv")
    if near_duplicate_frame.empty:
        _empty_csv(near_duplicate_path, NEAR_DUPLICATE_COLUMNS)
    else:
        near_duplicate_frame.to_csv(
            near_duplicate_path,
            index=False,
            lineterminator="\n",
        )
    cross_group_near_duplicates = near_duplicate_frame.loc[
        ~near_duplicate_frame["same_group"]
    ]
    cross_group_policy = integrity.get(
        "cross_group_near_duplicate_policy",
        "block",
    )
    allow_train_only_candidates = (
        dataset.get("split_policy") == "train_only"
        and cross_group_policy == "allow_train_only_reviewed_candidates"
    )
    if (
        not cross_group_near_duplicates.empty
        and not allow_train_only_candidates
    ):
        raise ManifestValidationError(
            "Near-duplicate images detected across patient groups; "
            f"review {near_duplicate_path} before splitting"
        )

    manifest_sha256 = write_manifest(frame, output)
    target_counts = {
        str(int(target)): int(count)
        for target, count in frame["target"].value_counts().sort_index().items()
    }
    dataset_card = {
        "schema_version": 1,
        "dataset": dataset,
        "label_mapping": config["label_mapping"],
        "config_sha256": sha256_file(config_path),
        "metadata_sha256": sha256_file(metadata_path),
        "manifest_sha256": manifest_sha256,
        "records": int(len(frame)),
        "groups": int(frame["group_id"].nunique()),
        "target_counts": target_counts,
        "exclusions": int(len(exclusions)),
        "unexpected_exclusions": int(len(unexpected_exclusions)),
        "source_duplicate_exclusions": int(len(duplicate_exclusions)),
        "exact_duplicate_records": exact_duplicate_records,
        "exact_duplicate_groups": exact_duplicate_groups,
        "exact_duplicate_exclusions": exact_duplicate_exclusion_count,
        "exact_duplicate_policy": integrity.get(
            "exact_duplicate_policy",
            "block",
        ),
        "perceptual_hash": perceptual_config,
        "near_duplicate_hamming_threshold": near_duplicate_threshold,
        "near_duplicate_pairs": int(len(near_duplicate_frame)),
        "cross_group_near_duplicate_pairs": int(
            len(cross_group_near_duplicates)
        ),
        "cross_group_near_duplicate_policy": cross_group_policy,
        "train_only_near_duplicate_candidates_allowed": bool(
            allow_train_only_candidates
        ),
    }
    if duplicate_receipt is not None:
        dataset_card["source_duplicate_pairs"] = duplicate_receipt
    dataset_card_path = artifact_path(output, "dataset.json")
    save_json(dataset_card_path, dataset_card)
    return {
        "manifest": output,
        "manifest_sha256": manifest_sha256,
        "dataset_card": dataset_card_path,
        "duplicates": duplicate_path,
        "near_duplicates": near_duplicate_path,
        "exclusions": exclusion_path,
        "records": len(frame),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a validated, attributed image manifest.",
    )
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-exclusions", action="store_true")
    parser.add_argument(
        "--duplicate-pairs",
        help=(
            "Official duplicate-pair CSV. Each image_name_2 is excluded after "
            "same-group and same-label validation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_manifest(
        metadata_path=args.metadata,
        images_dir=args.images_dir,
        config_path=args.config,
        output_path=args.output,
        allow_exclusions=args.allow_exclusions,
        duplicate_pairs_path=args.duplicate_pairs,
    )
    print(
        f"Wrote {result['records']} records to {result['manifest']} "
        f"({result['manifest_sha256']})"
    )


if __name__ == "__main__":
    main()
