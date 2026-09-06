from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import require_absent, write_json_exclusive
from .common import sha256_file
from .integrity import NEAR_DUPLICATE_COLUMNS, find_near_duplicate_pairs
from .manifest import (
    ManifestValidationError,
    canonical_manifest_sha256,
    read_manifest,
    write_manifest,
)


EXTERNAL_TEXT_COLUMNS = (
    "diagnosis_class",
    "diagnosis_full",
    "diagnosis_confirm_type",
    "patient_identifier_status",
    "image_type",
)


def _external_candidate_names(candidates: pd.DataFrame) -> set[str]:
    names: set[str] = set()
    for row in candidates.itertuples(index=False):
        left_external = str(row.left_group_id).startswith("external:")
        right_external = str(row.right_group_id).startswith("external:")
        if left_external == right_external:
            raise ManifestValidationError(
                "Cross-source candidate must have exactly one external side"
            )
        names.add(
            str(row.left_image_name)
            if left_external
            else str(row.right_image_name)
        )
    return names


def _read_recorded_manifest_hash(path: Path) -> str:
    digest_path = Path(f"{path}.sha256")
    digest = digest_path.read_text(encoding="ascii").strip()
    if len(digest) != 64:
        raise ManifestValidationError(f"Invalid manifest digest: {digest_path}")
    return digest


def _load_verified_external_card(
    manifest_path: Path,
    *,
    manifest_sha256: str,
    records: int,
    lesions: int,
) -> tuple[dict[str, Any], Path]:
    card_path = manifest_path.with_name(f"{manifest_path.stem}.dataset.json")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    required = {
        "manifest_sha256": manifest_sha256,
        "records": records,
        "lesions": lesions,
        "training_authorized": False,
        "purpose": "external_research_evaluation_only",
        "cross_source_exact_duplicates": 0,
        "cross_source_near_duplicate_candidates": 0,
    }
    for key, expected in required.items():
        if card.get(key) != expected:
            raise ManifestValidationError(
                f"Existing external dataset card mismatch for {key}"
            )
    source_hashes = card.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ManifestValidationError(
            "Existing external dataset card has no verified source hashes"
        )
    return card, card_path


def _verify_external_images(frame: pd.DataFrame, images_root: Path) -> None:
    root = images_root.resolve()
    for position, row in enumerate(frame.itertuples(index=False), start=1):
        if position == 1 or position % 500 == 0:
            print(
                f"Revalidating external image {position:,}/{len(frame):,}",
                flush=True,
            )
        path = (root / str(row.image_path)).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ManifestValidationError(
                "External image path escapes its root"
            ) from error
        if not path.is_file() or sha256_file(path) != str(row.sha256):
            raise ManifestValidationError(
                f"External image integrity check failed: {row.image_name}"
            )


def revalidate_external_manifest(
    *,
    external_manifest_path: str | Path,
    images_dir: str | Path,
    reference_manifest_path: str | Path,
    output_path: str | Path,
    near_duplicate_hamming_threshold: int,
    exclude_cross_source_candidates: bool = False,
) -> dict[str, Any]:
    if near_duplicate_hamming_threshold < 0:
        raise ValueError("Near-duplicate threshold cannot be negative")
    external_path = Path(external_manifest_path).resolve()
    reference_path = Path(reference_manifest_path).resolve()
    output = Path(output_path)
    report_path = output.with_name(
        f"{output.stem}.reference_near_duplicates.csv"
    )
    output_card_path = output.with_name(f"{output.stem}.dataset.json")
    exclusion_path = output.with_name(
        f"{output.stem}.reference_candidate_exclusions.csv"
    )
    require_absent(
        [
            output,
            Path(f"{output}.sha256"),
            report_path,
            output_card_path,
            exclusion_path,
        ]
    )

    external = read_manifest(
        external_path,
        additional_text_columns=EXTERNAL_TEXT_COLUMNS,
    )
    required_columns = {
        "image_name",
        "image_path",
        "lesion_id",
        "group_id",
        "sha256",
        "perceptual_hash",
    }
    missing = required_columns - set(external.columns)
    if missing:
        raise ManifestValidationError(
            f"External manifest is missing columns: {sorted(missing)}"
        )
    if external.empty or external["lesion_id"].duplicated().any():
        raise ManifestValidationError(
            "External manifest must contain one image per lesion"
        )
    external_hash = canonical_manifest_sha256(external)
    if external_hash != _read_recorded_manifest_hash(external_path):
        raise ManifestValidationError("Existing external manifest hash mismatch")
    existing_card, existing_card_path = _load_verified_external_card(
        external_path,
        manifest_sha256=external_hash,
        records=int(len(external)),
        lesions=int(external["lesion_id"].nunique()),
    )
    _verify_external_images(external, Path(images_dir))
    source_record_count = int(len(external))

    reference = read_manifest(reference_path)
    reference_hash = canonical_manifest_sha256(reference)
    if reference_hash != _read_recorded_manifest_hash(reference_path):
        raise ManifestValidationError("Reference manifest hash mismatch")
    exact_reference_hashes = set(reference["sha256"].astype(str))
    if external["sha256"].astype(str).isin(exact_reference_hashes).any():
        raise ManifestValidationError(
            "External images exactly overlap the reference corpus"
        )

    reference_audit = reference.loc[
        :, ["image_name", "group_id", "sha256", "perceptual_hash"]
    ].copy()
    reference_audit["group_id"] = (
        "reference:" + reference_audit["group_id"].astype(str)
    )
    external_audit = external.loc[
        :, ["image_name", "group_id", "sha256", "perceptual_hash"]
    ].copy()
    external_audit["group_id"] = (
        "external:" + external_audit["group_id"].astype(str)
    )
    near_duplicates = find_near_duplicate_pairs(
        pd.concat([reference_audit, external_audit], ignore_index=True),
        threshold=near_duplicate_hamming_threshold,
    )
    cross_source = near_duplicates.loc[
        near_duplicates["left_group_id"].str.split(":").str[0]
        != near_duplicates["right_group_id"].str.split(":").str[0]
    ].copy()

    excluded_names: set[str] = set()
    excluded_candidates = cross_source.copy()
    if not cross_source.empty and not exclude_cross_source_candidates:
        output.parent.mkdir(parents=True, exist_ok=True)
        cross_source.to_csv(
            report_path,
            index=False,
            columns=NEAR_DUPLICATE_COLUMNS,
            lineterminator="\n",
        )
        raise ManifestValidationError(
            "External manifest has perceptual candidates against the "
            "reference corpus"
        )
    if not cross_source.empty:
        excluded_names = _external_candidate_names(cross_source)
        external = external.loc[
            ~external["image_name"].astype(str).isin(excluded_names)
        ].copy()
        cleaned_audit = external.loc[
            :, ["image_name", "group_id", "sha256", "perceptual_hash"]
        ].copy()
        cleaned_audit["group_id"] = (
            "external:" + cleaned_audit["group_id"].astype(str)
        )
        cleaned_pairs = find_near_duplicate_pairs(
            pd.concat([reference_audit, cleaned_audit], ignore_index=True),
            threshold=near_duplicate_hamming_threshold,
        )
        cross_source = cleaned_pairs.loc[
            cleaned_pairs["left_group_id"].str.split(":").str[0]
            != cleaned_pairs["right_group_id"].str.split(":").str[0]
        ].copy()
        if not cross_source.empty:
            raise RuntimeError(
                "Cross-source candidates remain after deterministic exclusion"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    cross_source.to_csv(
        report_path,
        index=False,
        columns=NEAR_DUPLICATE_COLUMNS,
        lineterminator="\n",
    )
    if excluded_candidates.empty:
        pd.DataFrame(
            columns=[*NEAR_DUPLICATE_COLUMNS, "excluded_external_image_name"]
        ).to_csv(exclusion_path, index=False, lineterminator="\n")
    else:
        excluded_candidates["excluded_external_image_name"] = (
            excluded_candidates.apply(
                lambda row: (
                    str(row["left_image_name"])
                    if str(row["left_group_id"]).startswith("external:")
                    else str(row["right_image_name"])
                ),
                axis=1,
            )
        )
        excluded_candidates.to_csv(
            exclusion_path,
            index=False,
            columns=[
                *NEAR_DUPLICATE_COLUMNS,
                "excluded_external_image_name",
            ],
            lineterminator="\n",
        )

    rebound_hash = write_manifest(external, output)
    if not excluded_names and rebound_hash != external_hash:
        raise RuntimeError("Revalidated external manifest identity changed")
    card = dict(existing_card)
    card.update(
        {
            "schema_version": 2,
            "manifest_sha256": rebound_hash,
            "reference_manifest_sha256": reference_hash,
            "records": int(len(external)),
            "lesions": int(external["lesion_id"].nunique()),
            "melanoma_positive": int(
                external["target_melanoma"].astype(int).sum()
            ),
            "broad_malignancy_positive": int(
                external["target_broad_malignancy"].astype(int).sum()
            ),
            "cross_source_exact_duplicates": 0,
            "cross_source_near_duplicate_candidates": 0,
            "revalidation": {
                "source_manifest_sha256": external_hash,
                "source_manifest_file_sha256": sha256_file(external_path),
                "source_dataset_card_sha256": sha256_file(existing_card_path),
                "reference_manifest_file_sha256": sha256_file(reference_path),
                "external_image_hashes_verified": source_record_count,
                "source_records": source_record_count,
                "excluded_cross_source_perceptual_candidates": int(
                    len(excluded_names)
                ),
                "exclusion_policy": (
                    "exclude_external_image_for_every_cross_source_phash_pair"
                    if excluded_names
                    else "none_required"
                ),
                "candidate_exclusions_sha256": sha256_file(exclusion_path),
                "near_duplicate_hamming_threshold": (
                    near_duplicate_hamming_threshold
                ),
            },
        }
    )
    write_json_exclusive(output_card_path, card)
    return card


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Revalidate an immutable external manifest against a new "
            "reference corpus without requiring the original archive file."
        )
    )
    parser.add_argument("--external-manifest", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--near-duplicate-hamming-threshold",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--exclude-cross-source-candidates",
        action="store_true",
        help=(
            "Create a decontaminated external subset by excluding every "
            "external image in a cross-source perceptual candidate pair."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = revalidate_external_manifest(
        external_manifest_path=args.external_manifest,
        images_dir=args.images_dir,
        reference_manifest_path=args.reference_manifest,
        output_path=args.output,
        near_duplicate_hamming_threshold=(
            args.near_duplicate_hamming_threshold
        ),
        exclude_cross_source_candidates=(
            args.exclude_cross_source_candidates
        ),
    )
    print(result)


if __name__ == "__main__":
    main()
