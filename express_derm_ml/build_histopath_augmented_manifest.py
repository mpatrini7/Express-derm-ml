from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from .artifacts import require_absent, write_json_exclusive
from .common import sha256_file
from .integrity import (
    NEAR_DUPLICATE_COLUMNS,
    find_near_duplicate_pairs,
    perceptual_hash_file,
    validate_perceptual_hash_runtime,
)
from .manifest import (
    artifact_path,
    canonical_manifest_sha256,
    read_manifest,
    write_manifest,
)


LICENSE_URLS = {
    "CC-0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC-BY": "https://creativecommons.org/licenses/by/4.0/",
    "CC-BY-NC": "https://creativecommons.org/licenses/by-nc/4.0/",
}


class _Components:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root

    def groups(self) -> list[list[str]]:
        groups: dict[str, list[str]] = {}
        for value in self.parent:
            groups.setdefault(self.find(value), []).append(value)
        return sorted(
            (sorted(values) for values in groups.values()),
            key=lambda values: values[0],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a train-only histopathology augmentation while protecting "
            "the base splits, external evaluations, and demo images."
        ),
    )
    parser.add_argument("--downloaded-metadata", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument(
        "--protected-manifest",
        action="append",
        default=[],
        help="External manifest with image_name, sha256, and perceptual_hash.",
    )
    parser.add_argument("--demo-manifest", required=True)
    parser.add_argument("--demo-images-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--source-collection-id",
        default="isic-collection-294-histopath-mel-nevus",
    )
    parser.add_argument(
        "--source-dataset-version",
        default="isic-collection-294-api-snapshot-2026-08-13",
    )
    parser.add_argument("--source-description", default="histopathology")
    parser.add_argument("--component-prefix", default="HISTO")
    return parser.parse_args()


def _scan_image(
    row: Any,
    images_root: Path,
    *,
    hash_size: int,
    highfreq_factor: int,
) -> dict[str, Any]:
    path = (images_root / str(row.image_path)).resolve()
    try:
        path.relative_to(images_root)
    except ValueError as error:
        raise RuntimeError("Downloaded image path escapes its root") from error
    if not path.is_file() or sha256_file(path) != str(row.sha256):
        raise RuntimeError(f"Downloaded image integrity mismatch: {row.image_name}")
    with Image.open(path) as image:
        width, height = image.size
        image.verify()
    return {
        "image_name": str(row.image_name),
        "width": int(width),
        "height": int(height),
        "perceptual_hash": perceptual_hash_file(
            path,
            hash_size=hash_size,
            highfreq_factor=highfreq_factor,
        ),
    }


def _scan_demo(
    row: Any,
    images_root: Path,
    *,
    hash_size: int,
    highfreq_factor: int,
) -> dict[str, str]:
    path = (images_root / str(row.filename)).resolve()
    try:
        path.relative_to(images_root)
    except ValueError as error:
        raise RuntimeError("Demo image path escapes its root") from error
    if not path.is_file() or sha256_file(path) != str(row.sha256):
        raise RuntimeError(f"Demo image integrity mismatch: {row.isic_id}")
    return {
        "image_name": str(row.isic_id),
        "sha256": str(row.sha256),
        "perceptual_hash": perceptual_hash_file(
            path,
            hash_size=hash_size,
            highfreq_factor=highfreq_factor,
        ),
    }


def _protected_rows(frame: pd.DataFrame, namespace: str) -> pd.DataFrame:
    required = {"image_name", "sha256", "perceptual_hash", "group_id"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"Protected manifest is missing columns: {sorted(missing)}"
        )
    result = frame.loc[:, sorted(required)].copy()
    result["audit_id"] = namespace + "::" + result["image_name"].astype(str)
    result["source_kind"] = "protected"
    result["target"] = -1
    return result


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("Workers must be positive")
    output = Path(args.output)
    report_path = artifact_path(output, "report.json")
    exclusions_path = artifact_path(output, "exclusions.csv")
    near_duplicates_path = artifact_path(output, "near_duplicates.csv")
    require_absent(
        [
            output,
            Path(f"{output}.sha256"),
            report_path,
            exclusions_path,
            near_duplicates_path,
        ]
    )

    base_path = Path(args.base_manifest)
    base = read_manifest(base_path)
    base_sha256 = canonical_manifest_sha256(base)
    recorded_base_sha256 = Path(f"{base_path}.sha256").read_text().strip()
    if base_sha256 != recorded_base_sha256:
        raise RuntimeError("Base manifest hash mismatch")
    settings = {
        "algorithm": str(base["perceptual_hash_algorithm"].iloc[0]),
        "implementation": str(
            base["perceptual_hash_implementation"].iloc[0]
        ),
        "implementation_version": str(
            base["perceptual_hash_implementation_version"].iloc[0]
        ),
        "hash_size": int(int(base["perceptual_hash_bits"].iloc[0]) ** 0.5),
        "highfreq_factor": 4,
    }
    if any(
        base[column].astype(str).nunique() != 1
        for column in (
            "perceptual_hash_algorithm",
            "perceptual_hash_implementation",
            "perceptual_hash_implementation_version",
            "perceptual_hash_bits",
            "near_duplicate_hamming_threshold",
        )
    ):
        raise RuntimeError("Base manifest integrity settings are inconsistent")
    validate_perceptual_hash_runtime(settings)
    hash_size = int(settings["hash_size"])
    highfreq_factor = int(settings["highfreq_factor"])
    threshold = int(base["near_duplicate_hamming_threshold"].iloc[0])

    metadata_path = Path(args.downloaded_metadata)
    metadata = pd.read_csv(metadata_path, keep_default_na=False)
    if metadata["image_name"].duplicated().any():
        raise RuntimeError("Downloaded metadata image IDs are not unique")
    base_patients = set(base["patient_id"].astype(str))
    excluded_patient_overlap = metadata["patient_id"].astype(str).isin(
        base_patients
    )
    source = metadata.loc[~excluded_patient_overlap].copy()
    images_root = Path(args.images_dir).resolve()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        scans = []
        for position, scan in enumerate(
            executor.map(
                lambda row: _scan_image(
                    row,
                    images_root,
                    hash_size=hash_size,
                    highfreq_factor=highfreq_factor,
                ),
                source.itertuples(index=False),
            ),
            start=1,
        ):
            scans.append(scan)
            if position == 1 or position % 250 == 0 or position == len(source):
                print(
                    f"Hashed {args.source_description} image "
                    f"{position:,}/{len(source):,}",
                    flush=True,
                )
    source = source.merge(
        pd.DataFrame.from_records(scans),
        on="image_name",
        validate="one_to_one",
    )

    source_records = pd.DataFrame(
        {
            "attribution": source["attribution"].astype(str),
            "collection_id": str(args.source_collection_id),
            "curated_label": source["target"].map(
                {0: "melanoma_attention_negative", 1: "melanoma_attention_positive"}
            ),
            "dataset_version": str(args.source_dataset_version),
            "diagnosis": source["diagnosis_3"].astype(str),
            "fold": -1,
            "group_id": source["patient_id"].astype(str),
            "height": source["height"].astype(int),
            "image_name": source["image_name"].astype(str),
            "image_path": source["image_path"].astype(str),
            "lesion_id": source["lesion_id"].astype(str),
            "license": source["copyright_license"].astype(str),
            "license_url": source["copyright_license"].map(LICENSE_URLS),
            "near_duplicate_hamming_threshold": threshold,
            "patient_id": source["patient_id"].astype(str),
            "perceptual_hash": source["perceptual_hash"].astype(str),
            "perceptual_hash_algorithm": settings["algorithm"],
            "perceptual_hash_bits": hash_size * hash_size,
            "perceptual_hash_implementation": settings["implementation"],
            "perceptual_hash_implementation_version": settings[
                "implementation_version"
            ],
            "sha256": source["sha256"].astype(str),
            "source_label": source["diagnosis_1"].str.lower(),
            "source_url": source["full_image_url"].astype(str),
            "split": "train",
            "target": source["target"].astype(int),
            "width": source["width"].astype(int),
        }
    )
    if source_records["license_url"].isna().any():
        raise RuntimeError("A source license URL is missing")

    protected_frames = [_protected_rows(base, "base")]
    protected_hashes = {
        "base_manifest": base_sha256,
        "downloaded_metadata": sha256_file(metadata_path),
    }
    for index, manifest_value in enumerate(args.protected_manifest, start=1):
        manifest_path = Path(manifest_value)
        protected = pd.read_csv(manifest_path, keep_default_na=False)
        protected_frames.append(_protected_rows(protected, f"external{index}"))
        protected_hashes[f"protected_manifest_{index}"] = sha256_file(
            manifest_path
        )

    demo_path = Path(args.demo_manifest)
    demo = pd.read_csv(demo_path, keep_default_na=False)
    demo_root = Path(args.demo_images_dir).resolve()
    with ThreadPoolExecutor(max_workers=min(args.workers, len(demo))) as executor:
        demo_scans = list(
            executor.map(
                lambda row: _scan_demo(
                    row,
                    demo_root,
                    hash_size=hash_size,
                    highfreq_factor=highfreq_factor,
                ),
                demo.itertuples(index=False),
            )
        )
    demo_frame = pd.DataFrame.from_records(demo_scans)
    demo_frame["group_id"] = "demo::" + demo_frame["image_name"]
    protected_frames.append(_protected_rows(demo_frame, "demo"))
    protected_hashes["demo_manifest"] = sha256_file(demo_path)

    new_audit = source_records.loc[
        :, ["image_name", "sha256", "perceptual_hash", "group_id", "target"]
    ].copy()
    new_audit["audit_id"] = "new::" + new_audit["image_name"]
    new_audit["source_kind"] = "new"
    audit = pd.concat(
        [*protected_frames, new_audit],
        ignore_index=True,
        sort=False,
    )
    audit_for_pairs = audit.rename(columns={"image_name": "original_image_name"})
    audit_for_pairs["image_name"] = audit_for_pairs["audit_id"]

    exact_excluded = set(
        new_audit.loc[
            new_audit["sha256"].isin(
                set(
                    pd.concat(protected_frames, ignore_index=True)["sha256"].astype(
                        str
                    )
                )
            ),
            "audit_id",
        ].astype(str)
    )
    new_sha_duplicates = new_audit.loc[
        new_audit["sha256"].duplicated(keep=False)
    ]
    for _, group in new_sha_duplicates.groupby("sha256", sort=True):
        audit_ids = sorted(group["audit_id"].astype(str))
        if group["target"].nunique() != 1:
            exact_excluded.update(audit_ids)
        else:
            exact_excluded.update(audit_ids[1:])

    pair_input = audit_for_pairs.loc[
        ~audit_for_pairs["audit_id"].isin(exact_excluded)
    ].reset_index(drop=True)
    candidate_pairs = find_near_duplicate_pairs(pair_input, threshold=threshold)
    components = _Components()
    for row in candidate_pairs.itertuples(index=False):
        components.union(str(row.left_image_name), str(row.right_image_name))
    indexed_audit = audit.set_index("audit_id", drop=False)
    near_excluded: set[str] = set()
    component_records = []
    for component_index, audit_ids in enumerate(components.groups(), start=1):
        component = indexed_audit.loc[audit_ids]
        new_members = sorted(
            component.loc[component["source_kind"].eq("new"), "audit_id"].astype(
                str
            )
        )
        if not new_members:
            continue
        protected_members = component.loc[
            component["source_kind"].eq("protected")
        ]
        if not protected_members.empty:
            excluded = new_members
            reason = "near_duplicate_of_protected_image"
            retained = sorted(protected_members["audit_id"].astype(str))[0]
        else:
            targets = component.loc[new_members, "target"].astype(int)
            if targets.nunique() != 1:
                excluded = new_members
                retained = ""
                reason = "near_duplicate_component_conflicting_labels"
            else:
                retained = new_members[0]
                excluded = new_members[1:]
                reason = "near_duplicate_component_redundant"
        near_excluded.update(excluded)
        for audit_id in excluded:
            component_records.append(
                {
                    "component_id": (
                        f"{args.component_prefix}-ND-{component_index:05d}"
                    ),
                    "image_name": str(indexed_audit.at[audit_id, "image_name"]),
                    "reason": reason,
                    "retained_audit_id": retained,
                }
            )

    excluded_audit_ids = exact_excluded | near_excluded
    excluded_image_names = {
        str(indexed_audit.at[audit_id, "image_name"])
        for audit_id in excluded_audit_ids
    }
    curated_additional = source_records.loc[
        ~source_records["image_name"].isin(excluded_image_names)
    ].copy()
    combined_columns = sorted(set(base.columns) | set(curated_additional.columns))
    combined = pd.concat(
        [
            base.reindex(columns=combined_columns),
            curated_additional.reindex(columns=combined_columns),
        ],
        ignore_index=True,
    ).sort_values("image_name", kind="mergesort")
    if combined["image_name"].duplicated().any() or combined["sha256"].duplicated().any():
        raise RuntimeError("Exact duplicates remain in augmented manifest")
    final_pairs = find_near_duplicate_pairs(combined, threshold=threshold)
    if not final_pairs.empty and (~final_pairs["same_group"]).any():
        raise RuntimeError("Cross-group near duplicates remain after curation")
    split_by_image = combined.set_index("image_name")["split"]
    if not final_pairs.empty:
        left_split = final_pairs["left_image_name"].map(split_by_image)
        right_split = final_pairs["right_image_name"].map(split_by_image)
        if (left_split != right_split).any():
            raise RuntimeError("Near duplicates cross splits after curation")
        final_pairs["left_split"] = left_split
        final_pairs["right_split"] = right_split
    else:
        final_pairs["left_split"] = pd.Series(dtype="string")
        final_pairs["right_split"] = pd.Series(dtype="string")

    output.parent.mkdir(parents=True, exist_ok=True)
    digest = write_manifest(combined, output)
    final_pairs.reindex(
        columns=[*NEAR_DUPLICATE_COLUMNS, "left_split", "right_split"]
    ).to_csv(
        near_duplicates_path,
        index=False,
        lineterminator="\n",
    )
    exclusions = pd.DataFrame.from_records(component_records)
    exact_records = [
        {
            "component_id": "exact",
            "image_name": str(indexed_audit.at[audit_id, "image_name"]),
            "reason": "exact_duplicate",
            "retained_audit_id": "",
        }
        for audit_id in sorted(exact_excluded)
    ]
    exclusions = pd.concat(
        [pd.DataFrame.from_records(exact_records), exclusions],
        ignore_index=True,
    )
    exclusions.reindex(
        columns=("component_id", "image_name", "reason", "retained_audit_id")
    ).to_csv(exclusions_path, index=False, lineterminator="\n")
    report = {
        "schema_version": 1,
        "status": "complete",
        "policy": (
            "train-only, patient-disjoint from base; exclude exact or pHash "
            "near duplicates of base, external, and demo images; retain one "
            "same-label representative for new-only near-duplicate components"
        ),
        "source_hashes": protected_hashes,
        "near_duplicate_hamming_threshold": threshold,
        "downloaded_records": int(len(metadata)),
        "excluded_base_patient_records": int(excluded_patient_overlap.sum()),
        "scanned_source_records": int(len(source_records)),
        "exact_duplicate_exclusions": int(len(exact_excluded)),
        "near_duplicate_exclusions": int(len(near_excluded)),
        "additional_records": int(len(curated_additional)),
        "additional_patients": int(curated_additional["patient_id"].nunique()),
        "additional_target_counts": {
            str(int(target)): int(count)
            for target, count in curated_additional["target"]
            .value_counts()
            .sort_index()
            .items()
        },
        "final_records": int(len(combined)),
        "final_split_counts": {
            str(split): int(count)
            for split, count in combined["split"].value_counts().items()
        },
        "final_manifest_sha256": digest,
        "split_manifest_sha256": digest,
        "final_near_duplicate_pairs": int(len(final_pairs)),
        "near_duplicate_pairs": int(len(final_pairs)),
        "cross_group_near_duplicate_pairs": 0,
        "split_leakage_records": 0,
        "group_leakage_records": 0,
        "hash_leakage_records": 0,
        "near_duplicate_split_leakage_records": 0,
        "splits": {
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
            for split_frame in [combined.loc[combined["split"].eq(split)]]
        },
        "training_authorized": True,
        "research_only": True,
    }
    write_json_exclusive(report_path, report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
