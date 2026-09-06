from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import require_absent, write_json_exclusive
from .common import load_yaml, sha256_file


CONFIRMATION_RANK = {
    "histopathology": 0,
    "serial imaging showing no change": 1,
    "single image expert consensus": 2,
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


def curate_dicm17k_near_duplicates(
    *,
    metadata_path: str | Path,
    near_duplicates_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    metadata_source = Path(metadata_path)
    candidates_source = Path(near_duplicates_path)
    config_file = Path(config_path)
    output = Path(output_path)
    exclusions_path = output.with_name(f"{output.stem}.exclusions.csv")
    receipt_path = output.with_name(f"{output.stem}.receipt.json")
    require_absent([output, exclusions_path, receipt_path])

    metadata = pd.read_csv(metadata_source, keep_default_na=False)
    candidates = pd.read_csv(candidates_source, keep_default_na=False)
    required_candidate_columns = {
        "left_image_name",
        "right_image_name",
        "hamming_distance",
        "same_group",
    }
    missing = required_candidate_columns - set(candidates.columns)
    if missing:
        raise RuntimeError(
            f"Near-duplicate report is missing columns: {sorted(missing)}"
        )
    if metadata["image_name"].duplicated().any():
        raise RuntimeError("DICM-17K curated source image IDs must be unique")
    metadata_by_image = metadata.set_index("image_name", drop=False)
    candidate_images = set(candidates["left_image_name"]) | set(
        candidates["right_image_name"]
    )
    unknown = sorted(candidate_images - set(metadata_by_image.index))
    if unknown:
        raise RuntimeError(
            f"Near-duplicate report references unknown images: {unknown[:5]}"
        )

    components = _Components()
    for row in candidates.itertuples(index=False):
        components.union(str(row.left_image_name), str(row.right_image_name))

    excluded_records: list[dict[str, Any]] = []
    conflict_components = 0
    for component_index, image_names in enumerate(
        components.groups(),
        start=1,
    ):
        component = metadata_by_image.loc[image_names]
        labels = sorted(set(component["diagnosis_class"]))
        if len(labels) != 1:
            conflict_components += 1
            retained = ""
            reason = "near_duplicate_component_conflicting_labels"
            excluded_names = image_names
        else:
            ranked = sorted(
                image_names,
                key=lambda image_name: (
                    CONFIRMATION_RANK[
                        str(
                            metadata_by_image.at[
                                image_name, "diagnosis_confirm_type"
                            ]
                        )
                    ],
                    0
                    if str(metadata_by_image.at[image_name, "lesion_id"])
                    else 1,
                    image_name,
                ),
            )
            retained = ranked[0]
            reason = "near_duplicate_component_redundant"
            excluded_names = ranked[1:]
        component_id = f"DICM-ND-{component_index:04d}"
        for image_name in excluded_names:
            row = metadata_by_image.loc[image_name]
            excluded_records.append(
                {
                    "component_id": component_id,
                    "image_name": image_name,
                    "patient_id": str(row["patient_id"]),
                    "diagnosis_class": str(row["diagnosis_class"]),
                    "diagnosis_confirm_type": str(
                        row["diagnosis_confirm_type"]
                    ),
                    "retained_image_name": retained,
                    "reason": reason,
                }
            )

    exclusions = pd.DataFrame.from_records(excluded_records).sort_values(
        ["component_id", "image_name"],
        kind="mergesort",
    )
    excluded_images = set(exclusions["image_name"])
    curated = metadata.loc[
        ~metadata["image_name"].isin(excluded_images)
    ].sort_values("image_name", kind="mergesort")
    config = load_yaml(config_file)
    dataset = config["dataset"]
    if len(curated) != int(dataset["expected_curated_records"]):
        raise RuntimeError("Unexpected deduplicated DICM-17K record count")
    if curated["patient_id"].nunique() != int(
        dataset["expected_curated_patients"]
    ):
        raise RuntimeError("Unexpected deduplicated DICM-17K patient count")
    if curated["diagnosis_class"].eq("melanoma").sum() != int(
        dataset["expected_curated_melanoma"]
    ):
        raise RuntimeError("Unexpected deduplicated DICM-17K melanoma count")
    if not candidates.loc[
        candidates["left_image_name"].isin(set(curated["image_name"]))
        & candidates["right_image_name"].isin(set(curated["image_name"]))
    ].empty:
        raise RuntimeError("Near-duplicate pairs remain after DICM curation")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        curated.to_csv(destination, index=False, lineterminator="\n")
    with exclusions_path.open("x", encoding="utf-8") as destination:
        exclusions.to_csv(destination, index=False, lineterminator="\n")
    receipt = {
        "schema_version": 1,
        "policy": (
            "exclude conflicting-label components; otherwise retain the "
            "strongest confirmation, then a specified lesion ID, then the "
            "lexicographically first image ID"
        ),
        "metadata_sha256": sha256_file(metadata_source),
        "near_duplicate_report_sha256": sha256_file(candidates_source),
        "config_sha256": sha256_file(config_file),
        "candidate_pairs": int(len(candidates)),
        "candidate_images": int(len(candidate_images)),
        "components": int(len(components.groups())),
        "conflicting_label_components": conflict_components,
        "excluded_records": int(len(exclusions)),
        "retained_records": int(len(curated)),
        "retained_patients": int(curated["patient_id"].nunique()),
        "retained_target_counts": {
            str(label): int(count)
            for label, count in curated["diagnosis_class"]
            .value_counts()
            .sort_index()
            .items()
        },
        "curated_metadata_sha256": sha256_file(output),
        "exclusions_sha256": sha256_file(exclusions_path),
        "training_authorized": False,
        "status": "deduplicated_pending_manifest_curation",
    }
    write_json_exclusive(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--near-duplicates", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = curate_dicm17k_near_duplicates(
        metadata_path=args.metadata,
        near_duplicates_path=args.near_duplicates,
        config_path=args.config,
        output_path=args.output,
    )
    print(result)


if __name__ == "__main__":
    main()
