from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import require_absent, write_json_exclusive
from .manifest import (
    artifact_path,
    canonical_manifest_sha256,
    read_manifest,
    write_manifest,
)


def materialize_oof_fold(
    *,
    manifest_path: str | Path,
    plan_path: str | Path,
    fold_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    plan_path = Path(plan_path)
    output_path = Path(output_path)
    report_path = artifact_path(output_path, "report.json")
    near_duplicate_path = artifact_path(output_path, "near_duplicates.csv")
    require_absent(
        [
            output_path,
            Path(f"{output_path}.sha256"),
            report_path,
            near_duplicate_path,
        ]
    )
    manifest = read_manifest(manifest_path)
    source_manifest_sha256 = canonical_manifest_sha256(manifest)
    if Path(f"{manifest_path}.sha256").read_text(encoding="ascii").strip() != (
        source_manifest_sha256
    ):
        raise ValueError("Source manifest digest mismatch")
    plan = read_manifest(plan_path, additional_text_columns=("oof_fold",))
    plan_sha256 = canonical_manifest_sha256(plan)
    if Path(f"{plan_path}.sha256").read_text(encoding="ascii").strip() != (
        plan_sha256
    ):
        raise ValueError("OOF plan digest mismatch")
    plan_report_path = artifact_path(plan_path, "report.json")
    plan_report = json.loads(plan_report_path.read_text(encoding="utf-8"))
    expected_plan = {
        "source_manifest_sha256": source_manifest_sha256,
        "plan_sha256": plan_sha256,
        "purpose": "patient_group_safe_out_of_fold_plan",
    }
    for field, expected in expected_plan.items():
        if plan_report.get(field) != expected:
            raise ValueError(f"OOF plan report mismatch: {field}")
    available_folds = set(plan["oof_fold"].astype(str))
    if fold_id not in available_folds:
        raise ValueError(
            f"Unknown OOF fold {fold_id!r}; expected {sorted(available_folds)}"
        )
    source_train = manifest.loc[manifest["split"].astype(str).eq("train")]
    if set(source_train["image_name"].astype(str)) != set(
        plan["image_name"].astype(str)
    ):
        raise ValueError("OOF plan does not exactly cover source training records")
    if set(source_train["group_id"].astype(str)) != set(
        plan["group_id"].astype(str)
    ):
        raise ValueError("OOF plan training groups differ from source manifest")

    plan_columns = ["image_name", "oof_fold"]
    planned = source_train.merge(
        plan.loc[:, plan_columns],
        on="image_name",
        how="left",
        validate="one_to_one",
    )
    planned["split"] = "train"
    planned.loc[planned["oof_fold"].astype(str).eq(fold_id), "split"] = (
        "validation"
    )
    frozen_test = manifest.loc[manifest["split"].astype(str).eq("test")].copy()
    if frozen_test.empty:
        raise ValueError("Source manifest must retain a frozen test split")
    frozen_test["oof_fold"] = ""
    columns = sorted(set(planned.columns) | set(frozen_test.columns))
    derived = pd.concat(
        [planned.reindex(columns=columns), frozen_test.reindex(columns=columns)],
        ignore_index=True,
    ).sort_values("image_name", kind="mergesort")
    if set(derived["split"].astype(str)) != {"train", "validation", "test"}:
        raise RuntimeError("Derived OOF manifest is missing a required split")
    if derived.groupby("group_id")["split"].nunique().gt(1).any():
        raise RuntimeError("Patient-group leakage in derived OOF manifest")
    if derived["image_name"].duplicated().any() or derived["sha256"].duplicated().any():
        raise RuntimeError("Duplicate records in derived OOF manifest")
    split_target_counts = derived.groupby("split")["target"].nunique()
    if not split_target_counts.reindex(["train", "validation", "test"]).eq(2).all():
        raise ValueError("Every derived OOF split must contain both targets")

    source_near_path = artifact_path(manifest_path, "near_duplicates.csv")
    source_near = pd.read_csv(source_near_path)
    retained_ids = set(derived["image_name"].astype(str))
    if source_near.empty:
        near_duplicates = source_near.copy()
    else:
        required_near = {"left_image_name", "right_image_name", "same_group"}
        if not required_near.issubset(source_near.columns):
            raise ValueError("Source near-duplicate report lacks required evidence")
        near_duplicates = source_near.loc[
            source_near["left_image_name"].astype(str).isin(retained_ids)
            & source_near["right_image_name"].astype(str).isin(retained_ids)
        ].copy()
        split_by_image = derived.set_index("image_name")["split"].astype(str)
        near_duplicates["left_split"] = near_duplicates[
            "left_image_name"
        ].map(split_by_image)
        near_duplicates["right_split"] = near_duplicates[
            "right_image_name"
        ].map(split_by_image)
        if near_duplicates["left_split"].ne(
            near_duplicates["right_split"]
        ).any():
            raise ValueError("Near-duplicate candidates cross derived OOF splits")
        same_group = (
            near_duplicates["same_group"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("true")
        )
        if not same_group.all():
            raise ValueError("Cross-group near duplicates remain in OOF fold")
    near_duplicates.to_csv(near_duplicate_path, index=False, lineterminator="\n")
    derived_sha256 = write_manifest(derived, output_path)
    split_summary = {
        split: {
            "records": int(len(frame)),
            "groups": int(frame["group_id"].nunique()),
            "target_counts": {
                str(int(target)): int(count)
                for target, count in frame["target"]
                .value_counts()
                .sort_index()
                .items()
            },
        }
        for split in ("train", "validation", "test")
        for frame in [derived.loc[derived["split"].astype(str).eq(split)]]
    }
    report = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "oof_fold_training_manifest",
        "fold_id": fold_id,
        "source_manifest_sha256": source_manifest_sha256,
        "oof_plan_sha256": plan_sha256,
        "split_manifest_sha256": derived_sha256,
        "dropped_source_validation_records": int(
            manifest["split"].astype(str).eq("validation").sum()
        ),
        "splits": split_summary,
        "group_leakage_records": 0,
        "hash_leakage_records": 0,
        "near_duplicate_pairs": int(len(near_duplicates)),
        "near_duplicate_split_leakage_records": 0,
        "training_authorized": True,
        "research_only": True,
    }
    write_json_exclusive(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize one OOF fold as an immutable train/validation/test manifest."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = materialize_oof_fold(
        manifest_path=args.manifest,
        plan_path=args.plan,
        fold_id=args.fold_id,
        output_path=args.output,
    )
    print(report)


if __name__ == "__main__":
    main()
