from __future__ import annotations

import argparse
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
from .oof import canonical_id_set_sha256, deterministic_group_order


def create_oof_plan(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    folds: int,
    seed: int,
    source_column: str = "collection_id",
) -> dict[str, Any]:
    if folds < 2:
        raise ValueError("OOF planning requires at least two folds")
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    report_path = artifact_path(output_path, "report.json")
    require_absent([output_path, Path(f"{output_path}.sha256"), report_path])
    manifest = read_manifest(manifest_path)
    expected_digest_path = Path(f"{manifest_path}.sha256")
    if not expected_digest_path.is_file():
        raise ValueError("Input manifest digest is missing")
    source_manifest_sha256 = canonical_manifest_sha256(manifest)
    if expected_digest_path.read_text(encoding="ascii").strip() != (
        source_manifest_sha256
    ):
        raise ValueError("Input manifest digest mismatch")
    required = {
        "image_name",
        "group_id",
        "split",
        "target",
        source_column,
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Input manifest is missing columns: {sorted(missing)}")
    training = manifest.loc[manifest["split"].astype(str).eq("train")].copy()
    if training.empty:
        raise ValueError("Input manifest has no training records")
    training["target"] = pd.to_numeric(training["target"], errors="raise").astype(int)
    if not training["target"].isin([0, 1]).all():
        raise ValueError("OOF targets must be binary")
    for column in ("image_name", "group_id", source_column):
        training[column] = training[column].astype(str)
        if training[column].str.strip().eq("").any():
            raise ValueError(f"OOF planning column cannot be blank: {column}")
    if training["image_name"].duplicated().any():
        raise ValueError("OOF record IDs must be unique")

    group_summary = training.groupby("group_id", sort=True).agg(
        source_count=(source_column, "nunique"),
        source_id=(source_column, "first"),
        group_target=("target", "max"),
        records=("image_name", "size"),
    )
    if group_summary["source_count"].ne(1).any():
        raise ValueError("A patient group cannot span multiple source datasets")

    assignment: dict[str, str] = {}
    sparse_strata: list[dict[str, Any]] = []
    for (source_id, group_target), stratum in group_summary.groupby(
        ["source_id", "group_target"], sort=True
    ):
        groups = sorted(
            stratum.index.astype(str),
            key=lambda group_id: deterministic_group_order(group_id, seed),
        )
        if len(groups) < folds:
            sparse_strata.append(
                {
                    "source_id": str(source_id),
                    "group_target": int(group_target),
                    "groups": len(groups),
                    "folds": folds,
                }
            )
            continue
        for index, group_id in enumerate(groups):
            assignment[group_id] = f"fold_{index % folds}"
    if sparse_strata:
        raise ValueError(
            "Every source/target stratum must have at least one group per fold: "
            f"{sparse_strata}"
        )

    training["oof_fold"] = training["group_id"].map(assignment)
    if training["oof_fold"].isna().any():
        raise RuntimeError("OOF assignment is incomplete")
    if training.groupby("group_id")["oof_fold"].nunique().gt(1).any():
        raise RuntimeError("OOF patient-group leakage detected")
    training = training.sort_values("image_name", kind="mergesort")
    plan_sha256 = write_manifest(training, output_path)
    all_groups = set(training["group_id"].astype(str))
    fold_reports = []
    for fold_id in sorted(training["oof_fold"].unique()):
        held_out = set(
            training.loc[training["oof_fold"].eq(fold_id), "group_id"].astype(str)
        )
        training_groups = all_groups - held_out
        fold_frame = training.loc[training["oof_fold"].eq(fold_id)]
        fold_reports.append(
            {
                "fold_id": str(fold_id),
                "held_out_records": int(len(fold_frame)),
                "held_out_groups": int(len(held_out)),
                "held_out_group_ids_sha256": canonical_id_set_sha256(held_out),
                "training_groups": int(len(training_groups)),
                "training_group_ids_sha256": canonical_id_set_sha256(
                    training_groups
                ),
                "source_target_records": {
                    f"{source}|{int(target)}": int(count)
                    for (source, target), count in fold_frame.groupby(
                        [source_column, "target"]
                    ).size().sort_index().items()
                },
            }
        )
    report = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "patient_group_safe_out_of_fold_plan",
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": source_manifest_sha256,
        "plan_sha256": plan_sha256,
        "seed": int(seed),
        "fold_count": int(folds),
        "source_column": source_column,
        "records": int(len(training)),
        "groups": int(training["group_id"].nunique()),
        "patient_group_leakage_records": 0,
        "folds": fold_reports,
        "research_only": True,
    }
    write_json_exclusive(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic, source-stratified patient-safe OOF folds."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--source-column", default="collection_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = create_oof_plan(
        manifest_path=args.manifest,
        output_path=args.output,
        folds=args.folds,
        seed=args.seed,
        source_column=args.source_column,
    )
    print(report)


if __name__ == "__main__":
    main()
