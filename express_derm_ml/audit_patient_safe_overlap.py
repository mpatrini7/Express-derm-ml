from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import write_json_exclusive
from .common import sha256_file


def _patient_safe_records(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"image_name", "patient_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    if frame["image_name"].duplicated().any():
        raise ValueError(f"Metadata contains duplicate image IDs: {path}")
    return frame.loc[frame["patient_id"].str.strip().ne("")].copy()


def audit_patient_safe_overlap(
    *,
    reference_path: str | Path,
    candidate_paths: list[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    reference_file = Path(reference_path)
    reference = _patient_safe_records(reference_file)
    reference_images = set(reference["image_name"])
    candidates = []
    for value in candidate_paths:
        candidate_file = Path(value)
        candidate = _patient_safe_records(candidate_file)
        candidate_images = set(candidate["image_name"])
        candidates.append(
            {
                "path": str(candidate_file),
                "sha256": sha256_file(candidate_file),
                "patient_safe_records": int(len(candidate)),
                "patients": int(candidate["patient_id"].nunique()),
                "image_id_overlap_with_reference": int(
                    len(candidate_images & reference_images)
                ),
                "unique_image_ids_not_in_reference": int(
                    len(candidate_images - reference_images)
                ),
            }
        )
    report = {
        "schema_version": 1,
        "purpose": "patient_safe_cross_collection_overlap_audit",
        "reference": {
            "path": str(reference_file),
            "sha256": sha256_file(reference_file),
            "patient_safe_records": int(len(reference)),
            "patients": int(reference["patient_id"].nunique()),
        },
        "candidates": candidates,
    }
    write_json_exclusive(output_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit patient-safe image-ID overlap between ISIC metadata snapshots."
        )
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_patient_safe_overlap(
        reference_path=args.reference,
        candidate_paths=args.candidate,
        output_path=args.output,
    )
    print(report)


if __name__ == "__main__":
    main()
