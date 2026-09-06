from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import require_absent, write_json_exclusive
from .manifest import artifact_path, canonical_manifest_sha256, read_manifest
from .oof import canonical_id_set_sha256


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return bool(SHA256_PATTERN.fullmatch(str(value).strip().lower()))


def mine_oof_errors(
    *,
    plan_path: str | Path,
    predictions_path: str | Path,
    provenance_path: str | Path,
    output_path: str | Path,
    low_threshold: float,
    high_threshold: float,
    max_per_error_source: int,
    probability_key: str = "uncalibrated_probabilities",
) -> dict[str, Any]:
    if not 0.0 <= low_threshold < high_threshold <= 1.0:
        raise ValueError("OOF mining thresholds must satisfy 0 <= low < high <= 1")
    if max_per_error_source < 1:
        raise ValueError("OOF mining quota must be positive")
    if not probability_key.strip():
        raise ValueError("OOF probability key cannot be blank")
    plan_path = Path(plan_path)
    predictions_path = Path(predictions_path)
    provenance_path = Path(provenance_path)
    output_path = Path(output_path)
    report_path = artifact_path(output_path, "report.json")
    digest_path = Path(f"{output_path}.sha256")
    require_absent([output_path, report_path, digest_path])

    plan = read_manifest(plan_path, additional_text_columns=("oof_fold",))
    plan_sha256 = canonical_manifest_sha256(plan)
    if Path(f"{plan_path}.sha256").read_text(encoding="ascii").strip() != plan_sha256:
        raise ValueError("OOF plan digest mismatch")
    required_plan = {
        "image_name",
        "group_id",
        "target",
        "oof_fold",
        "collection_id",
        "sha256",
        "image_path",
    }
    missing_plan = required_plan - set(plan.columns)
    if missing_plan:
        raise ValueError(f"OOF plan is missing columns: {sorted(missing_plan)}")

    predictions_sha256 = _sha256_file(predictions_path)
    provenance_sha256 = _sha256_file(provenance_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("schema_version") != 1:
        raise ValueError("OOF provenance must use schema_version 1")
    if provenance.get("evaluation_role") != "out_of_fold_training":
        raise ValueError("Only out-of-fold training predictions may be mined")
    if provenance.get("probability_key") != probability_key:
        raise ValueError("OOF provenance probability key mismatch")
    if provenance.get("score_fit_uses_oof_labels") is not False:
        raise ValueError("OOF mining requires model-only scores not fit on OOF labels")
    expected_values = {
        "plan_sha256": plan_sha256,
        "predictions_sha256": predictions_sha256,
    }
    for field, expected in expected_values.items():
        if provenance.get(field) != expected:
            raise ValueError(f"OOF provenance mismatch: {field}")

    with np.load(predictions_path, allow_pickle=False) as arrays:
        required_keys = {
            probability_key,
            "targets",
            "image_names",
            "group_ids",
            "fold_ids",
        }
        missing_keys = required_keys - set(arrays.files)
        if missing_keys:
            raise ValueError(f"OOF predictions are missing {sorted(missing_keys)}")
        probabilities = np.asarray(
            arrays[probability_key], dtype=np.float64
        )
        targets = np.asarray(arrays["targets"], dtype=np.int64)
        image_names = np.asarray(arrays["image_names"], dtype=str)
        group_ids = np.asarray(arrays["group_ids"], dtype=str)
        fold_ids = np.asarray(arrays["fold_ids"], dtype=str)
    lengths = {
        len(probabilities),
        len(targets),
        len(image_names),
        len(group_ids),
        len(fold_ids),
    }
    if len(lengths) != 1 or not probabilities.ndim == targets.ndim == 1:
        raise ValueError("OOF prediction arrays must be aligned and flat")
    if len(image_names) != len(plan) or len(np.unique(image_names)) != len(plan):
        raise ValueError("OOF predictions must cover every plan record exactly once")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0.0) | (probabilities > 1.0)
    ).any():
        raise ValueError("OOF probabilities must be finite and in [0, 1]")

    predicted = pd.DataFrame(
        {
            "image_name": image_names,
            "prediction_target": targets,
            "prediction_group_id": group_ids,
            "prediction_fold": fold_ids,
            "calibrated_probability": probabilities,
        }
    ).set_index("image_name")
    aligned = plan.set_index("image_name").join(predicted, how="left", validate="one_to_one")
    if aligned["calibrated_probability"].isna().any():
        raise ValueError("OOF predictions do not align with the plan")
    if not np.array_equal(
        aligned["target"].astype(int).to_numpy(),
        aligned["prediction_target"].astype(int).to_numpy(),
    ):
        raise ValueError("OOF targets do not match the plan")
    if not aligned["group_id"].astype(str).eq(
        aligned["prediction_group_id"].astype(str)
    ).all():
        raise ValueError("OOF group IDs do not match the plan")
    if not aligned["oof_fold"].astype(str).eq(
        aligned["prediction_fold"].astype(str)
    ).all():
        raise ValueError("OOF fold IDs do not match the plan")

    fold_provenance = provenance.get("folds")
    if not isinstance(fold_provenance, list):
        raise ValueError("OOF provenance must contain fold evidence")
    by_fold = {str(item.get("fold_id")): item for item in fold_provenance}
    expected_folds = set(aligned["oof_fold"].astype(str))
    if set(by_fold) != expected_folds:
        raise ValueError("OOF provenance fold set mismatch")
    all_groups = set(aligned["group_id"].astype(str))
    checkpoint_hashes = []
    for fold_id in sorted(expected_folds):
        evidence = by_fold[fold_id]
        held_out = set(
            aligned.loc[
                aligned["oof_fold"].astype(str).eq(fold_id), "group_id"
            ].astype(str)
        )
        training_groups = all_groups - held_out
        expected = {
            "held_out_group_ids_sha256": canonical_id_set_sha256(held_out),
            "training_group_ids_sha256": canonical_id_set_sha256(training_groups),
        }
        for field, value in expected.items():
            if evidence.get(field) != value:
                raise ValueError(f"OOF fold evidence mismatch: {fold_id}.{field}")
        checkpoint_sha256 = str(evidence.get("checkpoint_sha256", ""))
        if not _valid_sha256(checkpoint_sha256):
            raise ValueError(f"OOF fold checkpoint hash is invalid: {fold_id}")
        checkpoint_hashes.append(checkpoint_sha256)

    aligned["error_type"] = ""
    aligned.loc[
        aligned["target"].astype(int).eq(0)
        & aligned["calibrated_probability"].ge(high_threshold),
        "error_type",
    ] = "hard_false_positive"
    aligned.loc[
        aligned["target"].astype(int).eq(1)
        & aligned["calibrated_probability"].lt(low_threshold),
        "error_type",
    ] = "hard_false_negative"
    candidates = aligned.loc[aligned["error_type"].ne("")].reset_index()
    candidates["hardness"] = np.where(
        candidates["error_type"].eq("hard_false_positive"),
        candidates["calibrated_probability"],
        1.0 - candidates["calibrated_probability"],
    )
    candidates = candidates.sort_values(
        ["error_type", "collection_id", "hardness", "image_name"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    candidates["source_error_rank"] = (
        candidates.groupby(["error_type", "collection_id"]).cumcount() + 1
    )
    selected = candidates.loc[
        candidates["source_error_rank"].le(max_per_error_source)
    ].copy()
    selected_columns = [
        "image_name",
        "image_path",
        "sha256",
        "group_id",
        "collection_id",
        "target",
        "oof_fold",
        "calibrated_probability",
        "error_type",
        "hardness",
        "source_error_rank",
    ]
    selected.loc[:, selected_columns].to_csv(
        output_path, index=False, lineterminator="\n"
    )
    output_sha256 = _sha256_file(output_path)
    digest_path.write_text(f"{output_sha256}\n", encoding="ascii")
    counts = {
        f"{error_type}|{source_id}": int(count)
        for (error_type, source_id), count in selected.groupby(
            ["error_type", "collection_id"]
        ).size().sort_index().items()
    }
    report = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "out_of_fold_hard_error_mining",
        "evaluation_role": "out_of_fold_training",
        "plan_sha256": plan_sha256,
        "predictions_sha256": predictions_sha256,
        "provenance_sha256": provenance_sha256,
        "fold_checkpoint_sha256": sorted(checkpoint_hashes),
        "low_threshold": float(low_threshold),
        "high_threshold": float(high_threshold),
        "probability_key": probability_key,
        "score_kind": provenance.get("score_kind"),
        "max_per_error_source": int(max_per_error_source),
        "candidate_records_before_quota": int(len(candidates)),
        "selected_records": int(len(selected)),
        "selected_counts": counts,
        "output_sha256": output_sha256,
        "training_authorized": True,
        "test_or_external_records_used": 0,
        "research_only": True,
    }
    write_json_exclusive(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mine hard errors only from provenance-verified out-of-fold "
            "predictions of training-authorized records."
        )
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--low-threshold", required=True, type=float)
    parser.add_argument("--high-threshold", required=True, type=float)
    parser.add_argument("--max-per-error-source", type=int, default=100)
    parser.add_argument(
        "--probability-key", default="uncalibrated_probabilities"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = mine_oof_errors(
        plan_path=args.plan,
        predictions_path=args.predictions,
        provenance_path=args.provenance,
        output_path=args.output,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        max_per_error_source=args.max_per_error_source,
        probability_key=args.probability_key,
    )
    print(report)


if __name__ == "__main__":
    main()
