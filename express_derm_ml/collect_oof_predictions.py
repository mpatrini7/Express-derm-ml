from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import require_absent, write_json_exclusive, write_npz_exclusive
from .manifest import canonical_manifest_sha256, read_manifest
from .oof import canonical_id_set_sha256


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_oof_predictions(
    *,
    plan_path: str | Path,
    run_dirs: list[str | Path],
    output_path: str | Path,
    provenance_path: str | Path,
) -> dict[str, Any]:
    plan_path = Path(plan_path)
    output_path = Path(output_path)
    provenance_path = Path(provenance_path)
    require_absent([output_path, provenance_path])
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
    }
    missing_plan = required_plan - set(plan.columns)
    if missing_plan:
        raise ValueError(f"OOF plan is missing columns: {sorted(missing_plan)}")
    expected_folds = set(plan["oof_fold"].astype(str))
    if len(run_dirs) != len(expected_folds):
        raise ValueError("Exactly one completed run directory is required per OOF fold")

    all_groups = set(plan["group_id"].astype(str))
    prediction_frames = []
    fold_receipts = []
    observed_folds: set[str] = set()
    for value in run_dirs:
        run_dir = Path(value)
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        run_manifest_path = run_dir / "manifest.csv"
        run_manifest = read_manifest(
            run_manifest_path, additional_text_columns=("oof_fold",)
        )
        run_manifest_sha256 = canonical_manifest_sha256(run_manifest)
        if run.get("manifest_sha256") != run_manifest_sha256:
            raise ValueError(f"Run manifest hash mismatch: {run_dir}")
        if Path(f"{run_manifest_path}.sha256").read_text().strip() != (
            run_manifest_sha256
        ):
            raise ValueError(f"Run manifest digest sidecar mismatch: {run_dir}")
        checkpoint_path = run_dir / "best.pt"
        checkpoint_sha256 = _sha256_file(checkpoint_path)
        if run.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError(f"Run checkpoint hash mismatch: {run_dir}")
        validation = run_manifest.loc[
            run_manifest["split"].astype(str).eq("validation")
        ].copy()
        validation_folds = set(validation["oof_fold"].dropna().astype(str))
        if len(validation_folds) != 1:
            raise ValueError(f"Run validation must represent one OOF fold: {run_dir}")
        fold_id = next(iter(validation_folds))
        if fold_id not in expected_folds or fold_id in observed_folds:
            raise ValueError(f"Duplicate or unknown OOF run fold: {fold_id}")
        observed_folds.add(fold_id)
        planned_fold = plan.loc[plan["oof_fold"].astype(str).eq(fold_id)]
        if set(validation["image_name"].astype(str)) != set(
            planned_fold["image_name"].astype(str)
        ):
            raise ValueError(f"Run validation records differ from plan: {fold_id}")
        training_groups = set(
            run_manifest.loc[
                run_manifest["split"].astype(str).eq("train"), "group_id"
            ].astype(str)
        )
        held_out_groups = set(planned_fold["group_id"].astype(str))
        if training_groups != all_groups - held_out_groups:
            raise ValueError(f"Run training groups are not the OOF complement: {fold_id}")
        training_records = set(
            run_manifest.loc[
                run_manifest["split"].astype(str).eq("train"), "image_name"
            ].astype(str)
        )
        expected_training_records = set(
            plan.loc[
                ~plan["oof_fold"].astype(str).eq(fold_id), "image_name"
            ].astype(str)
        )
        if training_records != expected_training_records:
            raise ValueError(
                f"Run training records are not the OOF complement: {fold_id}"
            )

        predictions_path = run_dir / "predictions_validation.npz"
        predictions_sha256 = _sha256_file(predictions_path)
        with np.load(predictions_path, allow_pickle=False) as arrays:
            required_keys = {
                "uncalibrated_probabilities",
                "targets",
                "image_names",
            }
            if not required_keys.issubset(arrays.files):
                raise ValueError(
                    f"Run validation predictions are incomplete: {run_dir}"
                )
            prediction_frame = pd.DataFrame(
                {
                    "image_name": np.asarray(arrays["image_names"], dtype=str),
                    "target": np.asarray(arrays["targets"], dtype=np.int64),
                    "uncalibrated_probability": np.asarray(
                        arrays["uncalibrated_probabilities"], dtype=np.float64
                    ),
                }
            )
        if prediction_frame["image_name"].duplicated().any():
            raise ValueError(f"Duplicate validation predictions: {run_dir}")
        aligned = planned_fold.set_index("image_name").join(
            prediction_frame.set_index("image_name"),
            how="left",
            rsuffix="_prediction",
            validate="one_to_one",
        )
        if aligned["uncalibrated_probability"].isna().any():
            raise ValueError(f"Missing planned predictions: {fold_id}")
        if not np.array_equal(
            aligned["target"].astype(int).to_numpy(),
            aligned["target_prediction"].astype(int).to_numpy(),
        ):
            raise ValueError(f"Prediction targets differ from plan: {fold_id}")
        aligned = aligned.reset_index()
        aligned["fold_id"] = fold_id
        prediction_frames.append(aligned)
        calibration_path = run_dir / "calibration.json"
        fold_receipts.append(
            {
                "fold_id": fold_id,
                "run_dir": str(run_dir),
                "run_manifest_sha256": run_manifest_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "validation_predictions_sha256": predictions_sha256,
                "calibration_sha256": (
                    _sha256_file(calibration_path)
                    if calibration_path.is_file()
                    else None
                ),
                "held_out_group_ids_sha256": canonical_id_set_sha256(
                    held_out_groups
                ),
                "training_group_ids_sha256": canonical_id_set_sha256(
                    training_groups
                ),
                "held_out_records": int(len(aligned)),
            }
        )
    if observed_folds != expected_folds:
        raise RuntimeError("OOF run coverage is incomplete")
    combined = pd.concat(prediction_frames, ignore_index=True).sort_values(
        "image_name", kind="mergesort"
    )
    if len(combined) != len(plan) or combined["image_name"].duplicated().any():
        raise RuntimeError("Collected OOF predictions do not cover the plan once")
    probabilities = combined["uncalibrated_probability"].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0.0) | (probabilities > 1.0)
    ).any():
        raise ValueError("Collected OOF probabilities must be finite and in [0, 1]")
    write_npz_exclusive(
        output_path,
        uncalibrated_probabilities=probabilities,
        targets=combined["target"].astype(int).to_numpy(),
        image_names=combined["image_name"].to_numpy(dtype=str),
        group_ids=combined["group_id"].to_numpy(dtype=str),
        fold_ids=combined["fold_id"].to_numpy(dtype=str),
    )
    output_sha256 = _sha256_file(output_path)
    provenance = {
        "schema_version": 1,
        "status": "complete",
        "evaluation_role": "out_of_fold_training",
        "score_kind": "uncalibrated_sigmoid_probability",
        "probability_key": "uncalibrated_probabilities",
        "score_fit_uses_oof_labels": False,
        "plan_sha256": plan_sha256,
        "predictions_sha256": output_sha256,
        "records": int(len(combined)),
        "groups": int(combined["group_id"].nunique()),
        "folds": sorted(fold_receipts, key=lambda item: item["fold_id"]),
        "test_or_external_records_used": 0,
        "training_authorized": True,
        "research_only": True,
    }
    write_json_exclusive(provenance_path, provenance)
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect one model-only validation prediction set per OOF fold."
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provenance", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = collect_oof_predictions(
        plan_path=args.plan,
        run_dirs=args.run_dir,
        output_path=args.output,
        provenance_path=args.provenance,
    )
    print(report)


if __name__ == "__main__":
    main()
