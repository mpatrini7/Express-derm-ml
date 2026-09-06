from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from express_derm_ml.create_oof_plan import create_oof_plan
from express_derm_ml.collect_oof_predictions import collect_oof_predictions
from express_derm_ml.generalization_gate import evaluate_generalization_config
from express_derm_ml.manifest import read_manifest, write_manifest
from express_derm_ml.materialize_oof_fold import materialize_oof_fold
from express_derm_ml.mine_oof_errors import mine_oof_errors
from express_derm_ml.oof import canonical_id_set_sha256
from express_derm_ml.train import (
    SourceClassBalancedSampler,
    load_priority_record_indices,
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generalization_gate_uses_frozen_threshold_and_loso_source(tmp_path: Path) -> None:
    internal_predictions = tmp_path / "internal.npz"
    np.savez_compressed(
        internal_predictions,
        calibrated_probabilities=np.array([0.9, 0.8, 0.2, 0.1]),
        targets=np.array([1, 1, 0, 0]),
        image_names=np.array(["i1", "i2", "i3", "i4"]),
    )
    internal_manifest = tmp_path / "internal.csv"
    pd.DataFrame(
        {
            "image_name": ["i1", "i2", "i3", "i4"],
            "group_id": ["p1", "p2", "p3", "p4"],
            "target": [1, 1, 0, 0],
        }
    ).to_csv(internal_manifest, index=False)
    external_predictions = tmp_path / "external.npz"
    np.savez_compressed(
        external_predictions,
        calibrated_probabilities=np.array([0.8, 0.7, 0.6, 0.55]),
        melanoma_targets=np.array([1, 1, 0, 0]),
        image_names=np.array(["e1", "e2", "e3", "e4"]),
        lesion_ids=np.array(["l1", "l2", "l3", "l4"]),
    )
    config = {
        "schema_version": 1,
        "candidate": {
            "model_version": "candidate-1",
            "checkpoint_sha256": "a" * 64,
            "calibration_sha256": "b" * 64,
            "frozen_high_threshold": 0.5,
            "training_source_ids": ["source-a"],
        },
        "datasets": [
            {
                "id": "internal",
                "source_id": "source-a",
                "modality": "dermoscopic",
                "evaluation_protocol": "within_source_test",
                "training_authorized": False,
                "predictions_path": str(internal_predictions),
                "predictions_sha256": _file_sha256(internal_predictions),
                "group_lookup": {
                    "manifest_path": str(internal_manifest),
                    "manifest_file_sha256": _file_sha256(internal_manifest),
                },
            },
            {
                "id": "external",
                "source_id": "source-b",
                "modality": "dermoscopic",
                "evaluation_protocol": "leave_one_source_out",
                "training_authorized": False,
                "predictions_path": str(external_predictions),
                "predictions_sha256": _file_sha256(external_predictions),
                "target_key": "melanoma_targets",
                "group_id_key": "lesion_ids",
            },
        ],
        "bootstrap": {"samples": 100, "confidence_level": 0.9, "seed": 7},
        "research_gate": {
            "minimum_datasets": 2,
            "minimum_leave_one_source_out_datasets": 1,
            "minimum_worst_source": {
                "sensitivity": 0.9,
                "specificity": 0.75,
            },
            "maximum_source_gap": {"specificity": 0.2},
        },
    }
    config_path = tmp_path / "gate.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    report = evaluate_generalization_config(
        config_path, tmp_path / "report.json"
    )

    assert report["gate"]["status"] == "fail"
    assert report["gate"]["leave_one_source_out_datasets"] == 1
    assert report["evaluations"][0]["group_evidence"]["method"] == (
        "manifest_lookup"
    )
    assert any(
        blocker["criterion"] == "minimum_worst_source.specificity"
        for blocker in report["gate"]["blockers"]
    )


def _training_manifest() -> pd.DataFrame:
    rows = []
    counter = 0
    for source in ("source-a", "source-b"):
        for target in (0, 1):
            for group_index in range(2):
                counter += 1
                rows.append(
                    {
                        "image_name": f"image-{counter}",
                        "image_path": f"images/image-{counter}.jpg",
                        "group_id": f"{source}-target-{target}-group-{group_index}",
                        "patient_id": f"patient-{counter}",
                        "lesion_id": f"lesion-{counter}",
                        "collection_id": source,
                        "target": target,
                        "split": "train",
                        "sha256": f"{counter:064x}",
                    }
                )
    return pd.DataFrame(rows)


def _full_manifest() -> pd.DataFrame:
    frame = _training_manifest()
    extra = []
    for split, offset in (("validation", 100), ("test", 200)):
        for target in (0, 1):
            counter = offset + target
            extra.append(
                {
                    "image_name": f"image-{counter}",
                    "image_path": f"images/image-{counter}.jpg",
                    "group_id": f"{split}-group-{target}",
                    "patient_id": f"{split}-patient-{target}",
                    "lesion_id": f"{split}-lesion-{target}",
                    "collection_id": "source-a",
                    "target": target,
                    "split": split,
                    "sha256": f"{counter:064x}",
                }
            )
    return pd.concat([frame, pd.DataFrame(extra)], ignore_index=True)


def test_oof_plan_and_mining_require_fold_provenance(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(_training_manifest(), manifest_path)
    plan_path = tmp_path / "oof-plan.csv"
    plan_report = create_oof_plan(
        manifest_path=manifest_path,
        output_path=plan_path,
        folds=2,
        seed=13,
    )
    plan = read_manifest(plan_path, additional_text_columns=("oof_fold",))
    assert plan.groupby("group_id")["oof_fold"].nunique().max() == 1
    assert plan_report["patient_group_leakage_records"] == 0

    probabilities = np.where(
        plan["target"].astype(int).eq(0), 0.9, 0.1
    ).astype(float)
    predictions_path = tmp_path / "oof-predictions.npz"
    np.savez_compressed(
        predictions_path,
        uncalibrated_probabilities=probabilities,
        targets=plan["target"].astype(int).to_numpy(),
        image_names=plan["image_name"].to_numpy(dtype=str),
        group_ids=plan["group_id"].to_numpy(dtype=str),
        fold_ids=plan["oof_fold"].to_numpy(dtype=str),
    )
    all_groups = set(plan["group_id"].astype(str))
    folds = []
    for fold_id in sorted(plan["oof_fold"].astype(str).unique()):
        held_out = set(
            plan.loc[plan["oof_fold"].astype(str).eq(fold_id), "group_id"].astype(str)
        )
        folds.append(
            {
                "fold_id": fold_id,
                "held_out_group_ids_sha256": canonical_id_set_sha256(held_out),
                "training_group_ids_sha256": canonical_id_set_sha256(
                    all_groups - held_out
                ),
                "checkpoint_sha256": "c" * 64,
            }
        )
    provenance = {
        "schema_version": 1,
        "evaluation_role": "out_of_fold_training",
        "score_kind": "uncalibrated_sigmoid_probability",
        "probability_key": "uncalibrated_probabilities",
        "score_fit_uses_oof_labels": False,
        "plan_sha256": Path(f"{plan_path}.sha256").read_text().strip(),
        "predictions_sha256": _file_sha256(predictions_path),
        "folds": folds,
    }
    provenance_path = tmp_path / "oof-provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    output_path = tmp_path / "hard-errors.csv"
    report = mine_oof_errors(
        plan_path=plan_path,
        predictions_path=predictions_path,
        provenance_path=provenance_path,
        output_path=output_path,
        low_threshold=0.2,
        high_threshold=0.8,
        max_per_error_source=1,
    )

    selected = pd.read_csv(output_path)
    assert report["test_or_external_records_used"] == 0
    assert report["selected_records"] == 4
    assert set(selected["error_type"]) == {
        "hard_false_positive",
        "hard_false_negative",
    }
    assert selected.groupby(["error_type", "collection_id"]).size().max() == 1


def test_materialize_oof_fold_replaces_source_validation(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(_full_manifest(), manifest_path)
    pd.DataFrame(
        columns=[
            "left_image_name",
            "right_image_name",
            "same_group",
            "left_split",
            "right_split",
        ]
    ).to_csv(tmp_path / "manifest.near_duplicates.csv", index=False)
    plan_path = tmp_path / "plan.csv"
    create_oof_plan(
        manifest_path=manifest_path,
        output_path=plan_path,
        folds=2,
        seed=4,
    )
    output_path = tmp_path / "fold-0.csv"
    report = materialize_oof_fold(
        manifest_path=manifest_path,
        plan_path=plan_path,
        fold_id="fold_0",
        output_path=output_path,
    )

    derived = read_manifest(output_path, additional_text_columns=("oof_fold",))
    assert report["dropped_source_validation_records"] == 2
    assert set(derived["split"].astype(str)) == {"train", "validation", "test"}
    assert derived.groupby("group_id")["split"].nunique().max() == 1
    assert set(
        derived.loc[derived["split"].astype(str).eq("validation"), "oof_fold"]
        .dropna()
        .astype(str)
    ) == {"fold_0"}
    assert set(
        derived.loc[derived["split"].astype(str).eq("test"), "image_name"]
    ) == {"image-200", "image-201"}


def test_collect_oof_predictions_proves_training_complements(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(_full_manifest(), manifest_path)
    pd.DataFrame(
        columns=[
            "left_image_name",
            "right_image_name",
            "same_group",
            "left_split",
            "right_split",
        ]
    ).to_csv(tmp_path / "manifest.near_duplicates.csv", index=False)
    plan_path = tmp_path / "plan.csv"
    create_oof_plan(
        manifest_path=manifest_path,
        output_path=plan_path,
        folds=2,
        seed=5,
    )
    run_dirs = []
    for fold_id in ("fold_0", "fold_1"):
        fold_manifest_path = tmp_path / f"{fold_id}.csv"
        materialize_oof_fold(
            manifest_path=manifest_path,
            plan_path=plan_path,
            fold_id=fold_id,
            output_path=fold_manifest_path,
        )
        fold_manifest = read_manifest(
            fold_manifest_path, additional_text_columns=("oof_fold",)
        )
        run_dir = tmp_path / f"run-{fold_id}"
        run_dir.mkdir()
        run_manifest_sha256 = write_manifest(
            fold_manifest, run_dir / "manifest.csv"
        )
        checkpoint_path = run_dir / "best.pt"
        checkpoint_path.write_bytes(fold_id.encode("ascii"))
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "manifest_sha256": run_manifest_sha256,
                    "checkpoint_sha256": _file_sha256(checkpoint_path),
                }
            ),
            encoding="utf-8",
        )
        validation = fold_manifest.loc[
            fold_manifest["split"].astype(str).eq("validation")
        ]
        np.savez_compressed(
            run_dir / "predictions_validation.npz",
            uncalibrated_probabilities=np.where(
                validation["target"].astype(int).eq(1), 0.8, 0.2
            ),
            targets=validation["target"].astype(int).to_numpy(),
            image_names=validation["image_name"].to_numpy(dtype=str),
        )
        (run_dir / "calibration.json").write_text("{}\n", encoding="utf-8")
        run_dirs.append(run_dir)

    output_path = tmp_path / "oof-predictions.npz"
    provenance_path = tmp_path / "oof-provenance.json"
    provenance = collect_oof_predictions(
        plan_path=plan_path,
        run_dirs=run_dirs,
        output_path=output_path,
        provenance_path=provenance_path,
    )

    with np.load(output_path, allow_pickle=False) as arrays:
        assert len(arrays["targets"]) == len(_training_manifest())
        assert set(arrays.files) == {
            "uncalibrated_probabilities",
            "targets",
            "image_names",
            "group_ids",
            "fold_ids",
        }
    assert provenance["evaluation_role"] == "out_of_fold_training"
    assert provenance["score_fit_uses_oof_labels"] is False
    assert provenance["test_or_external_records_used"] == 0


def test_oof_mining_rejects_test_predictions(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(_training_manifest(), manifest_path)
    plan_path = tmp_path / "plan.csv"
    create_oof_plan(
        manifest_path=manifest_path,
        output_path=plan_path,
        folds=2,
        seed=2,
    )
    plan = read_manifest(plan_path, additional_text_columns=("oof_fold",))
    predictions_path = tmp_path / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        uncalibrated_probabilities=np.full(len(plan), 0.5),
        targets=plan["target"].astype(int).to_numpy(),
        image_names=plan["image_name"].to_numpy(dtype=str),
        group_ids=plan["group_id"].to_numpy(dtype=str),
        fold_ids=plan["oof_fold"].to_numpy(dtype=str),
    )
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evaluation_role": "held_out_test",
                "score_kind": "uncalibrated_sigmoid_probability",
                "probability_key": "uncalibrated_probabilities",
                "score_fit_uses_oof_labels": False,
                "plan_sha256": Path(f"{plan_path}.sha256").read_text().strip(),
                "predictions_sha256": _file_sha256(predictions_path),
                "folds": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Only out-of-fold training"):
        mine_oof_errors(
            plan_path=plan_path,
            predictions_path=predictions_path,
            provenance_path=provenance_path,
            output_path=tmp_path / "errors.csv",
            low_threshold=0.2,
            high_threshold=0.8,
            max_per_error_source=10,
        )


def test_source_balanced_sampler_preserves_priority_negatives() -> None:
    labels = np.array(
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    )
    sources = np.array(["a"] * 8 + ["b"] * 8)
    sampler = SourceClassBalancedSampler(
        labels,
        sources,
        negative_to_positive_ratio=2,
        seed=9,
        always_include_negative_indices=np.array([2, 10]),
        always_include_negative_repeats=2,
    )

    indices = list(sampler)
    assert len(indices) == 12
    assert indices.count(2) == 2
    assert indices.count(10) == 2
    assert sum(sources[index] == "a" for index in indices) == 6
    assert sum(sources[index] == "b" for index in indices) == 6


def test_source_balanced_sampler_preserves_priority_positives() -> None:
    labels = np.array(
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    )
    sources = np.array(["a"] * 8 + ["b"] * 8)
    sampler = SourceClassBalancedSampler(
        labels,
        sources,
        negative_to_positive_ratio=2,
        seed=9,
        always_include_positive_indices=np.array([0, 8]),
        always_include_positive_repeats=2,
    )

    indices = list(sampler)
    assert len(indices) == 12
    assert indices.count(0) == 2
    assert indices.count(8) == 2
    assert sum(sources[index] == "a" for index in indices) == 6
    assert sum(sources[index] == "b" for index in indices) == 6


def test_priority_records_require_verified_oof_training_rows(
    tmp_path: Path,
) -> None:
    training = _training_manifest().reset_index(drop=True)
    selected = pd.concat(
        [
            training.loc[training["target"].astype(int).eq(0)].head(1),
            training.loc[training["target"].astype(int).eq(1)].head(1),
        ],
        ignore_index=True,
    )
    selected["error_type"] = [
        "hard_false_positive",
        "hard_false_negative",
    ]
    priority_path = tmp_path / "hard-errors.csv"
    selected[
        [
            "image_name",
            "sha256",
            "group_id",
            "collection_id",
            "target",
            "error_type",
        ]
    ].to_csv(priority_path, index=False)
    digest = _file_sha256(priority_path)
    Path(f"{priority_path}.sha256").write_text(f"{digest}\n", encoding="ascii")
    (tmp_path / "hard-errors.report.json").write_text(
        json.dumps(
            {
                "purpose": "out_of_fold_hard_error_mining",
                "evaluation_role": "out_of_fold_training",
                "output_sha256": digest,
                "training_authorized": True,
                "test_or_external_records_used": 0,
            }
        ),
        encoding="utf-8",
    )

    negative_indices, positive_indices, evidence = load_priority_record_indices(
        priority_path, training
    )

    assert negative_indices.tolist() == [int(selected.index[0])]
    expected_positive_name = str(selected.loc[1, "image_name"])
    assert training.loc[positive_indices[0], "image_name"] == expected_positive_name
    assert evidence["hard_false_positive_records"] == 1
    assert evidence["hard_false_negative_records"] == 1
