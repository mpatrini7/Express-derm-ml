from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from express_derm_ml.artifacts import (
    ArtifactExistsError,
    create_new_directory,
    write_json_exclusive,
    write_npz_exclusive,
)
from express_derm_ml.calibration import (
    apply_calibration,
    fit_development_calibration,
)
from express_derm_ml.calibrate import clone_training_run_for_recalibration
from express_derm_ml.build_ensemble_candidate import (
    select_validation_ensemble,
    standardized_logit_average,
)
from express_derm_ml.metrics import (
    expected_calibration_error,
    grouped_bootstrap_intervals,
)
from express_derm_ml.runtime_manifest import (
    RuntimeManifestError,
    build_research_runtime_manifest,
)
from express_derm_ml.dataset import (
    DEPLOYMENT_PREPROCESSING_VERSION,
    LETTERBOX_PREPROCESSING_VERSION,
)
from express_derm_ml.select_research_candidate import (
    select_candidate,
    summarize_candidate,
)
from express_derm_ml.common import sha256_file
from express_derm_ml.train_distilled import (
    distillation_loss,
    teacher_run_provenance,
)


def test_artifacts_are_not_overwritten(tmp_path: Path) -> None:
    run_dir = create_new_directory(tmp_path / "run")
    with pytest.raises(ArtifactExistsError, match="will not be reused"):
        create_new_directory(run_dir)

    json_path = run_dir / "result.json"
    write_json_exclusive(json_path, {"status": "first"})
    with pytest.raises(ArtifactExistsError, match="will not be overwritten"):
        write_json_exclusive(json_path, {"status": "second"})
    assert json.loads(json_path.read_text(encoding="utf-8")) == {
        "status": "first"
    }

    predictions_path = run_dir / "predictions.npz"
    write_npz_exclusive(predictions_path, values=np.array([1, 2, 3]))
    with pytest.raises(ArtifactExistsError, match="will not be overwritten"):
        write_npz_exclusive(predictions_path, values=np.array([4, 5, 6]))
    assert np.load(predictions_path)["values"].tolist() == [1, 2, 3]


def test_recalibration_clone_preserves_training_and_omits_old_results(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for name in (
        "best.pt",
        "config.yaml",
        "history.json",
        "manifest.csv",
        "manifest.csv.sha256",
        "manifest.near_duplicates.csv",
        "manifest.report.json",
        "preflight.json",
    ):
        (source / name).write_bytes(f"content:{name}".encode("ascii"))
    checkpoint_hash = "a" * 64
    (source / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "checkpoint_sha256": checkpoint_hash,
            }
        ),
        encoding="utf-8",
    )
    (source / "calibration.json").write_text("{}", encoding="utf-8")
    (source / "metrics_test.json").write_text("{}", encoding="utf-8")

    output = clone_training_run_for_recalibration(
        source,
        tmp_path / "derived",
    )

    derived_run = json.loads((output / "run.json").read_text())
    assert (output / "best.pt").read_bytes() == (source / "best.pt").read_bytes()
    assert not (output / "calibration.json").exists()
    assert not (output / "metrics_test.json").exists()
    assert derived_run["derived_calibration"] is True
    assert derived_run["derived_from_checkpoint_sha256"] == checkpoint_hash
    assert derived_run["evaluation_preprocessing"] == (
        DEPLOYMENT_PREPROCESSING_VERSION
    )


def test_development_calibration_is_ordered_and_unvalidated() -> None:
    targets = np.array([0] * 8 + [1] * 8, dtype=np.int64)
    logits = np.array(
        [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5]
        + [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        dtype=np.float64,
    )
    calibration = fit_development_calibration(
        logits,
        targets,
        minimum_sensitivity=0.90,
        minimum_specificity=0.90,
        ece_bins=8,
    )

    assert calibration["selection_split"] == "validation"
    assert calibration["method"] == "affine_logistic_scaling"
    assert 0.0 < calibration["logit_scale"] <= 20.0
    assert np.isfinite(calibration["logit_bias"])
    assert (
        0.0
        <= calibration["low_threshold"]
        < calibration["high_threshold"]
        <= 1.0
    )
    assert (
        calibration["calibrated_log_loss"]
        <= calibration["uncalibrated_log_loss"]
    )
    assert calibration["threshold_status"] == "development_frozen"
    assert calibration["thresholds_validated"] is False
    assert calibration["research_only"] is True


def test_affine_calibration_corrects_a_weighted_logit_intercept() -> None:
    targets = np.array([0] * 950 + [1] * 50, dtype=np.int64)
    true_probabilities = np.linspace(0.001, 0.60, len(targets))
    true_logits = np.log(true_probabilities / (1.0 - true_probabilities))
    shifted_logits = true_logits + 4.0
    calibration = fit_development_calibration(
        shifted_logits,
        targets,
        minimum_sensitivity=0.80,
        minimum_specificity=0.80,
        ece_bins=10,
    )
    calibrated = apply_calibration(shifted_logits, calibration)

    assert abs(calibration["logit_bias"]) > 1.0
    assert expected_calibration_error(
        targets,
        calibrated,
    ) < expected_calibration_error(
        targets,
        1.0 / (1.0 + np.exp(-shifted_logits)),
    )


def test_grouped_bootstrap_reports_reproducible_intervals() -> None:
    targets = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    probabilities = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    groups = np.array(["n1", "n2", "n3", "p1", "p2", "p3"])
    first = grouped_bootstrap_intervals(
        targets,
        probabilities,
        groups,
        threshold=0.5,
        samples=100,
        seed=9,
    )
    second = grouped_bootstrap_intervals(
        targets,
        probabilities,
        groups,
        threshold=0.5,
        samples=100,
        seed=9,
    )

    assert first == second
    assert first["method"] == "grouped_stratified_bootstrap"


def test_expected_calibration_error_rewards_calibrated_predictions() -> None:
    targets = np.array([0, 0, 1, 1], dtype=np.int64)
    calibrated = np.array([0.05, 0.10, 0.90, 0.95], dtype=np.float64)
    inverted = 1.0 - calibrated
    assert expected_calibration_error(
        targets,
        calibrated,
        bins=4,
    ) < expected_calibration_error(
        targets,
        inverted,
        bins=4,
    )


def test_standardized_logit_ensemble_uses_validation_scale() -> None:
    logits = {
        "a": np.array([-2.0, 0.0, 2.0]),
        "b": np.array([-20.0, 0.0, 20.0]),
    }
    normalization = {
        "a": {"mean": 0.0, "standard_deviation": 2.0},
        "b": {"mean": 0.0, "standard_deviation": 20.0},
    }

    combined = standardized_logit_average(
        logits,
        normalization,
        ["a", "b"],
    )

    assert combined.tolist() == [-1.0, 0.0, 1.0]


def test_validation_ensemble_selection_prioritizes_pr_auc() -> None:
    targets = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    logits = {
        "weak": np.array([-1.0, 0.2, -0.2, 0.1, 0.3]),
        "strong": np.array([-1.0, -0.5, 0.0, 0.5, 1.0]),
    }
    normalization = {
        name: {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std()),
        }
        for name, values in logits.items()
    }

    selected, ranking = select_validation_ensemble(
        logits,
        targets,
        normalization,
        {
            "weak_only": ["weak"],
            "strong_only": ["strong"],
        },
    )

    assert selected == "strong_only"
    assert ranking[0]["pr_auc"] > ranking[1]["pr_auc"]


def test_distillation_loss_blends_hard_and_teacher_targets() -> None:
    logits = torch.tensor([-0.5, 0.5], requires_grad=True)
    hard_targets = torch.tensor([0.0, 1.0])
    teacher_scores = torch.tensor([-1.0, 1.0])
    criterion = torch.nn.BCEWithLogitsLoss()

    total, hard, soft = distillation_loss(
        logits,
        hard_targets,
        teacher_scores,
        criterion,
        temperature=2.0,
        hard_weight=0.35,
        soft_weight=0.65,
    )

    assert torch.isclose(total, 0.35 * hard + 0.65 * soft)
    total.backward()
    assert torch.isfinite(logits.grad).all()

    with pytest.raises(ValueError, match="sum to one"):
        distillation_loss(
            logits.detach(),
            hard_targets,
            teacher_scores,
            criterion,
            temperature=2.0,
            hard_weight=0.4,
            soft_weight=0.4,
        )

    mse_total, mse_hard, mse_soft = distillation_loss(
        logits.detach(),
        hard_targets,
        teacher_scores,
        criterion,
        temperature=2.0,
        hard_weight=0.1,
        soft_weight=0.9,
        soft_loss_kind="logit_mse",
    )
    assert torch.isclose(mse_soft, torch.tensor(0.25))
    assert torch.isclose(mse_total, 0.1 * mse_hard + 0.9 * mse_soft)


def test_weighted_teacher_provenance_does_not_require_ensemble_json() -> None:
    scale = {"transform": "(weighted_calibrated_logit-bias)/scale"}

    provenance = teacher_run_provenance(
        {
            "target_kind": "weighted_generalist_scale_logit",
            "ensemble_component_set_sha256": "a" * 64,
            "teacher_score_scale": scale,
            "source_teacher_receipt_sha256": "b" * 64,
        }
    )

    assert provenance["teacher_ensemble_json_sha256"] is None
    assert provenance["teacher_ensemble_component_set_sha256"] == "a" * 64
    assert provenance["teacher_score_scale"] == scale


def test_research_selection_validates_ensemble_artifacts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "ensemble"
    run_dir.mkdir()
    identity = "a" * 64
    calibration = {
        "ensemble_component_set_sha256": identity,
        "thresholds_validated": False,
    }
    metric_values = {
        "roc_auc": 0.90,
        "pr_auc": 0.30,
        "sensitivity": 0.91,
        "specificity": 0.70,
        "brier_score": 0.02,
        "expected_calibration_error": 0.01,
    }
    test_metrics = {
        "ensemble_component_set_sha256": identity,
        "thresholds_validated": False,
        "calibrated_low_threshold_metrics": metric_values,
    }
    external_metrics = {
        "ensemble_component_set_sha256": identity,
        "thresholds_validated": False,
        "melanoma_vs_all": {"low_threshold": metric_values},
    }
    write_json_exclusive(run_dir / "calibration.json", calibration)
    write_json_exclusive(run_dir / "metrics_test.json", test_metrics)
    write_json_exclusive(
        run_dir / "metrics_external_milk10k_dermoscopic.json",
        external_metrics,
    )
    write_json_exclusive(
        run_dir / "metrics_external_milk10k_clinical.json",
        external_metrics,
    )
    artifact_names = (
        "calibration.json",
        "metrics_test.json",
        "metrics_external_milk10k_dermoscopic.json",
        "metrics_external_milk10k_clinical.json",
    )
    ensemble = {
        "status": "complete",
        "deployment_authorized": False,
        "ensemble_component_set_sha256": identity,
        "components": [
            {
                "checkpoint_sha256": "b" * 64,
                "manifest_sha256": "c" * 64,
            }
        ],
        "inference_model_count": 1,
        "artifact_sha256": {
            filename: sha256_file(run_dir / filename)
            for filename in artifact_names
        },
    }
    write_json_exclusive(run_dir / "ensemble.json", ensemble)

    summary = summarize_candidate("ensemble", run_dir)

    assert summary["candidate_type"] == "ensemble"
    assert summary["milk10k_dermoscopic"]["pr_auc"] == 0.30
    assert summary["deployment_authorized"] is False


def test_research_runtime_manifest_cannot_claim_validated_thresholds() -> None:
    checkpoint = {
        "architecture": "efficientnet_b0",
        "image_size": 384,
    }
    config = {
        "deployment": {
            "input_name": "image",
            "output_name": "logit",
            "preprocessing": DEPLOYMENT_PREPROCESSING_VERSION,
        },
        "calibration": {
            "development_abstention_margin": 0.0,
        },
    }
    run = {
        "manifest_sha256": "a" * 64,
        "evaluation_preprocessing": DEPLOYMENT_PREPROCESSING_VERSION,
    }
    calibration = {
        "selection_split": "validation",
        "thresholds_validated": False,
        "low_threshold": 0.2,
        "high_threshold": 0.8,
        "temperature": 1.2,
        "method": "temperature_scaling",
        "evaluation_preprocessing": DEPLOYMENT_PREPROCESSING_VERSION,
    }
    manifest = build_research_runtime_manifest(
        model_version="research-v1",
        checkpoint=checkpoint,
        config=config,
        run=run,
        calibration=calibration,
        model_sha256="b" * 64,
        checkpoint_sha256="c" * 64,
        calibration_sha256="d" * 64,
    )
    assert manifest["validation_status"] == "research_only"
    assert manifest["threshold_status"] == "development_frozen"
    assert manifest["thresholds_validated"] is False
    assert manifest["research_only"] is True
    assert manifest["calibration_sha256"] == "d" * 64
    assert manifest["schema_version"] == 3
    assert manifest["release_version"] == "research-v1"
    assert manifest["model_family"] == "express-derm"
    assert manifest["runtime_identity"] == "express-derm"
    assert manifest["runtime_model_count"] == 1
    assert manifest["external_runtime_models"] == []
    assert manifest["preprocessing"]["version"] == (
        DEPLOYMENT_PREPROCESSING_VERSION
    )

    calibration = {
        **calibration,
        "method": "affine_logistic_scaling",
        "logit_scale": 0.8,
        "logit_bias": -3.2,
    }
    calibration.pop("temperature")
    manifest = build_research_runtime_manifest(
        model_version="research-v2",
        checkpoint=checkpoint,
        config=config,
        run=run,
        calibration=calibration,
        model_sha256="b" * 64,
        checkpoint_sha256="c" * 64,
        calibration_sha256="d" * 64,
    )
    assert manifest["calibration_method"] == "affine_logistic_scaling"
    assert manifest["logit_scale"] == 0.8
    assert manifest["logit_bias"] == -3.2

    distilled_run = {
        "manifest_sha256": "a" * 64,
        "evaluation_preprocessing": DEPLOYMENT_PREPROCESSING_VERSION,
        "candidate_type": "distilled_student",
        "teacher_ensemble_component_set_sha256": "e" * 64,
        "teacher_ensemble_json_sha256": "f" * 64,
        "teacher_targets_sha256": "1" * 64,
        "initial_checkpoint_sha256": "2" * 64,
    }
    manifest = build_research_runtime_manifest(
        model_version="research-kd-v1",
        checkpoint=checkpoint,
        config=config,
        run=distilled_run,
        calibration=calibration,
        model_sha256="b" * 64,
        checkpoint_sha256="c" * 64,
        calibration_sha256="d" * 64,
    )
    assert manifest["training_provenance"]["candidate_type"] == (
        "distilled_student"
    )
    assert manifest["training_provenance"][
        "teacher_ensemble_component_set_sha256"
    ] == "e" * 64

    weighted_distilled_run = {
        **distilled_run,
        "teacher_target_kind": "weighted_generalist_scale_logit",
        "teacher_ensemble_json_sha256": None,
        "teacher_score_scale": {"transform": "inverse_affine"},
    }
    weighted_manifest = build_research_runtime_manifest(
        model_version="research-weighted-kd-v1",
        checkpoint=checkpoint,
        config=config,
        run=weighted_distilled_run,
        calibration=calibration,
        model_sha256="b" * 64,
        checkpoint_sha256="c" * 64,
        calibration_sha256="d" * 64,
    )
    weighted_provenance = weighted_manifest["training_provenance"]
    assert "teacher_ensemble_json_sha256" not in weighted_provenance
    assert weighted_provenance["teacher_target_kind"] == (
        "weighted_generalist_scale_logit"
    )
    assert weighted_provenance["teacher_score_scale"] == {
        "transform": "inverse_affine"
    }

    calibration["thresholds_validated"] = True
    with pytest.raises(RuntimeManifestError, match="unvalidated"):
        build_research_runtime_manifest(
            model_version="research-v1",
            checkpoint=checkpoint,
            config=config,
            run=run,
            calibration=calibration,
            model_sha256="b" * 64,
            checkpoint_sha256="c" * 64,
            calibration_sha256="d" * 64,
        )


def test_research_runtime_manifest_records_letterbox_geometry() -> None:
    checkpoint = {
        "architecture": "efficientnet_b0",
        "image_size": 1024,
        "preprocessing": LETTERBOX_PREPROCESSING_VERSION,
    }
    config = {
        "deployment": {
            "input_name": "image",
            "output_name": "logit",
            "preprocessing": LETTERBOX_PREPROCESSING_VERSION,
        },
        "calibration": {"development_abstention_margin": 0.0},
    }
    run = {
        "manifest_sha256": "a" * 64,
        "evaluation_preprocessing": LETTERBOX_PREPROCESSING_VERSION,
    }
    calibration = {
        "selection_split": "validation",
        "thresholds_validated": False,
        "low_threshold": 0.2,
        "high_threshold": 0.8,
        "temperature": 1.2,
        "method": "temperature_scaling",
        "evaluation_preprocessing": LETTERBOX_PREPROCESSING_VERSION,
    }

    manifest = build_research_runtime_manifest(
        model_version="express-derm-highres-research",
        checkpoint=checkpoint,
        config=config,
        run=run,
        calibration=calibration,
        model_sha256="b" * 64,
        checkpoint_sha256="c" * 64,
        calibration_sha256="d" * 64,
    )

    assert manifest["image_size"] == 1024
    assert manifest["preprocessing"]["version"] == (
        LETTERBOX_PREPROCESSING_VERSION
    )
    assert manifest["preprocessing"]["resize_geometry"] == (
        "letterbox_square_imagenet_mean"
    )


def test_research_selection_prioritizes_external_pr_auc() -> None:
    def candidate(name: str, external_pr: float, test_pr: float):
        return {
            "name": name,
            "milk10k_dermoscopic": {
                "pr_auc": external_pr,
                "roc_auc": 0.8,
                "brier_score": 0.1,
            },
            "test": {"pr_auc": test_pr, "roc_auc": 0.9},
        }

    report = select_candidate(
        [
            candidate("internal-best", 0.25, 0.30),
            candidate("external-best", 0.27, 0.20),
        ]
    )

    assert report["selected_research_candidate"] == "external-best"
    assert report["deployment_authorized"] is False
    assert report["thresholds_validated"] is False
    assert report["research_only"] is True
