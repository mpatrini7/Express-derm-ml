from __future__ import annotations

import numpy as np
import pytest

from express_derm_ml.optimize_attention_thresholds import (
    CohortRecord,
    ThreeZoneThresholdError,
    build_cohort,
    cross_validate_three_zone_thresholds,
    optimize_three_zone_thresholds,
    patient_fold,
    patient_partition,
    reference_class,
    three_class_metrics,
)


def row(**overrides: str) -> dict[str, str]:
    base = {
        "image_name": "ISIC_1",
        "patient_id": "P1",
        "lesion_id": "L1",
        "diagnosis_1": "Benign",
        "diagnosis_2": "Benign melanocytic proliferations",
        "diagnosis_3": "Nevus",
        "diagnosis_4": "",
        "diagnosis_5": "",
        "diagnosis_confirm_type": "histopathology",
    }
    base.update(overrides)
    return base


def test_reference_class_requires_histopathology_and_preserves_atypical() -> None:
    assert reference_class(row()) == "benign"
    assert reference_class(
        row(diagnosis_4="Nevus, Atypical, Dysplastic, or Clark")
    ) == "atypical"
    assert reference_class(
        row(diagnosis_1="Malignant", diagnosis_3="Melanoma")
    ) == "melanoma"
    assert reference_class(
        row(diagnosis_confirm_type="single image expert consensus")
    ) is None


def test_patient_partition_is_deterministic_and_patient_safe() -> None:
    first = patient_partition("patient-a", seed=42, calibration_fraction=0.5)
    second = patient_partition("patient-a", seed=42, calibration_fraction=0.5)
    assert first == second
    assert first in {"calibration", "test"}


def test_build_cohort_uses_one_image_per_lesion_and_no_patient_leakage() -> None:
    rows = []
    receipts = {}
    for patient_index in range(1, 30):
        for class_index, class_name in enumerate(
            ("benign", "atypical", "melanoma")
        ):
            image_name = f"ISIC_{patient_index}_{class_index}"
            values = row(
                image_name=image_name,
                patient_id=f"P{patient_index}",
                lesion_id=f"L{patient_index}_{class_index}",
            )
            if class_name == "atypical":
                values["diagnosis_4"] = "Nevus, Atypical, Dysplastic, or Clark"
            elif class_name == "melanoma":
                values["diagnosis_1"] = "Malignant"
                values["diagnosis_3"] = "Melanoma"
            rows.append(values)
            receipts[image_name] = "a" * 64
    cohort = build_cohort(
        rows,
        receipts,
        excluded_image_names=set(),
        seed=7,
        calibration_fraction=0.5,
    )
    assert all(isinstance(record, CohortRecord) for record in cohort)
    assert len(cohort) == len(rows)
    calibration_patients = {
        record.patient_id for record in cohort if record.split == "calibration"
    }
    test_patients = {
        record.patient_id for record in cohort if record.split == "test"
    }
    assert not calibration_patients & test_patients


def test_maximin_optimizer_balances_all_three_reference_classes() -> None:
    targets = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2])
    scores = np.asarray([0.01, 0.02, 0.03, 0.40, 0.50, 0.60, 0.90, 0.95, 0.99])
    selected = optimize_three_zone_thresholds(targets, scores)
    assert selected["class_recall"] == {
        "benign": 1.0,
        "atypical": 1.0,
        "melanoma": 1.0,
    }
    assert selected["worst_class_recall"] == 1.0


def test_three_class_metrics_rejects_missing_reference_class() -> None:
    with pytest.raises(ThreeZoneThresholdError, match="three target classes"):
        three_class_metrics(
            np.asarray([0, 2]),
            np.asarray([0.1, 0.9]),
            low_threshold=0.2,
            high_threshold=0.8,
        )


def test_patient_grouped_cross_validation_is_perfect_for_separable_scores() -> None:
    patient_ids = []
    targets = []
    scores = []
    for fold in range(3):
        selected_patients = []
        candidate = 0
        while len(selected_patients) < 2:
            patient_id = f"fold-{fold}-candidate-{candidate}"
            if patient_fold(patient_id, seed=91, folds=3) == fold:
                selected_patients.append(patient_id)
            candidate += 1
        for patient_id in selected_patients:
            patient_ids.extend([patient_id] * 3)
            targets.extend([0, 1, 2])
            scores.extend([0.05, 0.50, 0.95])

    result = cross_validate_three_zone_thresholds(
        np.asarray(targets),
        np.asarray(scores),
        np.asarray(patient_ids),
        seed=91,
        folds=3,
    )
    assert result["pooled_out_of_fold"]["class_recall"] == {
        "benign": 1.0,
        "atypical": 1.0,
        "melanoma": 1.0,
    }
    assert result["mean_fold_worst_class_recall"] == 1.0
