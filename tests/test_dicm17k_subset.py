from __future__ import annotations

from pathlib import Path

import pandas as pd

from express_derm_ml.curate_dicm17k_near_duplicates import (
    curate_dicm17k_near_duplicates,
)
from express_derm_ml.download_dicm17k_patient_subset import _eligible_frame
from express_derm_ml.manifest import write_manifest


def test_dicm17k_selection_is_confirmed_and_patient_safe(tmp_path: Path) -> None:
    reference = pd.DataFrame.from_records(
        [
            {
                "image_name": "ISIC_OVERLAP_IMAGE",
                "patient_id": "BASE_PATIENT",
            }
        ]
    )
    reference_path = tmp_path / "reference.csv"
    write_manifest(reference, reference_path)
    common = {
        "attribution": "Synthetic contributor",
        "copyright_license": "CC-0",
        "image_type": "dermoscopic",
        "lesion_id": "",
    }
    metadata = pd.DataFrame.from_records(
        [
            {
                **common,
                "isic_id": "ISIC_POSITIVE",
                "patient_id": "PATIENT_POSITIVE",
                "diagnosis_1": "Malignant",
                "diagnosis_2": "Malignant melanocytic proliferations (Melanoma)",
                "diagnosis_confirm_type": "histopathology",
            },
            {
                **common,
                "isic_id": "ISIC_NEGATIVE",
                "patient_id": "PATIENT_NEGATIVE",
                "diagnosis_1": "Benign",
                "diagnosis_2": "Benign melanocytic proliferations",
                "diagnosis_confirm_type": "serial imaging showing no change",
            },
            {
                **common,
                "isic_id": "ISIC_UNCONFIRMED",
                "patient_id": "PATIENT_UNCONFIRMED",
                "diagnosis_1": "Benign",
                "diagnosis_2": "Benign melanocytic proliferations",
                "diagnosis_confirm_type": "",
            },
            {
                **common,
                "isic_id": "ISIC_OTHER_MALIGNANCY",
                "patient_id": "PATIENT_OTHER",
                "diagnosis_1": "Malignant",
                "diagnosis_2": "Malignant epidermal proliferations",
                "diagnosis_confirm_type": "histopathology",
            },
            {
                **common,
                "isic_id": "ISIC_OVERLAP_PATIENT",
                "patient_id": "BASE_PATIENT",
                "diagnosis_1": "Benign",
                "diagnosis_2": "Benign epidermal proliferations",
                "diagnosis_confirm_type": "single image expert consensus",
            },
            {
                **common,
                "isic_id": "ISIC_OVERLAP_IMAGE",
                "patient_id": "NEW_PATIENT",
                "diagnosis_1": "Benign",
                "diagnosis_2": "Benign epidermal proliferations",
                "diagnosis_confirm_type": "single image expert consensus",
            },
        ]
    )
    metadata_path = tmp_path / "dicm.csv"
    metadata.to_csv(metadata_path, index=False, lineterminator="\n")
    config = {
        "dataset": {
            "expected_source_records": 6,
            "expected_subset_records": 2,
            "expected_patients": 2,
            "expected_melanoma": 1,
            "expected_license_counts": {"CC-0": 2},
            "source_url": "https://example.test/dicm",
        },
        "selection": {
            "image_type": "dermoscopic",
            "positive_diagnosis_1": "Malignant",
            "positive_diagnosis_2": (
                "Malignant melanocytic proliferations (Melanoma)"
            ),
            "negative_diagnosis_1": "Benign",
            "accepted_confirmation_types": [
                "histopathology",
                "serial imaging showing no change",
                "single image expert consensus",
            ],
        },
    }

    selected = _eligible_frame(metadata_path, reference_path, config)

    assert selected["image_name"].tolist() == [
        "ISIC_NEGATIVE",
        "ISIC_POSITIVE",
    ]
    assert selected.set_index("image_name")["diagnosis_class"].to_dict() == {
        "ISIC_NEGATIVE": "benign",
        "ISIC_POSITIVE": "melanoma",
    }
    assert set(selected["patient_id"]) == {
        "PATIENT_NEGATIVE",
        "PATIENT_POSITIVE",
    }


def test_dicm17k_near_duplicate_curation_drops_label_conflicts(
    tmp_path: Path,
) -> None:
    metadata = pd.DataFrame.from_records(
        [
            {
                "image_name": "A",
                "patient_id": "P1",
                "lesion_id": "",
                "diagnosis_class": "benign",
                "diagnosis_confirm_type": "single image expert consensus",
            },
            {
                "image_name": "B",
                "patient_id": "P2",
                "lesion_id": "L2",
                "diagnosis_class": "benign",
                "diagnosis_confirm_type": "histopathology",
            },
            {
                "image_name": "C",
                "patient_id": "P3",
                "lesion_id": "L3",
                "diagnosis_class": "melanoma",
                "diagnosis_confirm_type": "histopathology",
            },
            {
                "image_name": "D",
                "patient_id": "P4",
                "lesion_id": "L4",
                "diagnosis_class": "benign",
                "diagnosis_confirm_type": "histopathology",
            },
            {
                "image_name": "E",
                "patient_id": "P5",
                "lesion_id": "L5",
                "diagnosis_class": "melanoma",
                "diagnosis_confirm_type": "histopathology",
            },
        ]
    )
    pairs = pd.DataFrame.from_records(
        [
            {
                "left_image_name": "A",
                "right_image_name": "B",
                "hamming_distance": 0,
                "same_group": False,
            },
            {
                "left_image_name": "C",
                "right_image_name": "D",
                "hamming_distance": 2,
                "same_group": False,
            },
        ]
    )
    metadata_path = tmp_path / "metadata.csv"
    pairs_path = tmp_path / "pairs.csv"
    config_path = tmp_path / "config.yaml"
    output = tmp_path / "curated.csv"
    metadata.to_csv(metadata_path, index=False)
    pairs.to_csv(pairs_path, index=False)
    config_path.write_text(
        """dataset:
  expected_curated_records: 2
  expected_curated_patients: 2
  expected_curated_melanoma: 1
""",
        encoding="utf-8",
    )

    receipt = curate_dicm17k_near_duplicates(
        metadata_path=metadata_path,
        near_duplicates_path=pairs_path,
        config_path=config_path,
        output_path=output,
    )

    curated = pd.read_csv(output)
    assert set(curated["image_name"]) == {"B", "E"}
    assert receipt["conflicting_label_components"] == 1
    exclusions = pd.read_csv(tmp_path / "curated.exclusions.csv")
    assert set(exclusions["image_name"]) == {"A", "C", "D"}
