from __future__ import annotations

from pathlib import Path

import pandas as pd

from express_derm_ml.download_msk1_benign_controls import _select_controls


def test_msk1_controls_are_patient_safe_and_one_per_lesion(
    tmp_path: Path,
) -> None:
    metadata = pd.DataFrame(
        [
            {
                "image_name": "ISIC_NEW_2",
                "patient_id": "P-NEW",
                "lesion_id": "L-NEW",
                "diagnosis_1": "Benign",
                "diagnosis_3": "Nevus",
                "diagnosis_confirm_type": "histopathology",
                "image_type": "dermoscopic",
                "copyright_license": "CC-0",
                "attribution": "Anonymous",
                "full_image_url": "https://example.invalid/2.jpg",
                "full_image_size": 200,
            },
            {
                "image_name": "ISIC_NEW_1",
                "patient_id": "P-NEW",
                "lesion_id": "L-NEW",
                "diagnosis_1": "Benign",
                "diagnosis_3": "Nevus",
                "diagnosis_confirm_type": "histopathology",
                "image_type": "dermoscopic",
                "copyright_license": "CC-0",
                "attribution": "Anonymous",
                "full_image_url": "https://example.invalid/1.jpg",
                "full_image_size": 100,
            },
            {
                "image_name": "ISIC_BASE_PATIENT",
                "patient_id": "P-BASE",
                "lesion_id": "L-OTHER",
                "diagnosis_1": "Benign",
                "diagnosis_3": "Nevus",
                "diagnosis_confirm_type": "histopathology",
                "image_type": "dermoscopic",
                "copyright_license": "CC-0",
                "attribution": "Anonymous",
                "full_image_url": "https://example.invalid/base.jpg",
                "full_image_size": 100,
            },
            {
                "image_name": "ISIC_DEMO",
                "patient_id": "P-DEMO",
                "lesion_id": "L-DEMO",
                "diagnosis_1": "Benign",
                "diagnosis_3": "Nevus",
                "diagnosis_confirm_type": "histopathology",
                "image_type": "dermoscopic",
                "copyright_license": "CC-0",
                "attribution": "Anonymous",
                "full_image_url": "https://example.invalid/demo.jpg",
                "full_image_size": 100,
            },
            {
                "image_name": "ISIC_MALIGNANT",
                "patient_id": "P-MALIGNANT",
                "lesion_id": "L-MALIGNANT",
                "diagnosis_1": "Malignant",
                "diagnosis_3": "Melanoma",
                "diagnosis_confirm_type": "histopathology",
                "image_type": "dermoscopic",
                "copyright_license": "CC-0",
                "attribution": "Anonymous",
                "full_image_url": "https://example.invalid/malignant.jpg",
                "full_image_size": 100,
            },
        ]
    )
    reference = pd.DataFrame(
        [{"image_name": "ISIC_BASE", "patient_id": "P-BASE"}]
    )
    demo = pd.DataFrame([{"isic_id": "ISIC_DEMO"}])
    metadata_path = tmp_path / "metadata.csv"
    reference_path = tmp_path / "reference.csv"
    demo_path = tmp_path / "demo.csv"
    metadata.to_csv(metadata_path, index=False)
    reference.to_csv(reference_path, index=False)
    demo.to_csv(demo_path, index=False)

    selected, audit = _select_controls(
        metadata_path,
        reference_path,
        demo_path,
    )

    assert selected["image_name"].tolist() == ["ISIC_NEW_1"]
    assert selected["image_path"].tolist() == [
        "isic-msk1-benign-controls/ISIC_NEW_1.jpg"
    ]
    assert selected["target"].tolist() == [0]
    assert audit["selected_records"] == 1
    assert audit["selected_patients"] == 1
    assert audit["selected_lesions"] == 1
