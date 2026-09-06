from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from express_derm_ml.artifacts import ArtifactExistsError
from express_derm_ml.audit_patient_safe_overlap import (
    audit_patient_safe_overlap,
)
from express_derm_ml.evaluate_external import (
    _load_external_dataset_card,
    resolve_num_workers,
)
from express_derm_ml.fetch_isic_collection_metadata import extract_image_record
from express_derm_ml.integrity import perceptual_hash_file
from express_derm_ml.manifest import write_manifest
from express_derm_ml.prepare_milk10k_external import (
    prepare_external_manifest,
)
from express_derm_ml.revalidate_external_manifest import (
    _external_candidate_names,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    pd.DataFrame.from_records(records).to_csv(
        path,
        index=False,
        lineterminator="\n",
    )


def test_prepare_external_manifest_is_lesion_unique_and_not_trainable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "milk"
    root.mkdir()
    archive = root / "images.zip"
    archive.write_bytes(b"synthetic archive receipt")
    metadata = root / "metadata.csv"
    supplement = root / "supplement.csv"
    ground_truth = root / "ground_truth.csv"
    metadata_records: list[dict[str, object]] = []
    supplement_records: list[dict[str, object]] = []
    external_hashes: list[str] = []
    for index in range(2):
        lesion_id = f"IL_{index:04d}"
        for image_type, suffix in (
            ("clinical: close-up", "C"),
            ("dermoscopic", "D"),
        ):
            image_id = f"ISIC_{index:04d}{suffix}"
            metadata_records.append(
                {
                    "lesion_id": lesion_id,
                    "isic_id": image_id,
                    "image_type": image_type,
                    "copyright_license": "CC-BY-NC",
                }
            )
            supplement_records.append(
                {
                    "isic_id": image_id,
                    "diagnosis_full": "melanoma" if index == 0 else "nevus",
                    "diagnosis_confirm_type": "histopathology",
                }
            )
            if image_type == "dermoscopic":
                image_dir = root / "MILK10k_Training_Input" / lesion_id
                image_dir.mkdir(parents=True)
                pixels = np.random.default_rng(index + 91).integers(
                    0,
                    256,
                    size=(48, 48, 3),
                    dtype=np.uint8,
                )
                image_path = image_dir / f"{image_id}.jpg"
                Image.fromarray(pixels).save(image_path)
                external_hashes.append(
                    perceptual_hash_file(
                        image_path,
                        hash_size=4,
                        highfreq_factor=4,
                    )
                )
    _write_csv(metadata, metadata_records)
    _write_csv(supplement, supplement_records)
    _write_csv(
        ground_truth,
        [
            {"lesion_id": "IL_0000", "MEL": 1, "NV": 0},
            {"lesion_id": "IL_0001", "MEL": 0, "NV": 1},
        ],
    )

    reference_hash = f"{(~int(external_hashes[0], 16)) & 0xFFFF:04x}"
    reference = pd.DataFrame.from_records(
        [
            {
                "image_name": "REFERENCE_1",
                "group_id": "PATIENT_1",
                "sha256": "f" * 64,
                "perceptual_hash": reference_hash,
            }
        ]
    )
    reference_path = tmp_path / "reference.csv"
    reference_sha256 = write_manifest(reference, reference_path)

    config = {
        "dataset": {
            "id": "milk-synthetic",
            "version": "test-v1",
            "expected_metadata_rows": 4,
            "expected_supplement_rows": 4,
            "expected_lesions": 2,
            "evaluation_image_type": "dermoscopic",
            "source_url": "https://example.test/milk",
            "license": "CC-BY-NC-4.0",
            "license_url": "https://example.test/license",
            "attribution": "Synthetic fixture",
            "patient_identifier_status": (
                "unavailable_in_public_challenge_package"
            ),
        },
        "sources": {
            name: {
                "filename": path.name,
                "url": f"https://example.test/{path.name}",
                "sha256": _sha256(path),
            }
            for name, path in {
                "archive": archive,
                "metadata": metadata,
                "supplement": supplement,
                "ground_truth": ground_truth,
            }.items()
        },
        "labels": {
            "classes": ["MEL", "NV"],
            "melanoma_positive": ["MEL"],
            "broad_malignancy_positive": ["MEL"],
        },
        "integrity": {
            "perceptual_hash": {
                "algorithm": "phash",
                "implementation": "ImageHash",
                "implementation_version": "4.3.2",
                "hash_size": 4,
                "highfreq_factor": 4,
            },
            "near_duplicate_hamming_threshold": 0,
        },
    }
    config_path = tmp_path / "milk.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    output = tmp_path / "artifacts" / "external.csv"

    card = prepare_external_manifest(
        data_root=root,
        reference_manifest_path=reference_path,
        config_path=config_path,
        output_path=output,
    )

    assert card["records"] == 2
    assert card["lesions"] == 2
    assert card["melanoma_positive"] == 1
    assert card["training_authorized"] is False
    assert card["evaluation_image_type"] == "dermoscopic"
    assert card["reference_manifest_sha256"] == reference_sha256
    assert card["config_sha256"] == _sha256(config_path)
    manifest = pd.read_csv(output)
    assert manifest["lesion_id"].is_unique
    assert set(manifest["target_melanoma"]) == {0, 1}

    with pytest.raises(ArtifactExistsError, match="will not be overwritten"):
        prepare_external_manifest(
            data_root=root,
            reference_manifest_path=reference_path,
            config_path=config_path,
            output_path=output,
        )


def test_external_dataset_card_must_explicitly_forbid_training(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "external.csv"
    manifest_path.write_text("image_name\nISIC_1\n", encoding="utf-8")
    card_path = tmp_path / "external.dataset.json"
    card = {
        "manifest_sha256": "a" * 64,
        "reference_manifest_sha256": "b" * 64,
        "purpose": "external_research_evaluation_only",
        "patient_identifier_status": (
            "unavailable_in_public_challenge_package"
        ),
        "records": 1,
        "lesions": 1,
        "cross_source_exact_duplicates": 0,
        "cross_source_near_duplicate_candidates": 0,
        "training_authorized": True,
        "dataset": {"id": "synthetic"},
        "evaluation_image_type": "dermoscopic",
    }
    card_path.write_text(json.dumps(card), encoding="utf-8")

    with pytest.raises(RuntimeError, match="explicitly forbid training"):
        _load_external_dataset_card(
            manifest_path,
            manifest_sha256="a" * 64,
            training_manifest_sha256="b" * 64,
            records=1,
            lesions=1,
        )


def test_extract_isic_api_record_preserves_group_and_provenance() -> None:
    record = extract_image_record(
        {
            "isic_id": "ISIC_1234567",
            "copyright_license": "CC-BY-NC",
            "attribution": "Test contributor",
            "files": {
                "full": {
                    "url": "https://isic.example/image.jpg",
                    "size": 1234,
                }
            },
            "metadata": {
                "acquisition": {"image_type": "dermoscopic"},
                "clinical": {
                    "patient_id": "IP_1",
                    "lesion_id": "IL_2",
                    "diagnosis_1": "Malignant",
                    "diagnosis_3": "Melanoma",
                },
            },
        }
    )

    assert record["image_name"] == "ISIC_1234567"
    assert record["patient_id"] == "IP_1"
    assert record["lesion_id"] == "IL_2"
    assert record["copyright_license"] == "CC-BY-NC"
    assert record["full_image_size"] == 1234


def test_external_worker_override_supports_single_process_hosts() -> None:
    assert resolve_num_workers(4, None) == 4
    assert resolve_num_workers(4, 0) == 0
    with pytest.raises(ValueError, match="zero or greater"):
        resolve_num_workers(4, -1)


def test_external_candidate_exclusion_identifies_external_side() -> None:
    candidates = pd.DataFrame(
        [
            {
                "left_image_name": "train-a",
                "right_image_name": "external-a",
                "left_group_id": "reference:patient-a",
                "right_group_id": "external:lesion-a",
            },
            {
                "left_image_name": "external-b",
                "right_image_name": "train-b",
                "left_group_id": "external:lesion-b",
                "right_group_id": "reference:patient-b",
            },
        ]
    )
    assert _external_candidate_names(candidates) == {
        "external-a",
        "external-b",
    }


def test_patient_safe_overlap_audit_identifies_only_unique_images(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    _write_csv(
        reference,
        [
            {"image_name": "ISIC_1", "patient_id": "IP_1"},
            {"image_name": "ISIC_2", "patient_id": ""},
        ],
    )
    _write_csv(
        candidate,
        [
            {"image_name": "ISIC_1", "patient_id": "IP_1"},
            {"image_name": "ISIC_3", "patient_id": "IP_2"},
            {"image_name": "ISIC_4", "patient_id": ""},
        ],
    )

    report = audit_patient_safe_overlap(
        reference_path=reference,
        candidate_paths=[candidate],
        output_path=tmp_path / "audit.json",
    )

    assert report["reference"]["patient_safe_records"] == 1
    assert report["candidates"][0]["patient_safe_records"] == 2
    assert report["candidates"][0]["image_id_overlap_with_reference"] == 1
    assert report["candidates"][0]["unique_image_ids_not_in_reference"] == 1
