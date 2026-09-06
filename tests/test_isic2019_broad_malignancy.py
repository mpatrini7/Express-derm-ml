from __future__ import annotations

import pandas as pd
import pytest
from PIL import Image

from express_derm_ml.download_isic2019_broad_malignancy import (
    _verified_jpeg_details,
    eligible_broad_attention_frame,
    eligible_broad_malignancy_frame,
)


def _source_frames():
    archive = pd.DataFrame(
        {
            "image_name": ["a", "b", "c", "d"],
            "patient_id": [None, None, None, None],
            "lesion_id": ["api-a", "api-b", "api-c", "api-d"],
            "image_type": ["dermoscopic"] * 4,
            "copyright_license": ["CC-BY-NC"] * 4,
            "full_image_url": [
                f"https://isic-archive.s3.amazonaws.com/images/{name}.jpg"
                for name in ("a", "b", "c", "d")
            ],
            "full_image_size": [10, 11, 12, 13],
            "diagnosis_confirm_type": ["histopathology"] * 4,
            "attribution": ["source"] * 4,
        }
    )
    challenge = pd.DataFrame(
        {
            "image": ["a", "b", "c", "d"],
            "lesion_id": ["lesion-a", "lesion-b", "lesion-c", "lesion-d"],
        }
    )
    ground_truth = pd.DataFrame(
        {
            "image": ["a", "b", "c", "d"],
            "MEL": [0.0, 0.0, 0.0, 1.0],
            "NV": [0.0, 0.0, 0.0, 0.0],
            "BCC": [1.0, 0.0, 0.0, 0.0],
            "AK": [0.0, 1.0, 0.0, 0.0],
            "BKL": [0.0, 0.0, 0.0, 0.0],
            "DF": [0.0, 0.0, 0.0, 0.0],
            "VASC": [0.0, 0.0, 0.0, 0.0],
            "SCC": [0.0, 0.0, 1.0, 0.0],
            "UNK": [0.0, 0.0, 0.0, 0.0],
        }
    )
    return archive, ground_truth, challenge


def test_broad_subset_selects_only_bcc_ak_and_scc() -> None:
    selected = eligible_broad_malignancy_frame(*_source_frames())
    assert selected["image_name"].tolist() == ["a", "b", "c"]
    assert selected["diagnosis_class"].tolist() == ["bcc", "ak", "scc"]
    assert selected["lesion_id_challenge"].tolist() == [
        "lesion-a",
        "lesion-b",
        "lesion-c",
    ]


def test_broad_subset_rejects_missing_lesion_identity() -> None:
    archive, ground_truth, challenge = _source_frames()
    challenge.loc[0, "lesion_id"] = None
    with pytest.raises(RuntimeError, match="require lesion IDs"):
        eligible_broad_malignancy_frame(archive, ground_truth, challenge)


def test_download_validation_uses_jpeg_content_not_stale_snapshot_size(
    tmp_path,
) -> None:
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (17, 13), color=(30, 60, 90)).save(
        image_path,
        format="JPEG",
    )
    actual_size, width, height = _verified_jpeg_details(image_path)
    assert actual_size == image_path.stat().st_size
    assert (width, height) == (17, 13)


def test_download_validation_rejects_non_image(tmp_path) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_text("not an image", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid"):
        _verified_jpeg_details(image_path)


def test_broad_attention_controls_are_deterministic_and_exclude_base() -> None:
    archive, ground_truth, challenge = _source_frames()
    ground_truth.loc[ground_truth["image"].eq("d"), ["MEL", "NV"]] = [
        0.0,
        1.0,
    ]
    for diagnosis in ("BKL", "DF", "VASC"):
        name = diagnosis.lower()
        archive.loc[len(archive)] = {
            **archive.iloc[0].to_dict(),
            "image_name": name,
            "lesion_id": f"api-{name}",
            "full_image_url": (
                f"https://isic-archive.s3.amazonaws.com/images/{name}.jpg"
            ),
        }
        challenge.loc[len(challenge)] = {
            "image": name,
            "lesion_id": f"lesion-{name}",
        }
        row = {column: 0.0 for column in ground_truth.columns if column != "image"}
        row.update({"image": name, diagnosis: 1.0})
        ground_truth.loc[len(ground_truth)] = row

    first = eligible_broad_attention_frame(
        archive,
        ground_truth,
        challenge,
        benign_control_quotas={"NV": 1, "BKL": 1, "DF": 1, "VASC": 1},
        excluded_image_names={"b"},
        selection_seed="test-seed",
    )
    second = eligible_broad_attention_frame(
        archive,
        ground_truth,
        challenge,
        benign_control_quotas={"NV": 1, "BKL": 1, "DF": 1, "VASC": 1},
        excluded_image_names={"b"},
        selection_seed="test-seed",
    )

    assert first["image_name"].tolist() == second["image_name"].tolist()
    assert "b" not in set(first["image_name"])
    assert first["selection_role"].value_counts().to_dict() == {
        "same_source_benign_control": 4,
        "broad_malignancy_positive": 2,
    }
