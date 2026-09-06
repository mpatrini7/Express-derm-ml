from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from express_derm_ml.augment_training_manifest import (
    augment_training_manifest,
)
from express_derm_ml.integrity import find_near_duplicate_pairs
from express_derm_ml.manifest import (
    ManifestValidationError,
    artifact_path,
    canonical_manifest_sha256,
    load_curation_config,
    read_manifest,
    write_manifest,
)
from express_derm_ml.prepare_manifest import prepare_manifest
from express_derm_ml.preflight import validate_training_input
from express_derm_ml.split_manifest import split_manifest


def _write_config(path: Path, expected_images: int) -> Path:
    config = {
        "schema_version": 1,
        "dataset": {
            "id": "synthetic-test-collection",
            "version": "test-v1",
            "expected_images": expected_images,
            "source_url": "https://example.test/source",
            "license": "CC-BY-NC-4.0",
            "license_url": "https://example.test/license",
            "attribution": "Synthetic test fixture; not clinical data.",
        },
        "columns": {
            "image": "image_name",
            "group": "patient_id",
            "patient": "patient_id",
            "diagnosis": "diagnosis",
        },
        "label_mapping": {
            "version": "test-labels-v1",
            "task": "binary_attention",
            "source_column": "benign_malignant",
            "values": {
                "benign": {
                    "target": 0,
                    "label": "curated_negative",
                },
                "malignant": {
                    "target": 1,
                    "label": "curated_positive",
                },
            },
        },
        "integrity": {
            "perceptual_hash": {
                "algorithm": "phash",
                "implementation": "ImageHash",
                "implementation_version": "4.3.2",
                "hash_size": 8,
                "highfreq_factor": 4,
            },
            "near_duplicate_hamming_threshold": 6,
        },
    }
    path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _create_collection(
    root: Path,
    *,
    groups: int = 10,
) -> tuple[Path, Path, Path, pd.DataFrame]:
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    records: list[dict[str, str]] = []
    for group_index in range(groups):
        for target, source_label in ((0, "benign"), (1, "malignant")):
            image_index = group_index * 2 + target
            image_name = f"ISIC_TEST_{image_index:04d}"
            pixels = np.random.default_rng(image_index).integers(
                0,
                256,
                size=(32, 32, 3),
                dtype=np.uint8,
            )
            image = Image.fromarray(pixels)
            image.save(images_dir / f"{image_name}.png")
            records.append(
                {
                    "image_name": image_name,
                    "patient_id": f"PATIENT_{group_index:03d}",
                    "diagnosis": "melanoma" if target else "nevus",
                    "benign_malignant": source_label,
                }
            )

    metadata = pd.DataFrame.from_records(records)
    metadata_path = root / "metadata.csv"
    metadata.to_csv(metadata_path, index=False, lineterminator="\n")
    config_path = _write_config(root / "curation.yaml", len(metadata))
    return metadata_path, images_dir, config_path, metadata


def test_augment_training_manifest_assigns_new_patients_to_train_only(
    tmp_path: Path,
) -> None:
    common = {
        "perceptual_hash_algorithm": "phash",
        "perceptual_hash_implementation": "ImageHash",
        "perceptual_hash_implementation_version": "4.3.2",
        "perceptual_hash_bits": 16,
        "near_duplicate_hamming_threshold": 0,
    }
    base = pd.DataFrame.from_records(
        [
            {
                **common,
                "image_name": f"BASE_{index}",
                "image_path": f"BASE_{index}.jpg",
                "target": target,
                "group_id": f"PATIENT_{index}",
                "patient_id": f"PATIENT_{index}",
                "sha256": f"{index + 1:064x}",
                "perceptual_hash": f"{index + 1:04x}",
                "split": split,
                "fold": index,
            }
            for index, (split, target) in enumerate(
                [
                    ("train", 0),
                    ("train", 1),
                    ("validation", 0),
                    ("validation", 1),
                    ("test", 0),
                    ("test", 1),
                ]
            )
        ]
    )
    additional = pd.DataFrame.from_records(
        [
            {
                **common,
                "image_name": "EXTRA_1",
                "image_path": "EXTRA_1.jpg",
                "target": 1,
                "group_id": "EXTRA_PATIENT",
                "patient_id": "EXTRA_PATIENT",
                "sha256": "f" * 64,
                "perceptual_hash": "ffff",
            }
        ]
    )
    base_path = tmp_path / "base.csv"
    additional_path = tmp_path / "additional.csv"
    write_manifest(base, base_path)
    write_manifest(additional, additional_path)
    output = tmp_path / "combined.csv"

    result = augment_training_manifest(
        base_manifest_path=base_path,
        additional_manifest_path=additional_path,
        output_path=output,
        base_image_prefix="isic-2020/images",
        additional_image_prefix="isic-2019/images",
    )

    combined = pd.read_csv(output)
    extra = combined.loc[combined["image_name"].eq("EXTRA_1")].iloc[0]
    assert extra["split"] == "train"
    assert extra["image_path"] == "isic-2019/images/EXTRA_1.jpg"
    assert result["summary"]["validation"]["images"] == 2
    assert result["summary"]["test"]["images"] == 2


def test_prepare_manifest_is_attributed_portable_and_deterministic(
    tmp_path: Path,
) -> None:
    metadata_path, images_dir, config_path, metadata = _create_collection(
        tmp_path
    )
    first_output = tmp_path / "first" / "manifest.csv"
    first = prepare_manifest(
        metadata_path=metadata_path,
        images_dir=images_dir,
        config_path=config_path,
        output_path=first_output,
    )

    shuffled_path = tmp_path / "metadata-shuffled.csv"
    metadata.sample(frac=1, random_state=91).to_csv(
        shuffled_path,
        index=False,
        lineterminator="\n",
    )
    second = prepare_manifest(
        metadata_path=shuffled_path,
        images_dir=images_dir,
        config_path=config_path,
        output_path=tmp_path / "second" / "manifest.csv",
    )

    assert first["manifest_sha256"] == second["manifest_sha256"]
    frame = pd.read_csv(first_output)
    assert canonical_manifest_sha256(frame) == first["manifest_sha256"]
    absolute_paths = frame["image_path"].map(
        lambda value: Path(value).is_absolute()
    )
    assert not absolute_paths.any()
    assert set(frame["target"]) == {0, 1}
    assert set(frame["curated_label"]) == {
        "curated_negative",
        "curated_positive",
    }
    assert frame["license"].eq("CC-BY-NC-4.0").all()
    assert frame["attribution"].str.len().min() > 0
    assert frame["perceptual_hash"].str.fullmatch(r"[0-9a-f]{16}").all()
    assert pd.read_csv(
        artifact_path(first_output, "near_duplicates.csv")
    ).empty

    dataset_card = json.loads(
        artifact_path(first_output, "dataset.json").read_text(encoding="utf-8")
    )
    assert dataset_card["records"] == 20
    assert dataset_card["groups"] == 10
    assert dataset_card["target_counts"] == {"0": 10, "1": 10}
    assert Path(f"{first_output}.sha256").read_text().strip() == first[
        "manifest_sha256"
    ]


def test_prepare_manifest_preserves_per_image_provenance(
    tmp_path: Path,
) -> None:
    metadata_path, images_dir, config_path, metadata = _create_collection(
        tmp_path,
        groups=1,
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["columns"].update(
        {
            "source_url": "row_source_url",
            "license": "row_license",
            "license_url": "row_license_url",
            "attribution": "row_attribution",
        }
    )
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    metadata["row_source_url"] = [
        "https://example.test/source/benign",
        "https://example.test/source/malignant",
    ]
    metadata["row_license"] = ["CC-0", "CC-BY-NC"]
    metadata["row_license_url"] = [
        "https://creativecommons.org/publicdomain/zero/1.0/",
        "https://creativecommons.org/licenses/by-nc/4.0/",
    ]
    metadata["row_attribution"] = ["Contributor A", "Contributor B"]
    metadata.to_csv(metadata_path, index=False, lineterminator="\n")
    output = tmp_path / "manifest.csv"

    prepare_manifest(
        metadata_path=metadata_path,
        images_dir=images_dir,
        config_path=config_path,
        output_path=output,
    )

    frame = pd.read_csv(output).sort_values("image_name")
    assert frame["license"].tolist() == ["CC-0", "CC-BY-NC"]
    assert frame["attribution"].tolist() == ["Contributor A", "Contributor B"]
    assert frame["source_url"].str.startswith("https://example.test/source/").all()


def test_prepare_manifest_reports_exclusions(tmp_path: Path) -> None:
    metadata_path, images_dir, config_path, metadata = _create_collection(
        tmp_path,
        groups=2,
    )
    metadata.loc[0, "patient_id"] = ""
    metadata.loc[1, "benign_malignant"] = "indeterminate"
    metadata.loc[2, "image_name"] = "DOES_NOT_EXIST"
    metadata.to_csv(metadata_path, index=False, lineterminator="\n")
    output = tmp_path / "artifacts" / "manifest.csv"

    with pytest.raises(ManifestValidationError, match="Excluded 3 records"):
        prepare_manifest(
            metadata_path=metadata_path,
            images_dir=images_dir,
            config_path=config_path,
            output_path=output,
        )

    exclusions = pd.read_csv(artifact_path(output, "exclusions.csv"))
    assert set(exclusions["reason"]) == {
        "missing_group_id",
        "unmapped_label:indeterminate",
        "image_not_found",
    }
    assert not output.exists()


def test_prepare_manifest_applies_attributed_duplicate_pairs(
    tmp_path: Path,
) -> None:
    metadata_path, images_dir, config_path, metadata = _create_collection(
        tmp_path,
        groups=2,
    )
    metadata.loc[1, "diagnosis"] = "nevus"
    metadata.loc[1, "benign_malignant"] = "benign"
    metadata.to_csv(metadata_path, index=False, lineterminator="\n")
    duplicate = images_dir / "ISIC_TEST_0001.png"
    duplicate.write_bytes((images_dir / "ISIC_TEST_0000.png").read_bytes())
    pairs_path = tmp_path / "official-duplicates.csv"
    pd.DataFrame.from_records(
        [
            {
                "image_name_1": "ISIC_TEST_0000",
                "image_name_2": "ISIC_TEST_0001",
            }
        ]
    ).to_csv(pairs_path, index=False, lineterminator="\n")
    output = tmp_path / "artifacts" / "manifest.csv"

    result = prepare_manifest(
        metadata_path=metadata_path,
        images_dir=images_dir,
        config_path=config_path,
        output_path=output,
        duplicate_pairs_path=pairs_path,
    )

    manifest = pd.read_csv(result["manifest"])
    assert set(manifest["image_name"]) == {
        "ISIC_TEST_0000",
        "ISIC_TEST_0002",
        "ISIC_TEST_0003",
    }
    exclusions = pd.read_csv(result["exclusions"])
    assert exclusions.to_dict("records") == [
        {
            "row_number": 1,
            "image_name": "ISIC_TEST_0001",
            "reason": "source_duplicate_of:ISIC_TEST_0000",
        }
    ]
    dataset_card = json.loads(
        artifact_path(output, "dataset.json").read_text(encoding="utf-8")
    )
    assert dataset_card["records"] == 3
    assert dataset_card["unexpected_exclusions"] == 0
    assert dataset_card["source_duplicate_exclusions"] == 1
    assert dataset_card["source_duplicate_pairs"] == {
        "file": pairs_path.name,
        "sha256": hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
        "pairs": 1,
        "excluded_records": 1,
        "policy": "retain_image_name_1_exclude_image_name_2",
    }


def test_prepare_manifest_rejects_conflicting_duplicate_pair_labels(
    tmp_path: Path,
) -> None:
    metadata_path, images_dir, config_path, _ = _create_collection(
        tmp_path,
        groups=2,
    )
    pairs_path = tmp_path / "official-duplicates.csv"
    pd.DataFrame.from_records(
        [
            {
                "image_name_1": "ISIC_TEST_0000",
                "image_name_2": "ISIC_TEST_0001",
            }
        ]
    ).to_csv(pairs_path, index=False, lineterminator="\n")

    with pytest.raises(
        ManifestValidationError,
        match="conflicting source labels",
    ):
        prepare_manifest(
            metadata_path=metadata_path,
            images_dir=images_dir,
            config_path=config_path,
            output_path=tmp_path / "artifacts" / "manifest.csv",
            duplicate_pairs_path=pairs_path,
        )


def test_prepare_manifest_blocks_exact_duplicates(tmp_path: Path) -> None:
    metadata_path, images_dir, config_path, _ = _create_collection(
        tmp_path,
        groups=2,
    )
    first = images_dir / "ISIC_TEST_0000.png"
    duplicate = images_dir / "ISIC_TEST_0001.png"
    duplicate.write_bytes(first.read_bytes())
    output = tmp_path / "artifacts" / "manifest.csv"

    with pytest.raises(
        ManifestValidationError,
        match="Exact duplicate images",
    ):
        prepare_manifest(
            metadata_path=metadata_path,
            images_dir=images_dir,
            config_path=config_path,
            output_path=output,
        )

    duplicates = pd.read_csv(artifact_path(output, "duplicates.csv"))
    assert set(duplicates["image_name"]) == {
        "ISIC_TEST_0000",
        "ISIC_TEST_0001",
    }
    assert not output.exists()


def test_prepare_manifest_can_deduplicate_within_group_and_label(
    tmp_path: Path,
) -> None:
    metadata_path, images_dir, config_path, metadata = _create_collection(
        tmp_path,
        groups=2,
    )
    metadata.loc[1, "diagnosis"] = "nevus"
    metadata.loc[1, "benign_malignant"] = "benign"
    metadata.to_csv(metadata_path, index=False, lineterminator="\n")
    duplicate = images_dir / "ISIC_TEST_0001.png"
    duplicate.write_bytes((images_dir / "ISIC_TEST_0000.png").read_bytes())
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["integrity"][
        "exact_duplicate_policy"
    ] = "exclude_within_group_and_label"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    output = tmp_path / "artifacts" / "manifest.csv"

    result = prepare_manifest(
        metadata_path=metadata_path,
        images_dir=images_dir,
        config_path=config_path,
        output_path=output,
    )

    manifest = pd.read_csv(result["manifest"])
    assert set(manifest["image_name"]) == {
        "ISIC_TEST_0000",
        "ISIC_TEST_0002",
        "ISIC_TEST_0003",
    }
    exclusions = pd.read_csv(result["exclusions"])
    assert exclusions.iloc[0]["reason"] == (
        "exact_duplicate_of:ISIC_TEST_0000"
    )
    dataset_card = json.loads(
        artifact_path(output, "dataset.json").read_text(encoding="utf-8")
    )
    assert dataset_card["exact_duplicate_records"] == 2
    assert dataset_card["exact_duplicate_groups"] == 1
    assert dataset_card["exact_duplicate_exclusions"] == 1


def test_prepare_manifest_blocks_cross_group_near_duplicates(
    tmp_path: Path,
) -> None:
    metadata_path, images_dir, config_path, _ = _create_collection(
        tmp_path,
        groups=3,
    )
    source = images_dir / "ISIC_TEST_0000.png"
    near_duplicate = images_dir / "ISIC_TEST_0002.png"
    with Image.open(source) as image:
        image.save(near_duplicate, compress_level=1)
    assert source.read_bytes() != near_duplicate.read_bytes()
    output = tmp_path / "artifacts" / "manifest.csv"

    with pytest.raises(
        ManifestValidationError,
        match="Near-duplicate images detected across patient groups",
    ):
        prepare_manifest(
            metadata_path=metadata_path,
            images_dir=images_dir,
            config_path=config_path,
            output_path=output,
        )

    report = pd.read_csv(artifact_path(output, "near_duplicates.csv"))
    match = report.loc[
        report["left_image_name"].eq("ISIC_TEST_0000")
        & report["right_image_name"].eq("ISIC_TEST_0002")
    ]
    assert len(match) == 1
    assert int(match.iloc[0]["hamming_distance"]) == 0
    assert not bool(match.iloc[0]["same_group"])
    assert not output.exists()


def test_prepare_manifest_records_train_only_cross_group_candidates(
    tmp_path: Path,
) -> None:
    metadata_path, images_dir, config_path, _ = _create_collection(
        tmp_path,
        groups=3,
    )
    source = images_dir / "ISIC_TEST_0000.png"
    candidate = images_dir / "ISIC_TEST_0002.png"
    with Image.open(source) as image:
        image.save(candidate, compress_level=1)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["dataset"]["split_policy"] = "train_only"
    config["integrity"]["cross_group_near_duplicate_policy"] = (
        "allow_train_only_reviewed_candidates"
    )
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    output = tmp_path / "artifacts" / "manifest.csv"

    result = prepare_manifest(
        metadata_path=metadata_path,
        images_dir=images_dir,
        config_path=config_path,
        output_path=output,
    )

    dataset_card = json.loads(
        Path(result["dataset_card"]).read_text(encoding="utf-8")
    )
    assert output.exists()
    assert dataset_card["cross_group_near_duplicate_pairs"] >= 1
    assert dataset_card["train_only_near_duplicate_candidates_allowed"]


def test_same_group_near_duplicates_stay_in_one_split(tmp_path: Path) -> None:
    metadata_path, images_dir, config_path, _ = _create_collection(tmp_path)
    source = images_dir / "ISIC_TEST_0000.png"
    same_group_candidate = images_dir / "ISIC_TEST_0001.png"
    with Image.open(source) as image:
        image.save(same_group_candidate, compress_level=1)
    manifest = tmp_path / "manifest.csv"
    prepared = prepare_manifest(
        metadata_path=metadata_path,
        images_dir=images_dir,
        config_path=config_path,
        output_path=manifest,
    )
    prepared_candidates = pd.read_csv(prepared["near_duplicates"])
    assert len(prepared_candidates) == 1
    assert bool(prepared_candidates.iloc[0]["same_group"])

    split_output = tmp_path / "manifest-split.csv"
    split_result = split_manifest(
        manifest_path=manifest,
        output_path=split_output,
    )
    split_candidates = pd.read_csv(split_result["near_duplicates"])
    assert len(split_candidates) == 1
    assert (
        split_candidates.iloc[0]["left_split"]
        == split_candidates.iloc[0]["right_split"]
    )


def test_grouped_split_is_leak_free_and_order_independent(
    tmp_path: Path,
) -> None:
    metadata_path, images_dir, config_path, _ = _create_collection(tmp_path)
    manifest_path = tmp_path / "manifest.csv"
    prepare_manifest(
        metadata_path=metadata_path,
        images_dir=images_dir,
        config_path=config_path,
        output_path=manifest_path,
    )
    first_output = tmp_path / "split-first.csv"
    first = split_manifest(
        manifest_path=manifest_path,
        output_path=first_output,
    )

    shuffled_manifest = tmp_path / "manifest-shuffled.csv"
    pd.read_csv(manifest_path).sample(frac=1, random_state=17).to_csv(
        shuffled_manifest,
        index=False,
        lineterminator="\n",
    )
    second_output = tmp_path / "split-second.csv"
    second = split_manifest(
        manifest_path=shuffled_manifest,
        output_path=second_output,
    )

    assert first["manifest_sha256"] == second["manifest_sha256"]
    frame = pd.read_csv(first_output)
    assert frame.groupby("group_id")["split"].nunique().max() == 1
    assert frame.groupby("sha256")["split"].nunique().max() == 1
    assert set(frame["split"]) == {"train", "validation", "test"}
    assert frame.groupby("split")["target"].nunique().eq(2).all()

    report = json.loads(
        artifact_path(first_output, "report.json").read_text(encoding="utf-8")
    )
    prepared_sha256 = Path(f"{manifest_path}.sha256").read_text().strip()
    assert report["source_manifest_sha256"] == prepared_sha256
    assert report["split_manifest_sha256"] == first["manifest_sha256"]
    assert report["group_leakage_records"] == 0
    assert report["hash_leakage_records"] == 0
    assert report["near_duplicate_split_leakage_records"] == 0


def test_config_requires_nonempty_attribution_and_both_targets(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path / "curation.yaml", 2)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["dataset"]["attribution"] = ""
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="Blank dataset fields"):
        load_curation_config(config_path)

    config["dataset"]["attribution"] = "Required attribution."
    config["label_mapping"]["values"].pop("malignant")
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="both target 0"):
        load_curation_config(config_path)


def test_prepare_manifest_rejects_hash_runtime_version_mismatch(
    tmp_path: Path,
) -> None:
    metadata_path, images_dir, config_path, _ = _create_collection(
        tmp_path,
        groups=2,
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["integrity"]["perceptual_hash"]["implementation_version"] = "0.0.0"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(
        ManifestValidationError,
        match="runtime version mismatch",
    ):
        prepare_manifest(
            metadata_path=metadata_path,
            images_dir=images_dir,
            config_path=config_path,
            output_path=tmp_path / "manifest.csv",
        )


def test_split_rejects_absolute_paths(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "image_name": [f"image-{index}" for index in range(10)],
            "image_path": [f"/private/image-{index}.jpg" for index in range(10)],
            "sha256": [f"{index:064x}" for index in range(10)],
            "perceptual_hash": [f"{index * 104729:016x}" for index in range(10)],
            "perceptual_hash_algorithm": ["phash"] * 10,
            "perceptual_hash_implementation": ["ImageHash"] * 10,
            "perceptual_hash_implementation_version": ["4.3.2"] * 10,
            "perceptual_hash_bits": [64] * 10,
            "near_duplicate_hamming_threshold": [6] * 10,
            "group_id": [f"group-{index}" for index in range(10)],
            "target": [index % 2 for index in range(10)],
        }
    )
    manifest = tmp_path / "manifest.csv"
    frame.to_csv(manifest, index=False)
    with pytest.raises(ManifestValidationError, match="must be relative"):
        split_manifest(
            manifest_path=manifest,
            output_path=tmp_path / "split.csv",
        )


def test_near_duplicate_index_matches_brute_force() -> None:
    values = [
        0x0000000000000000,
        0x0000000000000001,
        0x000000000000003F,
        0x000000000000007F,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFE,
        0x0F0F0F0F0F0F0F0F,
        0xF0F0F0F0F0F0F0F0,
    ]
    frame = pd.DataFrame(
        {
            "image_name": [f"image-{index}" for index in range(len(values))],
            "sha256": [f"{index + 100:064x}" for index in range(len(values))],
            "perceptual_hash": [f"{value:016x}" for value in values],
            "group_id": [f"group-{index}" for index in range(len(values))],
        }
    )
    result = find_near_duplicate_pairs(frame, threshold=6)
    actual = {
        (row.left_image_name, row.right_image_name)
        for row in result.itertuples()
    }
    expected = {
        (f"image-{left}", f"image-{right}")
        for left in range(len(values))
        for right in range(left + 1, len(values))
        if (values[left] ^ values[right]).bit_count() <= 6
    }
    assert actual == expected


def test_training_preflight_detects_image_tampering(tmp_path: Path) -> None:
    metadata_path, images_dir, config_path, _ = _create_collection(tmp_path)
    manifest = tmp_path / "manifest.csv"
    prepare_manifest(
        metadata_path=metadata_path,
        images_dir=images_dir,
        config_path=config_path,
        output_path=manifest,
    )
    split_manifest_path = tmp_path / "manifest-split.csv"
    split_manifest(
        manifest_path=manifest,
        output_path=split_manifest_path,
    )

    report = validate_training_input(
        manifest_path=split_manifest_path,
        images_dir=images_dir,
        verify_image_hashes=True,
    )
    assert report["records"] == 20
    assert report["verified_images"] == 20
    assert report["image_hashes_verified"]

    validation_report = validate_training_input(
        manifest_path=split_manifest_path,
        images_dir=images_dir,
        verify_image_hashes=True,
        image_hash_splits=("validation",),
    )
    assert validation_report["verified_images"] == validation_report[
        "split_counts"
    ]["validation"]
    assert validation_report["image_hash_splits"] == ["validation"]

    tampered = images_dir / "ISIC_TEST_0000.png"
    with Image.open(tampered) as image:
        changed = image.copy()
    changed.putpixel((0, 0), (0, 0, 0))
    changed.save(tampered)
    with pytest.raises(ManifestValidationError, match="SHA-256 mismatch"):
        validate_training_input(
            manifest_path=split_manifest_path,
            images_dir=images_dir,
            verify_image_hashes=True,
        )


def test_training_preflight_allows_cross_group_candidates_only_within_split(
    tmp_path: Path,
) -> None:
    metadata_path, images_dir, config_path, _ = _create_collection(tmp_path)
    manifest = tmp_path / "manifest.csv"
    prepare_manifest(
        metadata_path=metadata_path,
        images_dir=images_dir,
        config_path=config_path,
        output_path=manifest,
    )
    split_manifest_path = tmp_path / "manifest-split.csv"
    split_manifest(
        manifest_path=manifest,
        output_path=split_manifest_path,
    )
    frame = read_manifest(split_manifest_path)
    train = (
        frame.loc[frame["split"].eq("train")]
        .drop_duplicates("group_id")
        .head(2)
    )
    assert train["group_id"].nunique() == 2
    candidate = pd.DataFrame(
        [
            {
                "left_image_name": train.iloc[0]["image_name"],
                "right_image_name": train.iloc[1]["image_name"],
                "left_split": "train",
                "right_split": "train",
                "same_group": False,
            }
        ]
    )
    near_path = artifact_path(split_manifest_path, "near_duplicates.csv")
    candidate.to_csv(near_path, index=False, lineterminator="\n")

    report = validate_training_input(
        manifest_path=split_manifest_path,
        images_dir=images_dir,
        verify_image_hashes=False,
    )
    assert report["cross_group_near_duplicate_candidates"] == 1

    candidate.loc[0, "right_split"] = "validation"
    candidate.to_csv(near_path, index=False, lineterminator="\n")
    with pytest.raises(
        ManifestValidationError,
        match="near-duplicate split leakage",
    ):
        validate_training_input(
            manifest_path=split_manifest_path,
            images_dir=images_dir,
            verify_image_hashes=False,
        )
