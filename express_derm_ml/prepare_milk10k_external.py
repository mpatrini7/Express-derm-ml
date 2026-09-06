from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from .artifacts import require_absent, write_json_exclusive
from .common import load_yaml, sha256_file
from .integrity import (
    NEAR_DUPLICATE_COLUMNS,
    find_near_duplicate_pairs,
    perceptual_hash_file,
    validate_perceptual_hash_runtime,
)
from .manifest import (
    ManifestValidationError,
    read_manifest,
    write_manifest,
)


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    source_name: str,
) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ManifestValidationError(
            f"{source_name} is missing columns: {sorted(missing)}"
        )


def _verify_source_hashes(root: Path, config: dict[str, Any]) -> None:
    for source_name, source in config["sources"].items():
        path = root / str(source["filename"])
        if not path.is_file():
            raise ManifestValidationError(
                f"MILK10k source file is missing: {path}"
            )
        actual = sha256_file(path)
        if actual != str(source["sha256"]):
            raise ManifestValidationError(
                f"MILK10k {source_name} SHA-256 mismatch"
            )


def prepare_external_manifest(
    *,
    data_root: str | Path,
    reference_manifest_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(data_root).resolve()
    config_file = Path(config_path).resolve()
    output = Path(output_path)
    companion_paths = [
        output,
        Path(f"{output}.sha256"),
        output.with_name(f"{output.stem}.dataset.json"),
        output.with_name(
            f"{output.stem}.reference_near_duplicates.csv"
        ),
    ]
    require_absent(companion_paths)
    config = load_yaml(config_file)
    _verify_source_hashes(root, config)

    sources = config["sources"]
    metadata = pd.read_csv(root / sources["metadata"]["filename"])
    supplement = pd.read_csv(root / sources["supplement"]["filename"])
    ground_truth = pd.read_csv(root / sources["ground_truth"]["filename"])
    classes = [str(value) for value in config["labels"]["classes"]]
    _require_columns(
        metadata,
        {"lesion_id", "isic_id", "image_type", "copyright_license"},
        "MILK10k metadata",
    )
    _require_columns(
        supplement,
        {"isic_id", "diagnosis_full", "diagnosis_confirm_type"},
        "MILK10k supplement",
    )
    _require_columns(
        ground_truth,
        {"lesion_id", *classes},
        "MILK10k ground truth",
    )
    dataset = config["dataset"]
    if len(metadata) != int(dataset["expected_metadata_rows"]):
        raise ManifestValidationError("Unexpected MILK10k metadata row count")
    if len(supplement) != int(dataset["expected_supplement_rows"]):
        raise ManifestValidationError(
            "Unexpected MILK10k supplement row count"
        )
    if len(ground_truth) != int(dataset["expected_lesions"]):
        raise ManifestValidationError("Unexpected MILK10k lesion count")
    if metadata["isic_id"].duplicated().any():
        raise ManifestValidationError("MILK10k image IDs must be unique")
    if supplement["isic_id"].duplicated().any():
        raise ManifestValidationError(
            "MILK10k supplement image IDs must be unique"
        )
    if ground_truth["lesion_id"].duplicated().any():
        raise ManifestValidationError("MILK10k lesion IDs must be unique")
    if not ground_truth[classes].isin([0.0, 1.0]).all(axis=None):
        raise ManifestValidationError("MILK10k labels must be binary")
    if not ground_truth[classes].sum(axis=1).eq(1.0).all():
        raise ManifestValidationError(
            "Every MILK10k lesion must have exactly one diagnostic class"
        )

    evaluation_image_type = str(dataset["evaluation_image_type"])
    selected_images = metadata.loc[
        metadata["image_type"].eq(evaluation_image_type)
    ].copy()
    if len(selected_images) != int(dataset["expected_lesions"]):
        raise ManifestValidationError(
            "Expected one MILK10k evaluation image per lesion for "
            f"image_type={evaluation_image_type!r}"
        )
    if selected_images["lesion_id"].duplicated().any():
        raise ManifestValidationError(
            "MILK10k external set must contain one image per lesion"
        )
    merged = selected_images.merge(
        supplement,
        on="isic_id",
        how="left",
        validate="one_to_one",
    ).merge(
        ground_truth,
        on="lesion_id",
        how="left",
        validate="one_to_one",
    )
    if merged[classes].isna().any(axis=None):
        raise ManifestValidationError("MILK10k labels did not join completely")

    perceptual = config["integrity"]["perceptual_hash"]
    validate_perceptual_hash_runtime(perceptual)
    hash_size = int(perceptual["hash_size"])
    highfreq_factor = int(perceptual["highfreq_factor"])
    threshold = int(config["integrity"]["near_duplicate_hamming_threshold"])
    images_root = root / "MILK10k_Training_Input"
    records: list[dict[str, Any]] = []
    for position, row in enumerate(merged.itertuples(index=False), start=1):
        if position == 1 or position % 500 == 0:
            print(
                f"Scanning MILK10k {evaluation_image_type} image {position:,}/"
                f"{len(merged):,}",
                flush=True,
            )
        image_path = images_root / str(row.lesion_id) / f"{row.isic_id}.jpg"
        if not image_path.is_file():
            raise ManifestValidationError(
                f"MILK10k image is missing: {image_path}"
            )
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
        class_name = next(
            class_name
            for class_name in classes
            if float(getattr(row, class_name)) == 1.0
        )
        records.append(
            {
                "image_name": str(row.isic_id),
                "image_path": image_path.relative_to(root).as_posix(),
                "lesion_id": str(row.lesion_id),
                "group_id": str(row.lesion_id),
                "patient_id": "",
                "patient_identifier_status": dataset[
                    "patient_identifier_status"
                ],
                "image_type": evaluation_image_type,
                "diagnosis_class": class_name,
                "diagnosis_full": str(row.diagnosis_full),
                "diagnosis_confirm_type": str(row.diagnosis_confirm_type),
                "target_melanoma": int(
                    class_name in config["labels"]["melanoma_positive"]
                ),
                "target_broad_malignancy": int(
                    class_name
                    in config["labels"]["broad_malignancy_positive"]
                ),
                "collection_id": dataset["id"],
                "dataset_version": dataset["version"],
                "source_url": dataset["source_url"],
                "license": str(row.copyright_license),
                "license_url": dataset["license_url"],
                "attribution": dataset["attribution"],
                "width": int(width),
                "height": int(height),
                "sha256": sha256_file(image_path),
                "perceptual_hash": perceptual_hash_file(
                    image_path,
                    hash_size=hash_size,
                    highfreq_factor=highfreq_factor,
                ),
            }
        )

    frame = pd.DataFrame.from_records(records).sort_values(
        "image_name",
        kind="mergesort",
    )
    if frame["sha256"].duplicated().any():
        raise ManifestValidationError(
            "MILK10k external set contains exact duplicate images"
        )
    reference = read_manifest(reference_manifest_path)
    reference_audit = reference.loc[
        :, ["image_name", "group_id", "sha256", "perceptual_hash"]
    ].copy()
    reference_audit["group_id"] = "reference:" + reference_audit[
        "group_id"
    ].astype(str)
    external_audit = frame.loc[
        :, ["image_name", "group_id", "sha256", "perceptual_hash"]
    ].copy()
    external_audit["group_id"] = "external:" + external_audit[
        "group_id"
    ].astype(str)
    combined = pd.concat(
        [reference_audit, external_audit],
        ignore_index=True,
    )
    exact_reference_hashes = set(reference_audit["sha256"])
    exact_overlap = frame.loc[frame["sha256"].isin(exact_reference_hashes)]
    if not exact_overlap.empty:
        raise ManifestValidationError(
            "MILK10k external images exactly overlap the training corpus"
        )
    near_duplicates = find_near_duplicate_pairs(
        combined,
        threshold=threshold,
    )
    cross_source = near_duplicates.loc[
        near_duplicates["left_group_id"].str.split(":").str[0]
        != near_duplicates["right_group_id"].str.split(":").str[0]
    ].copy()

    output.parent.mkdir(parents=True, exist_ok=True)
    cross_source.to_csv(
        output.with_name(f"{output.stem}.reference_near_duplicates.csv"),
        index=False,
        columns=NEAR_DUPLICATE_COLUMNS,
        lineterminator="\n",
    )
    if not cross_source.empty:
        raise ManifestValidationError(
            "MILK10k has perceptual candidates against the training corpus; "
            "review the cross-source report before external evaluation"
        )
    manifest_sha256 = write_manifest(frame, output)
    card = {
        "schema_version": 1,
        "dataset": dataset,
        "records": int(len(frame)),
        "lesions": int(frame["lesion_id"].nunique()),
        "melanoma_positive": int(frame["target_melanoma"].sum()),
        "broad_malignancy_positive": int(
            frame["target_broad_malignancy"].sum()
        ),
        "patient_identifier_status": dataset["patient_identifier_status"],
        "training_authorized": False,
        "purpose": "external_research_evaluation_only",
        "evaluation_image_type": evaluation_image_type,
        "config_sha256": sha256_file(config_file),
        "manifest_sha256": manifest_sha256,
        "reference_manifest_sha256": Path(
            f"{reference_manifest_path}.sha256"
        ).read_text(encoding="ascii").strip(),
        "cross_source_exact_duplicates": 0,
        "cross_source_near_duplicate_candidates": 0,
        "source_hashes": {
            name: source["sha256"] for name, source in sources.items()
        },
    }
    write_json_exclusive(
        output.with_name(f"{output.stem}.dataset.json"),
        card,
    )
    return card


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_external_manifest(
        data_root=args.data_root,
        reference_manifest_path=args.reference_manifest,
        config_path=args.config,
        output_path=args.output,
    )
    print(result)


if __name__ == "__main__":
    main()
