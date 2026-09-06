from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import require_absent, write_json_exclusive
from .common import load_yaml, sha256_file
from .download_isic2019_patient_subset import HTTPRangeReader, _file_crc32
from .manifest import canonical_manifest_sha256, read_manifest


LICENSE_URLS = {
    "CC-0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC-BY": "https://creativecommons.org/licenses/by/4.0/",
    "CC-BY-NC": "https://creativecommons.org/licenses/by-nc/4.0/",
}


def _verified_reference_manifest(path: Path) -> pd.DataFrame:
    frame = read_manifest(path)
    digest_path = Path(f"{path}.sha256")
    if not digest_path.is_file():
        raise RuntimeError(f"Reference manifest digest is missing: {path}")
    expected = digest_path.read_text(encoding="ascii").strip()
    if canonical_manifest_sha256(frame) != expected:
        raise RuntimeError("Reference manifest digest mismatch")
    return frame


def _eligible_frame(
    metadata_path: Path,
    reference_manifest_path: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path, keep_default_na=False)
    dataset = config["dataset"]
    selection = config["selection"]
    if len(metadata) != int(dataset["expected_source_records"]):
        raise RuntimeError("Unexpected DICM-17K metadata record count")
    if metadata["isic_id"].duplicated().any():
        raise RuntimeError("DICM-17K image identifiers must be unique")
    if set(metadata["image_type"]) != {selection["image_type"]}:
        raise RuntimeError("DICM-17K contains an unexpected image type")

    reference = _verified_reference_manifest(reference_manifest_path)
    reference_images = set(reference["image_name"].astype(str))
    reference_patients = set(reference["patient_id"].astype(str))
    candidate = metadata.loc[
        metadata["patient_id"].ne("")
        & ~metadata["isic_id"].isin(reference_images)
        & ~metadata["patient_id"].isin(reference_patients)
        & metadata["diagnosis_confirm_type"].isin(
            selection["accepted_confirmation_types"]
        )
    ].copy()
    positive = (
        candidate["diagnosis_1"].eq(selection["positive_diagnosis_1"])
        & candidate["diagnosis_2"].eq(selection["positive_diagnosis_2"])
    )
    negative = candidate["diagnosis_1"].eq(
        selection["negative_diagnosis_1"]
    )
    selected = candidate.loc[positive | negative].copy()
    selected["diagnosis_class"] = "benign"
    selected.loc[positive.loc[selected.index], "diagnosis_class"] = "melanoma"
    selected["image_name"] = selected["isic_id"]
    selected["source_url"] = str(dataset["source_url"])
    selected["license_url"] = selected["copyright_license"].map(
        LICENSE_URLS
    )

    if selected["license_url"].isna().any():
        unknown = sorted(
            set(selected.loc[selected["license_url"].isna(), "copyright_license"])
        )
        raise RuntimeError(f"Unsupported DICM-17K licenses: {unknown}")
    if selected["attribution"].eq("").any():
        raise RuntimeError("DICM-17K selected records require attribution")
    if len(selected) != int(dataset["expected_subset_records"]):
        raise RuntimeError("Unexpected patient-safe DICM-17K subset count")
    if selected["patient_id"].nunique() != int(dataset["expected_patients"]):
        raise RuntimeError("Unexpected patient-safe DICM-17K patient count")
    if selected["diagnosis_class"].eq("melanoma").sum() != int(
        dataset["expected_melanoma"]
    ):
        raise RuntimeError("Unexpected patient-safe DICM-17K melanoma count")
    expected_licenses = {
        str(key): int(value)
        for key, value in dataset["expected_license_counts"].items()
    }
    actual_licenses = {
        str(key): int(value)
        for key, value in selected["copyright_license"].value_counts().items()
    }
    if actual_licenses != expected_licenses:
        raise RuntimeError("Unexpected patient-safe DICM-17K license counts")
    if set(selected["isic_id"]) & reference_images:
        raise RuntimeError("DICM-17K image IDs overlap the reference corpus")
    if set(selected["patient_id"]) & reference_patients:
        raise RuntimeError("DICM-17K patient IDs overlap the reference corpus")
    return selected.sort_values("isic_id", kind="mergesort")


def download_dicm17k_patient_subset(
    *,
    metadata_path: str | Path,
    images_dir: str | Path,
    reference_manifest_path: str | Path,
    config_path: str | Path,
    metadata_output_path: str | Path,
    archive_file: str | Path | None = None,
) -> dict[str, Any]:
    metadata_source = Path(metadata_path).resolve()
    images_root = Path(images_dir).resolve()
    reference_path = Path(reference_manifest_path).resolve()
    config_file = Path(config_path).resolve()
    metadata_output = Path(metadata_output_path)
    image_receipt_path = metadata_output.with_name(
        f"{metadata_output.stem}.images.csv"
    )
    receipt_path = metadata_output.with_name(
        f"{metadata_output.stem}.receipt.json"
    )
    require_absent([metadata_output, image_receipt_path, receipt_path])
    config = load_yaml(config_file)
    if sha256_file(metadata_source) != str(
        config["sources"]["metadata_sha256"]
    ):
        raise RuntimeError("DICM-17K metadata hash mismatch")
    selected = _eligible_frame(metadata_source, reference_path, config)
    images_root.mkdir(parents=True, exist_ok=True)

    archive_url = str(config["dataset"]["archive_url"])
    local_archive = (
        Path(archive_file).resolve() if archive_file is not None else None
    )
    if local_archive is not None and not local_archive.is_file():
        raise RuntimeError(f"DICM-17K archive is missing: {local_archive}")
    if local_archive is not None and local_archive.stat().st_size != int(
        config["dataset"]["archive_content_length"]
    ):
        raise RuntimeError("DICM-17K local archive size mismatch")
    archive_reader = (
        None
        if local_archive is not None
        else HTTPRangeReader(
            archive_url,
            allowed_path_prefixes=("/dois/10.34970-233480/",),
        )
    )
    image_records: list[dict[str, Any]] = []
    archive_source: Any = local_archive or archive_reader
    with zipfile.ZipFile(archive_source) as archive:
        by_stem: dict[str, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            path = Path(info.filename)
            if path.suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            if path.stem in by_stem:
                raise RuntimeError(
                    f"Duplicate image identifier in DICM-17K ZIP: {path.stem}"
                )
            by_stem[path.stem] = info

        for position, row in enumerate(selected.itertuples(index=False), start=1):
            if position == 1 or position % 25 == 0:
                print(
                    f"Downloading patient-safe DICM-17K image {position}/"
                    f"{len(selected)}",
                    flush=True,
                )
            info = by_stem.get(str(row.isic_id))
            if info is None:
                raise RuntimeError(
                    f"DICM-17K image is absent from bundle: {row.isic_id}"
                )
            image_path = images_root / f"{row.isic_id}.jpg"
            if image_path.exists():
                if (
                    not image_path.is_file()
                    or image_path.stat().st_size != info.file_size
                    or _file_crc32(image_path) != info.CRC
                ):
                    raise RuntimeError(
                        f"Existing image fails ZIP integrity: {image_path}"
                    )
            else:
                part = image_path.with_suffix(".jpg.part")
                if part.exists():
                    part.unlink()
                try:
                    with archive.open(info) as source, part.open("xb") as output:
                        while chunk := source.read(1024 * 1024):
                            output.write(chunk)
                    if (
                        part.stat().st_size != info.file_size
                        or _file_crc32(part) != info.CRC
                    ):
                        raise RuntimeError("Extracted ZIP member failed integrity")
                    part.replace(image_path)
                except Exception:
                    if part.exists():
                        part.unlink()
                    raise
            image_records.append(
                {
                    "image_name": str(row.isic_id),
                    "archive_member": info.filename,
                    "bytes": int(image_path.stat().st_size),
                    "zip_crc32": f"{info.CRC:08x}",
                    "sha256": sha256_file(image_path),
                }
            )

    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    with metadata_output.open("x", encoding="utf-8") as destination:
        selected.to_csv(destination, index=False, lineterminator="\n")
    with image_receipt_path.open("x", encoding="utf-8") as destination:
        pd.DataFrame.from_records(image_records).to_csv(
            destination,
            index=False,
            lineterminator="\n",
        )
    receipt = {
        "schema_version": 1,
        "dataset": config["dataset"],
        "config_sha256": sha256_file(config_file),
        "source_metadata_sha256": sha256_file(metadata_source),
        "curated_metadata_sha256": sha256_file(metadata_output),
        "image_receipt_sha256": sha256_file(image_receipt_path),
        "reference_manifest_sha256": Path(
            f"{reference_path}.sha256"
        ).read_text(encoding="ascii").strip(),
        "records": int(len(selected)),
        "patients": int(selected["patient_id"].nunique()),
        "melanoma_positive": int(
            selected["diagnosis_class"].eq("melanoma").sum()
        ),
        "downloaded_bytes": int(
            sum(record["bytes"] for record in image_records)
        ),
        "archive_url": archive_url,
        "archive_content_length": (
            int(local_archive.stat().st_size)
            if local_archive is not None
            else archive_reader.length
        ),
        "archive_sha256": (
            sha256_file(local_archive) if local_archive is not None else None
        ),
        "archive_etag": (
            str(config["dataset"]["archive_etag"])
            if archive_reader is None
            else archive_reader.etag
        ),
        "archive_last_modified": (
            str(config["dataset"]["archive_last_modified"])
            if archive_reader is None
            else archive_reader.last_modified
        ),
        "training_authorized": False,
        "status": "downloaded_pending_cross_corpus_curation",
    }
    write_json_exclusive(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument(
        "--archive-file",
        help="Use a complete local official bundle instead of HTTP ranges.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = download_dicm17k_patient_subset(
        metadata_path=args.metadata,
        images_dir=args.images_dir,
        reference_manifest_path=args.reference_manifest,
        config_path=args.config,
        metadata_output_path=args.metadata_output,
        archive_file=args.archive_file,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
