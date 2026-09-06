from __future__ import annotations

import argparse
import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd

from .artifacts import require_absent, write_json_exclusive
from .common import sha256_file
from .fetch_isic_collection_metadata import TLS_CONTEXT
from .manifest import canonical_manifest_sha256, read_manifest


ALLOWED_LICENSES = {
    "CC-0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC-BY": "https://creativecommons.org/licenses/by/4.0/",
    "CC-BY-NC": "https://creativecommons.org/licenses/by-nc/4.0/",
}


def _eligible_frame(
    metadata_path: Path,
    reference_manifest_path: Path,
    demo_manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    metadata = pd.read_csv(metadata_path, keep_default_na=False)
    reference = read_manifest(reference_manifest_path)
    demo = pd.read_csv(demo_manifest_path, keep_default_na=False)
    required_columns = {
        "image_name",
        "patient_id",
        "lesion_id",
        "diagnosis_1",
        "diagnosis_3",
        "diagnosis_confirm_type",
        "image_type",
        "copyright_license",
        "attribution",
        "full_image_url",
        "full_image_size",
    }
    if not required_columns.issubset(metadata.columns):
        raise RuntimeError("Histopathology metadata is incomplete")
    if metadata["image_name"].duplicated().any():
        raise RuntimeError("Histopathology metadata image names are not unique")
    if "isic_id" not in demo.columns:
        raise RuntimeError("Demo manifest is missing ISIC identifiers")

    known_patient = metadata["patient_id"].ne("")
    binary_diagnosis = metadata["diagnosis_1"].isin(("Benign", "Malignant"))
    dermoscopic = metadata["image_type"].eq("dermoscopic")
    eligible = metadata.loc[
        known_patient & binary_diagnosis & dermoscopic
    ].copy()
    held_out_patients = set(
        reference.loc[
            reference["split"].isin(("validation", "test")),
            "patient_id",
        ].astype(str)
    )
    reference_images = set(reference["image_name"].astype(str))
    demo_images = set(demo["isic_id"].astype(str))
    excluded_held_patient = eligible["patient_id"].astype(str).isin(
        held_out_patients
    )
    excluded_reference_image = eligible["image_name"].astype(str).isin(
        reference_images
    )
    excluded_demo_image = eligible["image_name"].astype(str).isin(demo_images)
    selected = eligible.loc[
        ~excluded_held_patient
        & ~excluded_reference_image
        & ~excluded_demo_image
    ].copy()
    if selected.empty:
        raise RuntimeError("Patient-safe histopathology subset is empty")
    if set(selected["copyright_license"]) - set(ALLOWED_LICENSES):
        raise RuntimeError("Histopathology subset contains an unsupported license")
    if set(selected["patient_id"].astype(str)) & held_out_patients:
        raise RuntimeError("Histopathology subset leaks held-out patients")
    if set(selected["image_name"].astype(str)) & (
        reference_images | demo_images
    ):
        raise RuntimeError("Histopathology subset leaks held-out images")
    for row in selected.itertuples(index=False):
        parsed = urlparse(str(row.full_image_url))
        if (
            parsed.scheme != "https"
            or parsed.netloc != "isic-archive.s3.amazonaws.com"
            or not parsed.path.startswith("/images/")
            or int(row.full_image_size) <= 0
        ):
            raise RuntimeError(f"Unexpected ISIC image source: {row.image_name}")

    selected["target"] = selected["diagnosis_1"].eq("Malignant").astype(int)
    selected["image_path"] = (
        "isic-histopath-mel-nevus/" + selected["image_name"] + ".jpg"
    )
    selected = selected.sort_values("image_name", kind="mergesort").reset_index(
        drop=True
    )
    audit = {
        "metadata_records": int(len(metadata)),
        "known_patient_binary_dermoscopic_records": int(len(eligible)),
        "excluded_held_out_patient": int(excluded_held_patient.sum()),
        "excluded_reference_image": int(excluded_reference_image.sum()),
        "excluded_demo_image": int(excluded_demo_image.sum()),
        "selected_records": int(len(selected)),
        "selected_patients": int(selected["patient_id"].nunique()),
        "selected_benign": int((selected["target"] == 0).sum()),
        "selected_malignant": int((selected["target"] == 1).sum()),
        "selected_expected_bytes": int(selected["full_image_size"].sum()),
    }
    return selected, audit


def _download_one(row: Any, images_root: Path) -> dict[str, Any]:
    image_path = images_root / f"{row.image_name}.jpg"
    api_bytes = int(row.full_image_size)
    request_headers = {"User-Agent": "Express-Derm-Research/1.0"}
    if image_path.exists():
        request = Request(
            str(row.full_image_url),
            method="HEAD",
            headers=request_headers,
        )
        with urlopen(
            request,
            timeout=120,
            context=TLS_CONTEXT,
        ) as response:
            content_length = int(response.headers["Content-Length"])
            etag = str(response.headers.get("ETag", "")).strip('"')
            last_modified = str(response.headers.get("Last-Modified", ""))
        if not image_path.is_file() or image_path.stat().st_size != content_length:
            raise RuntimeError(f"Existing image has unexpected size: {image_path}")
        md5 = hashlib.md5()  # noqa: S324 - S3 ETag integrity, not security
        sha256 = hashlib.sha256()
        with image_path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                md5.update(chunk)
                sha256.update(chunk)
        if re.fullmatch(r"[0-9a-fA-F]{32}", etag) and (
            md5.hexdigest().lower() != etag.lower()
        ):
            raise RuntimeError(f"Existing image ETag mismatch: {image_path}")
    else:
        part_path = image_path.with_suffix(".jpg.part")
        for attempt in range(5):
            if part_path.exists():
                part_path.unlink()
            try:
                request = Request(
                    str(row.full_image_url),
                    headers=request_headers,
                )
                with urlopen(
                    request,
                    timeout=120,
                    context=TLS_CONTEXT,
                ) as response, part_path.open("xb") as output:
                    content_length = int(response.headers["Content-Length"])
                    etag = str(response.headers.get("ETag", "")).strip('"')
                    last_modified = str(
                        response.headers.get("Last-Modified", "")
                    )
                    md5 = hashlib.md5()  # noqa: S324 - S3 ETag integrity
                    sha256 = hashlib.sha256()
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        md5.update(chunk)
                        sha256.update(chunk)
                if part_path.stat().st_size != content_length:
                    raise RuntimeError("Downloaded image size mismatch")
                if re.fullmatch(r"[0-9a-fA-F]{32}", etag) and (
                    md5.hexdigest().lower() != etag.lower()
                ):
                    raise RuntimeError("Downloaded image ETag mismatch")
                part_path.replace(image_path)
                break
            except Exception:
                if part_path.exists():
                    part_path.unlink()
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
    return {
        "image_name": str(row.image_name),
        "image_path": str(row.image_path),
        "bytes": int(image_path.stat().st_size),
        "api_snapshot_bytes": api_bytes,
        "api_size_matches_s3": bool(api_bytes == image_path.stat().st_size),
        "s3_etag": etag,
        "s3_last_modified": last_modified,
        "sha256": sha256.hexdigest(),
    }


def download_subset(
    *,
    metadata_path: str | Path,
    reference_manifest_path: str | Path,
    demo_manifest_path: str | Path,
    images_dir: str | Path,
    metadata_output_path: str | Path,
    workers: int,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("Download workers must be positive")
    metadata_file = Path(metadata_path).resolve()
    reference_manifest = Path(reference_manifest_path).resolve()
    demo_manifest = Path(demo_manifest_path).resolve()
    images_root = Path(images_dir).resolve()
    output_path = Path(metadata_output_path)
    image_receipt_path = output_path.with_name(
        f"{output_path.stem}.images.csv"
    )
    receipt_path = output_path.with_name(f"{output_path.stem}.receipt.json")
    require_absent([output_path, image_receipt_path, receipt_path])
    selected, audit = _eligible_frame(
        metadata_file,
        reference_manifest,
        demo_manifest,
    )
    images_root.mkdir(parents=True, exist_ok=True)
    rows = list(selected.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        image_records = []
        for position, record in enumerate(
            executor.map(lambda row: _download_one(row, images_root), rows),
            start=1,
        ):
            image_records.append(record)
            if position == 1 or position % 25 == 0 or position == len(rows):
                print(
                    f"Downloaded and verified histopathology image "
                    f"{position:,}/{len(rows):,}",
                    flush=True,
                )

    image_receipt = pd.DataFrame.from_records(image_records)
    selected_output = selected.loc[
        :,
        [
            "image_name",
            "image_path",
            "patient_id",
            "lesion_id",
            "diagnosis_1",
            "diagnosis_3",
            "diagnosis_confirm_type",
            "target",
            "copyright_license",
            "attribution",
            "full_image_url",
            "full_image_size",
        ],
    ].merge(image_receipt, on=("image_name", "image_path"), validate="one_to_one")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as destination:
        selected_output.to_csv(destination, index=False, lineterminator="\n")
    with image_receipt_path.open("x", encoding="utf-8") as destination:
        image_receipt.to_csv(destination, index=False, lineterminator="\n")

    reference = read_manifest(reference_manifest)
    receipt = {
        "schema_version": 1,
        "status": "downloaded_pending_cross_corpus_curation",
        "source": "ISIC Archive collection 294 API metadata snapshot",
        "source_url": "https://api.isic-archive.com/api/v2/images/search/",
        "metadata_sha256": sha256_file(metadata_file),
        "reference_manifest_sha256": canonical_manifest_sha256(reference),
        "demo_manifest_sha256": sha256_file(demo_manifest),
        "selection": audit,
        "licenses": {
            license_name: {
                "records": int(
                    selected["copyright_license"].eq(license_name).sum()
                ),
                "url": license_url,
            }
            for license_name, license_url in ALLOWED_LICENSES.items()
            if selected["copyright_license"].eq(license_name).any()
        },
        "downloaded_bytes": int(image_receipt["bytes"].sum()),
        "api_size_mismatch_records": int(
            (~image_receipt["api_size_matches_s3"]).sum()
        ),
        "curated_metadata_sha256": sha256_file(output_path),
        "image_receipt_sha256": sha256_file(image_receipt_path),
        "workers": workers,
        "training_authorized": False,
        "research_only": True,
    }
    write_json_exclusive(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--demo-manifest", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = download_subset(
        metadata_path=args.metadata,
        reference_manifest_path=args.reference_manifest,
        demo_manifest_path=args.demo_manifest,
        images_dir=args.images_dir,
        metadata_output_path=args.metadata_output,
        workers=args.workers,
    )
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
