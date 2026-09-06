from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .artifacts import require_absent, write_json_exclusive
from .common import sha256_file
from .download_histopath_mel_nevus_subset import _download_one
from .manifest import canonical_manifest_sha256, read_manifest


COLLECTION_ID = 289
COLLECTION_URL = "https://api.isic-archive.com/collections/289/"


def _select_controls(
    metadata_path: Path,
    reference_manifest_path: Path,
    demo_manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    metadata = pd.read_csv(metadata_path, keep_default_na=False)
    reference = read_manifest(reference_manifest_path)
    demo = pd.read_csv(demo_manifest_path, keep_default_na=False)
    required = {
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
    if not required.issubset(metadata.columns):
        raise RuntimeError("MSK-1 metadata is incomplete")
    if metadata["image_name"].duplicated().any():
        raise RuntimeError("MSK-1 image identifiers are not unique")
    if "isic_id" not in demo.columns:
        raise RuntimeError("Demo manifest is missing ISIC identifiers")

    reference_patients = set(reference["patient_id"].dropna().astype(str))
    reference_patients.discard("")
    blocked_images = set(reference["image_name"].astype(str)) | set(
        demo["isic_id"].astype(str)
    )
    eligible = metadata.loc[
        metadata["patient_id"].ne("")
        & metadata["lesion_id"].ne("")
        & metadata["diagnosis_1"].eq("Benign")
        & metadata["image_type"].eq("dermoscopic")
        & metadata["copyright_license"].eq("CC-0")
        & ~metadata["patient_id"].isin(reference_patients)
        & ~metadata["image_name"].isin(blocked_images)
    ].copy()
    selected = (
        eligible.sort_values(["lesion_id", "image_name"], kind="mergesort")
        .drop_duplicates("lesion_id", keep="first")
        .sort_values("image_name", kind="mergesort")
        .reset_index(drop=True)
    )
    if selected.empty:
        raise RuntimeError("Patient-safe MSK-1 benign subset is empty")
    if set(selected["patient_id"].astype(str)) & reference_patients:
        raise RuntimeError("MSK-1 controls overlap reference patients")
    if set(selected["image_name"].astype(str)) & blocked_images:
        raise RuntimeError("MSK-1 controls overlap protected images")

    selected["target"] = 0
    selected["image_path"] = (
        "isic-msk1-benign-controls/" + selected["image_name"] + ".jpg"
    )
    audit = {
        "metadata_records": int(len(metadata)),
        "eligible_records_before_one_per_lesion": int(len(eligible)),
        "selected_records": int(len(selected)),
        "selected_patients": int(selected["patient_id"].nunique()),
        "selected_lesions": int(selected["lesion_id"].nunique()),
        "histopathology_confirmed": int(
            selected["diagnosis_confirm_type"].eq("histopathology").sum()
        ),
        "expert_consensus_confirmed": int(
            selected["diagnosis_confirm_type"]
            .eq("single image expert consensus")
            .sum()
        ),
        "expected_bytes_from_api_snapshot": int(
            selected["full_image_size"].sum()
        ),
    }
    return selected, audit


def download_controls(
    *,
    metadata_path: str | Path,
    reference_manifest_path: str | Path,
    demo_manifest_path: str | Path,
    images_dir: str | Path,
    metadata_output_path: str | Path,
    workers: int,
) -> dict[str, object]:
    if workers <= 0:
        raise ValueError("Download workers must be positive")
    metadata_file = Path(metadata_path).resolve()
    reference_manifest = Path(reference_manifest_path).resolve()
    demo_manifest = Path(demo_manifest_path).resolve()
    images_root = Path(images_dir).resolve()
    output = Path(metadata_output_path)
    image_receipt_path = output.with_name(f"{output.stem}.images.csv")
    receipt_path = output.with_name(f"{output.stem}.receipt.json")
    require_absent([output, image_receipt_path, receipt_path])

    selected, audit = _select_controls(
        metadata_file,
        reference_manifest,
        demo_manifest,
    )
    images_root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = []
        rows = list(selected.itertuples(index=False))
        for position, record in enumerate(
            executor.map(lambda row: _download_one(row, images_root), rows),
            start=1,
        ):
            records.append(record)
            if position == 1 or position % 25 == 0 or position == len(rows):
                print(
                    f"Downloaded and verified MSK-1 benign control "
                    f"{position:,}/{len(rows):,}",
                    flush=True,
                )

    image_receipt = pd.DataFrame.from_records(records)
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
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        selected_output.to_csv(destination, index=False, lineterminator="\n")
    with image_receipt_path.open("x", encoding="utf-8") as destination:
        image_receipt.to_csv(destination, index=False, lineterminator="\n")

    reference = read_manifest(reference_manifest)
    receipt: dict[str, object] = {
        "schema_version": 1,
        "status": "downloaded_pending_cross_corpus_curation",
        "source": "ISIC Archive collection 289 API metadata snapshot",
        "source_url": COLLECTION_URL,
        "collection_id": COLLECTION_ID,
        "metadata_sha256": sha256_file(metadata_file),
        "reference_manifest_sha256": canonical_manifest_sha256(reference),
        "demo_manifest_sha256": sha256_file(demo_manifest),
        "selection": audit,
        "license": "CC-0",
        "license_url": (
            "https://creativecommons.org/publicdomain/zero/1.0/"
        ),
        "downloaded_bytes": int(image_receipt["bytes"].sum()),
        "api_size_mismatch_records": int(
            (~image_receipt["api_size_matches_s3"]).sum()
        ),
        "curated_metadata_sha256": sha256_file(output),
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
    receipt = download_controls(
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
