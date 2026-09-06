from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd
from PIL import Image

from .artifacts import require_absent, write_json_exclusive
from .common import load_yaml, sha256_file
from .fetch_isic_collection_metadata import TLS_CONTEXT
from .manifest import read_manifest


BROAD_CLASSES = ("BCC", "AK", "SCC")
BENIGN_CLASSES = ("NV", "BKL", "DF", "VASC")


def _validated_sources(
    downloads_dir: Path,
    config: dict[str, Any],
) -> dict[str, Path]:
    paths = {
        name: downloads_dir / str(source["filename"])
        for name, source in config["sources"].items()
    }
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"ISIC 2019 source is missing: {path}")
        expected = str(config["sources"][name]["sha256"])
        if sha256_file(path) != expected:
            raise RuntimeError(f"ISIC 2019 source hash mismatch: {name}")
    return paths


def _validated_merged_frame(
    archive: pd.DataFrame,
    ground_truth: pd.DataFrame,
    challenge: pd.DataFrame,
) -> pd.DataFrame:
    class_columns = [
        "MEL",
        "NV",
        "BCC",
        "AK",
        "BKL",
        "DF",
        "VASC",
        "SCC",
        "UNK",
    ]
    if not ground_truth[class_columns].isin([0.0, 1.0]).all(axis=None):
        raise RuntimeError("ISIC 2019 labels must be binary")
    if not ground_truth[class_columns].sum(axis=1).eq(1.0).all():
        raise RuntimeError("ISIC 2019 labels must be one-hot")
    merged = archive.merge(
        challenge.loc[:, ["image", "lesion_id"]],
        left_on="image_name",
        right_on="image",
        validate="one_to_one",
        suffixes=("_archive", "_challenge"),
    ).merge(
        ground_truth,
        on="image",
        validate="one_to_one",
    )
    return merged


def _validate_selected_records(selected: pd.DataFrame) -> None:
    if selected["lesion_id_challenge"].astype(str).str.strip().eq("").any():
        raise RuntimeError("Broad-attention records require lesion IDs")
    if selected["lesion_id_challenge"].isna().any():
        raise RuntimeError("Broad-attention records require lesion IDs")
    if selected["patient_id"].notna().any():
        raise RuntimeError(
            "Expected selected ISIC 2019 records to have unavailable patient IDs"
        )
    if set(selected["image_type"].astype(str)) != {"dermoscopic"}:
        raise RuntimeError("Broad-attention records must be dermoscopic")
    if set(selected["copyright_license"].astype(str)) != {"CC-BY-NC"}:
        raise RuntimeError("Unexpected broad-attention source license")
    if selected["image_name"].duplicated().any():
        raise RuntimeError("Duplicate broad-attention image names")


def eligible_broad_malignancy_frame(
    archive: pd.DataFrame,
    ground_truth: pd.DataFrame,
    challenge: pd.DataFrame,
) -> pd.DataFrame:
    merged = _validated_merged_frame(archive, ground_truth, challenge)
    selected = merged.loc[merged[list(BROAD_CLASSES)].sum(axis=1).eq(1.0)].copy()
    selected["diagnosis_class"] = selected[list(BROAD_CLASSES)].idxmax(
        axis=1
    ).str.lower()
    selected["selection_role"] = "broad_malignancy_positive"
    _validate_selected_records(selected)
    return selected.sort_values("image_name", kind="mergesort").reset_index(
        drop=True
    )


def eligible_broad_attention_frame(
    archive: pd.DataFrame,
    ground_truth: pd.DataFrame,
    challenge: pd.DataFrame,
    *,
    benign_control_quotas: dict[str, int],
    excluded_image_names: set[str],
    selection_seed: str,
) -> pd.DataFrame:
    merged = _validated_merged_frame(archive, ground_truth, challenge)
    merged = merged.loc[
        ~merged["image_name"].astype(str).isin(excluded_image_names)
        & merged["patient_id"].isna()
        & merged["lesion_id_challenge"].notna()
        & merged["lesion_id_challenge"].astype(str).str.strip().ne("")
        & merged["image_type"].astype(str).eq("dermoscopic")
        & merged["copyright_license"].astype(str).eq("CC-BY-NC")
    ].copy()
    positives = merged.loc[
        merged[list(BROAD_CLASSES)].sum(axis=1).eq(1.0)
    ].copy()
    positives["diagnosis_class"] = positives[list(BROAD_CLASSES)].idxmax(
        axis=1
    ).str.lower()
    positives["selection_role"] = "broad_malignancy_positive"

    controls: list[pd.DataFrame] = []
    if set(benign_control_quotas) != set(BENIGN_CLASSES):
        raise RuntimeError("Benign-control quotas must cover NV, BKL, DF and VASC")
    for diagnosis in BENIGN_CLASSES:
        quota = int(benign_control_quotas[diagnosis])
        if quota <= 0:
            raise RuntimeError("Benign-control quotas must be positive")
        candidates = merged.loc[merged[diagnosis].eq(1.0)].copy()
        candidates["_selection_key"] = candidates["image_name"].map(
            lambda name: hashlib.sha256(
                f"{selection_seed}:{name}".encode("utf-8")
            ).hexdigest()
        )
        candidates = candidates.sort_values(
            ["_selection_key", "image_name"],
            kind="mergesort",
        )
        candidates["_lesion_repeat_rank"] = candidates.groupby(
            "lesion_id_challenge",
            sort=False,
        ).cumcount()
        candidates = candidates.sort_values(
            ["_lesion_repeat_rank", "_selection_key", "image_name"],
            kind="mergesort",
        )
        if len(candidates) < quota:
            raise RuntimeError(
                f"Insufficient {diagnosis} controls: {len(candidates)} < {quota}"
            )
        selected_controls = candidates.head(quota).copy()
        selected_controls["diagnosis_class"] = diagnosis.lower()
        selected_controls["selection_role"] = "same_source_benign_control"
        controls.append(selected_controls)

    selected = pd.concat([positives, *controls], ignore_index=True)
    _validate_selected_records(selected)
    if set(selected["image_name"].astype(str)) & excluded_image_names:
        raise RuntimeError("Broad-attention selection overlaps excluded images")
    return selected.sort_values("image_name", kind="mergesort").reset_index(
        drop=True
    )


def _validate_image_url(url: str) -> None:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "isic-archive.s3.amazonaws.com"
        or not parsed.path.startswith("/images/")
        or not parsed.path.endswith(".jpg")
    ):
        raise RuntimeError(f"Unexpected ISIC image URL: {url}")


def _verified_jpeg_details(path: Path) -> tuple[int, int, int]:
    actual_size = path.stat().st_size
    if actual_size <= 0:
        raise RuntimeError(f"Downloaded image is empty: {path}")
    try:
        with Image.open(path) as image:
            if image.format != "JPEG":
                raise RuntimeError(f"Downloaded image is not JPEG: {path}")
            width, height = image.size
            image.verify()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Downloaded image is invalid: {path}") from exc
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Downloaded image has invalid dimensions: {path}")
    return actual_size, width, height


def _download_image(row, images_dir: Path) -> dict[str, object]:
    url = str(row.full_image_url)
    _validate_image_url(url)
    destination = images_dir / f"{row.image_name}.jpg"
    archive_snapshot_size = int(row.full_image_size)
    if archive_snapshot_size <= 0:
        raise RuntimeError(f"Invalid expected size for {row.image_name}")
    if destination.exists():
        if not destination.is_file():
            raise RuntimeError(f"Existing image is not a file: {destination}")
    else:
        partial = destination.with_suffix(".jpg.part")
        if partial.exists():
            partial.unlink()
        request = Request(
            url,
            headers={"User-Agent": "Express-Derm-Research/1.0"},
        )
        for attempt in range(1, 5):
            try:
                with urlopen(
                    request,
                    timeout=90,
                    context=TLS_CONTEXT,
                ) as response, partial.open("xb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                _verified_jpeg_details(partial)
                partial.replace(destination)
                break
            except (URLError, TimeoutError, ConnectionError):
                if partial.exists():
                    partial.unlink()
                if attempt == 4:
                    raise
                time.sleep(2 ** (attempt - 1))
            except Exception:
                if partial.exists():
                    partial.unlink()
                raise
    actual_size, width, height = _verified_jpeg_details(destination)
    return {
        "image_name": str(row.image_name),
        "bytes": actual_size,
        "archive_snapshot_bytes": archive_snapshot_size,
        "archive_snapshot_size_matches": actual_size == archive_snapshot_size,
        "width": width,
        "height": height,
        "sha256": sha256_file(destination),
        "source_url": url,
    }


def download_broad_malignancy_subset(
    *,
    downloads_dir: str | Path,
    images_dir: str | Path,
    config_path: str | Path,
    metadata_output_path: str | Path,
    excluded_manifest_path: str | Path | None = None,
    workers: int = 8,
) -> dict[str, object]:
    if workers <= 0:
        raise ValueError("Download workers must be positive")
    downloads_root = Path(downloads_dir).resolve()
    images_root = Path(images_dir).resolve()
    images_root.mkdir(parents=True, exist_ok=True)
    config_file = Path(config_path).resolve()
    metadata_output = Path(metadata_output_path).resolve()
    image_receipt = metadata_output.with_name(
        f"{metadata_output.stem}.images.csv"
    )
    receipt_output = metadata_output.with_name(
        f"{metadata_output.stem}.receipt.json"
    )
    require_absent([metadata_output, image_receipt, receipt_output])
    config = load_yaml(config_file)
    sources = _validated_sources(downloads_root, config)
    archive = pd.read_csv(sources["archive_metadata"], low_memory=False)
    ground_truth = pd.read_csv(sources["ground_truth"])
    challenge = pd.read_csv(sources["challenge_metadata"])
    benign_control_quotas = config["dataset"].get("benign_control_quotas")
    if benign_control_quotas is None:
        if excluded_manifest_path is not None:
            raise RuntimeError(
                "An excluded manifest is only valid with benign controls"
            )
        selected = eligible_broad_malignancy_frame(
            archive,
            ground_truth,
            challenge,
        )
    else:
        if excluded_manifest_path is None:
            raise RuntimeError(
                "Same-source controls require an excluded base manifest"
            )
        excluded_manifest = Path(excluded_manifest_path).resolve()
        selected = eligible_broad_attention_frame(
            archive,
            ground_truth,
            challenge,
            benign_control_quotas={
                str(key): int(value)
                for key, value in benign_control_quotas.items()
            },
            excluded_image_names=set(
                read_manifest(excluded_manifest)["image_name"].astype(str)
            ),
            selection_seed=str(config["dataset"]["selection_seed"]),
        )
    expected = int(config["dataset"]["expected_images"])
    if len(selected) != expected:
        raise RuntimeError(
            f"Expected {expected} broad records, found {len(selected)}"
        )
    print(f"Downloading {len(selected):,} train-only broad images", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        image_records = list(
            executor.map(
                lambda row: _download_image(row, images_root),
                selected.itertuples(index=False),
            )
        )

    metadata = pd.DataFrame(
        {
            "image_name": selected["image_name"].astype(str),
            "group_id": (
                "isic2019-lesion::"
                + selected["lesion_id_challenge"].astype(str)
            ),
            "patient_id": "",
            "patient_identifier_status": "unavailable",
            "lesion_id": selected["lesion_id_challenge"].astype(str),
            "diagnosis_class": selected["diagnosis_class"].astype(str),
            "selection_role": selected["selection_role"].astype(str),
            "diagnosis_confirm_type": selected[
                "diagnosis_confirm_type"
            ].fillna("unavailable"),
            "source_url": selected["full_image_url"].astype(str),
            "license": "CC-BY-NC",
            "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
            "attribution": selected["attribution"].fillna("ISIC Archive"),
        }
    )
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    with metadata_output.open("x", encoding="utf-8") as output:
        metadata.to_csv(output, index=False, lineterminator="\n")
    with image_receipt.open("x", encoding="utf-8") as output:
        pd.DataFrame.from_records(image_records).to_csv(
            output,
            index=False,
            lineterminator="\n",
        )
    receipt = {
        "schema_version": 1,
        "dataset": config["dataset"],
        "config_sha256": sha256_file(config_file),
        "metadata_sha256": sha256_file(metadata_output),
        "image_receipt_sha256": sha256_file(image_receipt),
        "records": int(len(metadata)),
        "lesions": int(metadata["lesion_id"].nunique()),
        "diagnosis_counts": {
            str(label): int(count)
            for label, count in metadata["diagnosis_class"]
            .value_counts()
            .sort_index()
            .items()
        },
        "selection_role_counts": {
            str(label): int(count)
            for label, count in metadata["selection_role"]
            .value_counts()
            .sort_index()
            .items()
        },
        "excluded_manifest_sha256": (
            None
            if excluded_manifest_path is None
            else sha256_file(excluded_manifest_path)
        ),
        "patient_identifier_status": "unavailable_for_selected_records",
        "split_policy": "train_only",
        "downloaded_bytes": int(
            sum(int(record["bytes"]) for record in image_records)
        ),
        "archive_snapshot_size_mismatches": int(
            sum(
                not bool(record["archive_snapshot_size_matches"])
                for record in image_records
            )
        ),
        "training_authorized": True,
        "validation_authorized": False,
        "test_authorized": False,
        "research_only": True,
    }
    write_json_exclusive(receipt_output, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the ISIC 2019 BCC/AK/SCC subset for train-only broad "
            "malignancy research."
        )
    )
    parser.add_argument("--downloads-dir", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--exclude-manifest")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = download_broad_malignancy_subset(
        downloads_dir=args.downloads_dir,
        images_dir=args.images_dir,
        config_path=args.config,
        metadata_output_path=args.metadata_output,
        excluded_manifest_path=args.exclude_manifest,
        workers=args.workers,
    )
    print(result)


if __name__ == "__main__":
    main()
