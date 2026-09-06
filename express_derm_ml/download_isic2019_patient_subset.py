from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd

from .artifacts import require_absent, write_json_exclusive
from .common import load_yaml, sha256_file
from .fetch_isic_collection_metadata import TLS_CONTEXT


class HTTPRangeReader(io.RawIOBase):
    def __init__(
        self,
        url: str,
        *,
        allowed_path_prefixes: tuple[str, ...] = ("/challenges/2019/",),
    ) -> None:
        parsed = urlparse(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "isic-archive.s3.amazonaws.com"
            or not any(
                parsed.path.startswith(prefix)
                for prefix in allowed_path_prefixes
            )
        ):
            raise RuntimeError(f"Unexpected ISIC archive URL: {url}")
        self.url = url
        request = Request(
            url,
            method="HEAD",
            headers={"User-Agent": "Express-Derm-Research/1.0"},
        )
        with urlopen(
            request,
            timeout=60,
            context=TLS_CONTEXT,
        ) as response:
            self.length = int(response.headers["Content-Length"])
            self.etag = str(response.headers["ETag"])
            self.last_modified = str(response.headers.get("Last-Modified", ""))
        if self.length <= 0 or not self.etag:
            raise RuntimeError("ISIC archive identity is incomplete")
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.length + offset
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        if position < 0:
            raise ValueError("Negative seek position")
        self.position = min(position, self.length)
        return self.position

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.length:
            return b""
        if size is None or size < 0:
            end = self.length
        else:
            end = min(self.position + size, self.length)
        start = self.position
        if end <= start:
            return b""
        for attempt in range(5):
            try:
                request = Request(
                    self.url,
                    headers={
                        "User-Agent": "Express-Derm-Research/1.0",
                        "Range": f"bytes={start}-{end - 1}",
                        "If-Match": self.etag,
                    },
                )
                with urlopen(
                    request,
                    timeout=90,
                    context=TLS_CONTEXT,
                ) as response:
                    payload = response.read()
                    if response.status != 206 or len(payload) != end - start:
                        raise RuntimeError("Invalid HTTP range response")
                self.position = end
                return payload
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")


def _validate_sources(
    downloads_root: Path,
    config: dict[str, Any],
) -> dict[str, Path]:
    paths = {
        name: downloads_root / str(source["filename"])
        for name, source in config["sources"].items()
    }
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"ISIC 2019 source is missing: {path}")
        if sha256_file(path) != str(config["sources"][name]["sha256"]):
            raise RuntimeError(f"ISIC 2019 source hash mismatch: {name}")
    return paths


def _eligible_frame(
    source_paths: dict[str, Path],
    config: dict[str, Any],
    reference_manifest_path: Path,
) -> pd.DataFrame:
    archive = pd.read_csv(
        source_paths["archive_metadata"],
        keep_default_na=False,
    )
    ground_truth = pd.read_csv(source_paths["ground_truth"])
    challenge = pd.read_csv(source_paths["challenge_metadata"])
    dataset = config["dataset"]
    if len(archive) != int(dataset["expected_api_records"]):
        raise RuntimeError("Unexpected ISIC 2019 API snapshot count")
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
        raise RuntimeError("ISIC 2019 labels must be one-hot binary values")
    if not ground_truth[class_columns].sum(axis=1).eq(1.0).all():
        raise RuntimeError("ISIC 2019 labels must contain exactly one class")
    ground_truth = ground_truth.copy()
    ground_truth["archive_image_name"] = ground_truth["image"].str.removesuffix(
        "_downsampled"
    )
    challenge = challenge.copy()
    challenge["archive_image_name"] = challenge["image"].str.removesuffix(
        "_downsampled"
    )
    labels = ground_truth.merge(
        challenge.loc[:, ["archive_image_name", "lesion_id"]],
        on="archive_image_name",
        how="inner",
        validate="one_to_one",
    )
    merged = archive.merge(
        labels,
        left_on="image_name",
        right_on="archive_image_name",
        how="inner",
        validate="one_to_one",
        suffixes=("_api", "_challenge"),
    )
    if len(merged) != int(dataset["expected_api_records"]):
        raise RuntimeError("ISIC 2019 API and challenge metadata do not align")
    selected = merged.loc[merged["patient_id"].ne("")].copy()
    selected["diagnosis_class"] = selected[class_columns].idxmax(axis=1)
    if len(selected) != int(dataset["expected_subset_records"]):
        raise RuntimeError("Unexpected ISIC 2019 patient-ID subset count")
    if selected["patient_id"].nunique() != int(dataset["expected_patients"]):
        raise RuntimeError("Unexpected ISIC 2019 patient count")
    if int(selected["MEL"].sum()) != int(dataset["expected_melanoma"]):
        raise RuntimeError("Unexpected ISIC 2019 melanoma count")
    if selected["lesion_id_api"].eq("").any():
        raise RuntimeError("Eligible ISIC 2019 images require lesion IDs")
    if set(selected["copyright_license"]) != {"CC-0"}:
        raise RuntimeError("Eligible ISIC 2019 subset must be CC-0")
    if set(selected["image_type"]) != {"dermoscopic"}:
        raise RuntimeError("Eligible ISIC 2019 subset must be dermoscopic")
    reference = pd.read_csv(reference_manifest_path, dtype={"patient_id": str})
    overlap = set(selected["patient_id"]) & set(reference["patient_id"])
    if overlap:
        raise RuntimeError(
            "ISIC 2019 patient IDs overlap the reference corpus"
        )
    for url in selected["full_image_url"]:
        parsed = urlparse(str(url))
        if (
            parsed.scheme != "https"
            or parsed.netloc != "isic-archive.s3.amazonaws.com"
            or not parsed.path.startswith("/images/")
        ):
            raise RuntimeError(f"Unexpected ISIC image URL: {url}")
    return selected.sort_values("image_name", kind="mergesort")


def _file_crc32(path: Path) -> int:
    crc = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            crc = zipfile.crc32(chunk, crc)
    return crc & 0xFFFFFFFF


def download_patient_subset(
    *,
    downloads_dir: str | Path,
    images_dir: str | Path,
    reference_manifest_path: str | Path,
    config_path: str | Path,
    metadata_output_path: str | Path,
) -> dict[str, Any]:
    downloads_root = Path(downloads_dir).resolve()
    images_root = Path(images_dir).resolve()
    images_root.mkdir(parents=True, exist_ok=True)
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
    source_paths = _validate_sources(downloads_root, config)
    selected = _eligible_frame(
        source_paths,
        config,
        Path(reference_manifest_path),
    )
    archive_reader = HTTPRangeReader(str(config["dataset"]["archive_url"]))
    image_records: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_reader) as archive:
        by_basename: dict[str, zipfile.ZipInfo] = {}
        for info in archive.infolist():
            basename = Path(info.filename).name
            if not basename:
                continue
            if basename in by_basename:
                raise RuntimeError(
                    f"Duplicate filename in ISIC archive: {basename}"
                )
            by_basename[basename] = info
        for position, row in enumerate(
            selected.itertuples(index=False),
            start=1,
        ):
            if position == 1 or position % 25 == 0:
                print(
                    f"Downloading patient-safe ISIC 2019 image {position}/"
                    f"{len(selected)}",
                    flush=True,
                )
            archive_basename = f"{row.image}.jpg"
            info = by_basename.get(archive_basename)
            if info is None:
                raise RuntimeError(
                    f"ISIC challenge image is absent from archive: "
                    f"{archive_basename}"
                )
            image_path = images_root / f"{row.image_name}.jpg"
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
                    "image_name": str(row.image_name),
                    "challenge_image_name": str(row.image),
                    "archive_member": info.filename,
                    "bytes": int(image_path.stat().st_size),
                    "zip_crc32": f"{info.CRC:08x}",
                    "sha256": sha256_file(image_path),
                }
            )
    curated_metadata = pd.DataFrame(
        {
            "image_name": selected["image_name"],
            "challenge_image_name": selected["image"],
            "patient_id": selected["patient_id"],
            "lesion_id": selected["lesion_id_api"],
            "diagnosis_class": selected["diagnosis_class"].str.lower(),
            "diagnosis_confirm_type": selected["diagnosis_confirm_type"],
        }
    )
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    with metadata_output.open("x", encoding="utf-8") as destination:
        curated_metadata.to_csv(destination, index=False, lineterminator="\n")
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
        "records": int(len(curated_metadata)),
        "patients": int(curated_metadata["patient_id"].nunique()),
        "melanoma_positive": int(
            curated_metadata["diagnosis_class"].eq("mel").sum()
        ),
        "downloaded_bytes": int(
            sum(record["bytes"] for record in image_records)
        ),
        "archive_url": archive_reader.url,
        "archive_content_length": archive_reader.length,
        "archive_etag": archive_reader.etag,
        "archive_last_modified": archive_reader.last_modified,
        "metadata_sha256": sha256_file(metadata_output),
        "image_receipt_sha256": sha256_file(image_receipt_path),
        "reference_manifest_sha256": Path(
            f"{reference_manifest_path}.sha256"
        ).read_text(encoding="ascii").strip(),
        "training_authorized": False,
        "status": "downloaded_pending_cross_corpus_curation",
    }
    write_json_exclusive(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads-dir", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--metadata-output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = download_patient_subset(
        downloads_dir=args.downloads_dir,
        images_dir=args.images_dir,
        reference_manifest_path=args.reference_manifest,
        config_path=args.config,
        metadata_output_path=args.metadata_output,
    )
    print(result)


if __name__ == "__main__":
    main()
