from __future__ import annotations

import argparse
import json
import ssl
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd
import certifi

from .artifacts import require_absent, write_json_exclusive
from .common import sha256_file


API_ORIGIN = "https://api.isic-archive.com"
SEARCH_PATH = "/api/v2/images/search/"
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def extract_image_record(image: dict[str, Any]) -> dict[str, Any]:
    metadata = image.get("metadata") or {}
    clinical = metadata.get("clinical") or {}
    acquisition = metadata.get("acquisition") or {}
    full_file = (image.get("files") or {}).get("full") or {}
    return {
        "image_name": str(image["isic_id"]),
        "patient_id": str(clinical.get("patient_id") or ""),
        "lesion_id": str(clinical.get("lesion_id") or ""),
        "diagnosis_1": str(clinical.get("diagnosis_1") or ""),
        "diagnosis_2": str(clinical.get("diagnosis_2") or ""),
        "diagnosis_3": str(clinical.get("diagnosis_3") or ""),
        "diagnosis_confirm_type": str(
            clinical.get("diagnosis_confirm_type") or ""
        ),
        "image_type": str(acquisition.get("image_type") or ""),
        "copyright_license": str(image.get("copyright_license") or ""),
        "attribution": str(image.get("attribution") or ""),
        "full_image_url": str(full_file.get("url") or ""),
        "full_image_size": int(full_file.get("size") or 0),
    }


def _read_json(url: str, *, attempts: int = 5) -> dict[str, Any]:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "api.isic-archive.com":
        raise RuntimeError(f"Refusing unexpected ISIC API URL: {url}")
    for attempt in range(attempts):
        try:
            request = Request(
                url,
                headers={"User-Agent": "Express-Derm-Research/1.0"},
            )
            with urlopen(
                request,
                timeout=60,
                context=TLS_CONTEXT,
            ) as response:
                return json.load(response)
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def fetch_collection_metadata(
    *,
    collection_id: int,
    expected_count: int,
    output_path: str | Path,
) -> dict[str, Any]:
    output = Path(output_path)
    receipt_path = output.with_name(f"{output.stem}.receipt.json")
    require_absent([output, receipt_path])
    query = urlencode({"collections": collection_id, "limit": 100})
    next_url: str | None = f"{API_ORIGIN}{SEARCH_PATH}?{query}"
    records: list[dict[str, Any]] = []
    reported_count: int | None = None
    pages = 0
    while next_url:
        payload = _read_json(next_url)
        if reported_count is None:
            reported_count = int(payload["count"])
            if reported_count != expected_count:
                raise RuntimeError(
                    "Unexpected ISIC collection count: "
                    f"expected {expected_count}, found {reported_count}"
                )
        page_records = payload.get("results")
        if not isinstance(page_records, list) or not page_records:
            raise RuntimeError("ISIC API returned an empty metadata page")
        records.extend(extract_image_record(item) for item in page_records)
        pages += 1
        print(
            f"Fetched ISIC metadata {len(records):,}/{expected_count:,}",
            flush=True,
        )
        next_value = payload.get("next")
        next_url = str(next_value) if next_value else None

    frame = pd.DataFrame.from_records(records).sort_values(
        "image_name",
        kind="mergesort",
    )
    if len(frame) != expected_count or frame["image_name"].duplicated().any():
        raise RuntimeError("ISIC API metadata is incomplete or duplicated")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        frame.to_csv(destination, index=False, lineterminator="\n")
    receipt = {
        "schema_version": 1,
        "source": f"{API_ORIGIN}{SEARCH_PATH}",
        "collection_id": collection_id,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "records": int(len(frame)),
        "pages": pages,
        "patients": int(
            frame.loc[
                frame["patient_id"].ne(""),
                "patient_id",
            ].nunique()
        ),
        "patient_id_missing": int(frame["patient_id"].eq("").sum()),
        "lesions": int(
            frame.loc[
                frame["lesion_id"].ne(""),
                "lesion_id",
            ].nunique()
        ),
        "lesion_id_missing": int(frame["lesion_id"].eq("").sum()),
        "metadata_sha256": sha256_file(output),
    }
    write_json_exclusive(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Snapshot official ISIC collection metadata for audit.",
    )
    parser.add_argument("--collection-id", type=int, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = fetch_collection_metadata(
        collection_id=args.collection_id,
        expected_count=args.expected_count,
        output_path=args.output,
    )
    print(result)


if __name__ == "__main__":
    main()
