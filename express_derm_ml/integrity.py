from __future__ import annotations

import re
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import imagehash
import pandas as pd
from PIL import Image

from .manifest import ManifestValidationError

NEAR_DUPLICATE_COLUMNS = [
    "left_image_name",
    "right_image_name",
    "left_group_id",
    "right_group_id",
    "left_sha256",
    "right_sha256",
    "left_perceptual_hash",
    "right_perceptual_hash",
    "hamming_distance",
    "same_group",
]


def validate_perceptual_hash_runtime(config: dict[str, Any]) -> None:
    expected = str(config["implementation_version"])
    try:
        actual = version(str(config["implementation"]))
    except PackageNotFoundError as error:
        raise ManifestValidationError(
            "Configured perceptual-hash implementation is not installed"
        ) from error
    if actual != expected:
        raise ManifestValidationError(
            "Perceptual-hash runtime version mismatch: "
            f"expected {expected}, found {actual}"
        )


def perceptual_hash_file(
    path: str | Path,
    *,
    hash_size: int,
    highfreq_factor: int,
) -> str:
    with Image.open(path) as image:
        value = imagehash.phash(
            image.convert("RGB"),
            hash_size=hash_size,
            highfreq_factor=highfreq_factor,
        )
    return str(value)


def _segment_specs(bit_width: int, threshold: int) -> list[tuple[int, int]]:
    segment_count = threshold + 1
    minimum_width, wider_segments = divmod(bit_width, segment_count)
    specs: list[tuple[int, int]] = []
    shift = 0
    for segment in range(segment_count):
        width = minimum_width + (1 if segment < wider_segments else 0)
        specs.append((shift, (1 << width) - 1))
        shift += width
    return specs


def _validated_hash_values(
    frame: pd.DataFrame,
    *,
    hash_column: str,
) -> tuple[list[int], int]:
    raw_hashes = frame[hash_column].astype(str).str.strip()
    lengths = raw_hashes.str.len().unique()
    if len(lengths) != 1:
        raise ManifestValidationError(
            "Perceptual hashes must have one consistent bit width"
        )
    hex_length = int(lengths[0])
    if hex_length == 0:
        raise ManifestValidationError("Perceptual hashes cannot be blank")
    pattern = re.compile(rf"[0-9a-f]{{{hex_length}}}")
    if not raw_hashes.map(lambda value: bool(pattern.fullmatch(value))).all():
        raise ManifestValidationError(
            "Perceptual hashes must be lowercase hexadecimal values"
        )
    return [int(value, 16) for value in raw_hashes], hex_length * 4


def find_near_duplicate_pairs(
    frame: pd.DataFrame,
    *,
    threshold: int,
    hash_column: str = "perceptual_hash",
    group_column: str = "group_id",
) -> pd.DataFrame:
    required = {
        "image_name",
        "sha256",
        hash_column,
        group_column,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ManifestValidationError(
            f"Missing near-duplicate audit columns: {sorted(missing)}"
        )
    if frame.empty:
        return pd.DataFrame(columns=NEAR_DUPLICATE_COLUMNS)

    values, bit_width = _validated_hash_values(
        frame,
        hash_column=hash_column,
    )
    if not 0 <= threshold < bit_width:
        raise ManifestValidationError(
            "Near-duplicate threshold must fit the perceptual hash bit width"
        )

    specs = _segment_specs(bit_width, threshold)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    matched_pairs: set[tuple[int, int]] = set()
    for right_index, right_hash in enumerate(values):
        for segment, (shift, mask) in enumerate(specs):
            bucket_key = (segment, (right_hash >> shift) & mask)
            for left_index in buckets[bucket_key]:
                pair = (left_index, right_index)
                if pair in matched_pairs:
                    continue
                distance = (values[left_index] ^ right_hash).bit_count()
                if distance <= threshold:
                    matched_pairs.add(pair)
            buckets[bucket_key].append(right_index)

    records: list[dict[str, Any]] = []
    for left_index, right_index in sorted(matched_pairs):
        left = frame.iloc[left_index]
        right = frame.iloc[right_index]
        left_group = str(left[group_column])
        right_group = str(right[group_column])
        records.append(
            {
                "left_image_name": str(left["image_name"]),
                "right_image_name": str(right["image_name"]),
                "left_group_id": left_group,
                "right_group_id": right_group,
                "left_sha256": str(left["sha256"]),
                "right_sha256": str(right["sha256"]),
                "left_perceptual_hash": str(left[hash_column]),
                "right_perceptual_hash": str(right[hash_column]),
                "hamming_distance": (
                    values[left_index] ^ values[right_index]
                ).bit_count(),
                "same_group": left_group == right_group,
            }
        )
    if not records:
        return pd.DataFrame(columns=NEAR_DUPLICATE_COLUMNS)
    return (
        pd.DataFrame.from_records(records, columns=NEAR_DUPLICATE_COLUMNS)
        .sort_values(
            ["hamming_distance", "left_image_name", "right_image_name"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
