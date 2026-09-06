from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml


class ManifestValidationError(RuntimeError):
    pass


MANIFEST_TEXT_COLUMNS = {
    "image_name",
    "image_path",
    "curated_label",
    "source_label",
    "group_id",
    "patient_id",
    "patient_identifier_status",
    "selection_role",
    "lesion_id",
    "diagnosis",
    "collection_id",
    "dataset_version",
    "source_url",
    "license",
    "license_url",
    "attribution",
    "sha256",
    "perceptual_hash",
    "perceptual_hash_algorithm",
    "perceptual_hash_implementation",
    "perceptual_hash_implementation_version",
    "split",
}


def read_manifest(
    path: str | Path,
    *,
    additional_text_columns: Iterable[str] = (),
) -> pd.DataFrame:
    target = Path(path)
    columns = set(pd.read_csv(target, nrows=0).columns)
    requested = MANIFEST_TEXT_COLUMNS | set(additional_text_columns)
    dtypes = {
        column: "string"
        for column in requested
        if column in columns
    }
    return pd.read_csv(target, dtype=dtypes)


def load_curation_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ManifestValidationError("Curation config must be a mapping")

    required_sections = {
        "schema_version",
        "dataset",
        "columns",
        "label_mapping",
        "integrity",
    }
    missing_sections = required_sections - set(config)
    if missing_sections:
        raise ManifestValidationError(
            f"Missing config sections: {sorted(missing_sections)}"
        )

    dataset = config["dataset"]
    if not isinstance(dataset, dict):
        raise ManifestValidationError("Dataset config must be a mapping")
    required_dataset_fields = {
        "id",
        "version",
        "source_url",
        "license",
        "license_url",
        "attribution",
    }
    missing_dataset_fields = required_dataset_fields - set(dataset)
    if missing_dataset_fields:
        raise ManifestValidationError(
            "Missing dataset fields: "
            f"{sorted(missing_dataset_fields)}"
        )
    blank_dataset_fields = sorted(
        field
        for field in required_dataset_fields
        if not str(dataset[field]).strip()
    )
    if blank_dataset_fields:
        raise ManifestValidationError(
            f"Blank dataset fields: {blank_dataset_fields}"
        )

    columns = config["columns"]
    if not isinstance(columns, dict):
        raise ManifestValidationError("Column config must be a mapping")
    missing_columns = {"image", "group"} - set(columns)
    if missing_columns:
        raise ManifestValidationError(
            f"Missing column mappings: {sorted(missing_columns)}"
        )
    if not str(columns["image"]).strip() or not str(columns["group"]).strip():
        raise ManifestValidationError(
            "Image and group column mappings cannot be blank"
        )

    label_mapping = config["label_mapping"]
    if not isinstance(label_mapping, dict):
        raise ManifestValidationError("Label mapping must be a mapping")
    missing_label_fields = {
        "version",
        "task",
        "source_column",
        "values",
    } - set(label_mapping)
    if missing_label_fields:
        raise ManifestValidationError(
            "Missing label-mapping fields: "
            f"{sorted(missing_label_fields)}"
        )
    if not isinstance(label_mapping["values"], dict):
        raise ManifestValidationError("Label mapping values must be a mapping")
    if not label_mapping["values"]:
        raise ManifestValidationError("Label mapping values cannot be empty")
    mapped_targets: set[int] = set()
    for source_value, mapped in label_mapping["values"].items():
        if not isinstance(mapped, dict) or mapped.get("target") not in (0, 1):
            raise ManifestValidationError(
                f"Label {source_value!r} must map to target 0 or 1"
            )
        if not str(mapped.get("label", "")).strip():
            raise ManifestValidationError(
                f"Label {source_value!r} is missing a curated label"
            )
        mapped_targets.add(int(mapped["target"]))
    if mapped_targets != {0, 1}:
        raise ManifestValidationError(
            "Label mapping must define both target 0 and target 1"
        )

    integrity = config["integrity"]
    if not isinstance(integrity, dict):
        raise ManifestValidationError("Integrity config must be a mapping")
    if "perceptual_hash" not in integrity:
        raise ManifestValidationError(
            "Integrity config is missing perceptual_hash"
        )
    perceptual_hash = integrity["perceptual_hash"]
    if not isinstance(perceptual_hash, dict):
        raise ManifestValidationError(
            "Perceptual-hash config must be a mapping"
        )
    required_hash_fields = {
        "algorithm",
        "implementation",
        "implementation_version",
        "hash_size",
        "highfreq_factor",
    }
    missing_hash_fields = required_hash_fields - set(perceptual_hash)
    if missing_hash_fields:
        raise ManifestValidationError(
            "Missing perceptual-hash fields: "
            f"{sorted(missing_hash_fields)}"
        )
    if perceptual_hash["algorithm"] != "phash":
        raise ManifestValidationError(
            "Only the versioned phash algorithm is supported"
        )
    if perceptual_hash["implementation"] != "ImageHash":
        raise ManifestValidationError(
            "Only the pinned ImageHash implementation is supported"
        )
    if not str(perceptual_hash["implementation_version"]).strip():
        raise ManifestValidationError(
            "Perceptual-hash implementation version cannot be blank"
        )
    try:
        hash_size = int(perceptual_hash["hash_size"])
        highfreq_factor = int(perceptual_hash["highfreq_factor"])
        threshold = int(integrity["near_duplicate_hamming_threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise ManifestValidationError(
            "Invalid perceptual-hash dimensions or near-duplicate threshold"
        ) from error
    if not 4 <= hash_size <= 16 or hash_size % 2:
        raise ManifestValidationError(
            "Perceptual hash_size must be even and between 4 and 16"
        )
    if not 1 <= highfreq_factor <= 8:
        raise ManifestValidationError(
            "Perceptual highfreq_factor must be between 1 and 8"
        )
    if not 0 <= threshold < hash_size * hash_size:
        raise ManifestValidationError(
            "Near-duplicate threshold must fit the perceptual hash bit width"
        )
    return config


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if value == "":
        return None
    if isinstance(value, float) and math.isfinite(value):
        return round(value, 12)
    return value


def canonical_manifest_sha256(frame: pd.DataFrame) -> str:
    columns = sorted(str(column) for column in frame.columns)
    normalized = frame.loc[:, columns].copy()
    sort_columns = [
        column
        for column in ("image_name", "image_path", "sha256")
        if column in normalized.columns
    ]
    if sort_columns:
        normalized = normalized.sort_values(
            sort_columns,
            kind="mergesort",
        )
    records = [
        {column: _json_value(row[column]) for column in columns}
        for _, row in normalized.iterrows()
    ]
    payload = json.dumps(
        {"columns": columns, "records": records},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def artifact_path(output: str | Path, suffix: str) -> Path:
    target = Path(output)
    return target.with_name(f"{target.stem}.{suffix}")


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_manifest(
    frame: pd.DataFrame,
    output: str | Path,
) -> str:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, lineterminator="\n")
    digest = canonical_manifest_sha256(frame)
    Path(f"{target}.sha256").write_text(f"{digest}\n", encoding="ascii")
    return digest
