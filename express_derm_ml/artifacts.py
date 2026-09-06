from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


class ArtifactExistsError(RuntimeError):
    pass


def require_absent(paths: list[str | Path]) -> None:
    existing = [str(Path(path)) for path in paths if Path(path).exists()]
    if existing:
        raise ArtifactExistsError(
            "Artifacts already exist and will not be overwritten: "
            f"{existing}"
        )


def create_new_directory(path: str | Path) -> Path:
    target = Path(path)
    try:
        target.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ArtifactExistsError(
            f"Artifact directory already exists and will not be reused: {target}"
        ) from error
    return target


def write_json_exclusive(
    path: str | Path,
    payload: dict[str, Any],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as destination:
            json.dump(payload, destination, indent=2, sort_keys=True)
            destination.write("\n")
    except FileExistsError as error:
        raise ArtifactExistsError(
            f"Artifact already exists and will not be overwritten: {target}"
        ) from error
    return target


def copy_file_exclusive(
    source: str | Path,
    destination: str | Path,
) -> Path:
    source_path = Path(source)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source_path.open("rb") as input_file, target.open("xb") as output:
            shutil.copyfileobj(input_file, output)
    except FileExistsError as error:
        raise ArtifactExistsError(
            f"Artifact already exists and will not be overwritten: {target}"
        ) from error
    return target


def write_npz_exclusive(
    path: str | Path,
    **arrays: Any,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as output:
            np.savez_compressed(output, **arrays)
    except FileExistsError as error:
        raise ArtifactExistsError(
            f"Artifact already exists and will not be overwritten: {target}"
        ) from error
    return target
