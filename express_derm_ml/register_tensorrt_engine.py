from __future__ import annotations

import argparse
import json
import platform
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import (
    copy_file_exclusive,
    create_new_directory,
    write_json_exclusive,
)
from .manifest import canonical_manifest_sha256, read_manifest, sha256_file


REGISTRATION_SCHEMA_VERSION = 1
SUPPORTED_PRECISIONS = {"fp16", "fp32", "int8"}
SUPPORTED_LINUX_ARCHITECTURES = {"aarch64", "arm64", "x86_64", "amd64"}


class EngineRegistrationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a non-overwriting model package for a target-built "
            "TensorRT engine."
        )
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--platform-version", required=True)
    parser.add_argument("--tensorrt-version", required=True)
    parser.add_argument("--device-model", required=True)
    parser.add_argument(
        "--precision",
        choices=sorted(SUPPORTED_PRECISIONS),
        default="fp16",
    )
    return parser.parse_args()


def register_tensorrt_engine(
    *,
    model_dir: str | Path,
    engine_path: str | Path,
    output_dir: str | Path,
    platform_version: str,
    tensorrt_version: str,
    device_model: str,
    precision: str,
    host_system: str,
    host_machine: str,
    registered_at: str | None = None,
) -> dict[str, Any]:
    source_dir = Path(model_dir)
    engine = Path(engine_path)
    output = Path(output_dir)
    target = _validated_target(
        platform_version=platform_version,
        tensorrt_version=tensorrt_version,
        device_model=device_model,
        precision=precision,
        host_system=host_system,
        host_machine=host_machine,
    )
    source_files = _validated_source_files(source_dir)
    if not engine.is_file() or engine.is_symlink():
        raise EngineRegistrationError(
            f"TensorRT engine is not a regular file: {engine}"
        )
    if engine.stat().st_size == 0:
        raise EngineRegistrationError("TensorRT engine cannot be empty")
    if output.exists():
        raise EngineRegistrationError(
            f"Output directory already exists and will not be reused: {output}"
        )
    source_root = source_dir.resolve()
    engine_resolved = engine.resolve()
    output_resolved = output.resolve()
    if engine_resolved.parent == source_root:
        raise EngineRegistrationError(
            "Build the TensorRT engine outside the immutable source model "
            "directory"
        )
    try:
        output_resolved.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise EngineRegistrationError(
            "Registered output directory cannot be inside the source model "
            "directory"
        )

    manifest_path = source_dir / "manifest.json"
    model_path = source_dir / "model.onnx"
    manifest = _load_source_manifest(manifest_path)
    _verify_manifest_links(source_dir, manifest)
    source_file_sha256 = {
        source.name: sha256_file(source) for source in source_files
    }
    model_sha256 = sha256_file(model_path)
    if manifest["model_sha256"] != model_sha256:
        raise EngineRegistrationError(
            "ONNX model hash does not match the source manifest"
        )
    engine_sha256 = sha256_file(engine)
    timestamp = registered_at or datetime.now(timezone.utc).isoformat()

    registered_manifest = deepcopy(manifest)
    registered_manifest.setdefault("release_version", manifest["version"])
    registered_manifest.update(
        {
            "engine_filename": "model.engine",
            "engine_sha256": engine_sha256,
            "deployment_status": "engine_registered",
            "engine_registration": {
                "schema_version": REGISTRATION_SCHEMA_VERSION,
                "registered_at": timestamp,
                "precision": target["precision"],
                "platform_version": target["platform_version"],
                "tensorrt_version": target["tensorrt_version"],
                "device_model": target["device_model"],
                "host_system": target["host_system"],
                "host_machine": target["host_machine"],
                "parity_status": "pending",
                "target_runtime_status": "pending",
            },
        }
    )
    registered_manifest_bytes = _render_json(registered_manifest)

    registration = {
        "schema_version": REGISTRATION_SCHEMA_VERSION,
        "registered_at": timestamp,
        "model_version": manifest["version"],
        "model_release": registered_manifest["release_version"],
        "model_sha256": model_sha256,
        "engine_sha256": engine_sha256,
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_file_sha256": source_file_sha256,
        "registered_manifest_sha256": _sha256_bytes(
            registered_manifest_bytes
        ),
        "deployment_status": "engine_registered",
        "parity_status": "pending",
        "target_runtime_status": "pending",
        "inference_enabled": False,
        "target": target,
        "preserved_validation": {
            "validation_status": manifest["validation_status"],
            "domain_status": manifest["domain_status"],
            "thresholds_validated": manifest["thresholds_validated"],
            "research_only": manifest.get("research_only"),
        },
    }

    create_new_directory(output)
    for source in source_files:
        if source.name == "manifest.json":
            continue
        copy_file_exclusive(source, output / source.name)
    copy_file_exclusive(engine, output / "model.engine")
    write_json_exclusive(output / "manifest.json", registered_manifest)
    write_json_exclusive(output / "engine-registration.json", registration)

    if sha256_file(output / "model.onnx") != model_sha256:
        raise EngineRegistrationError("Copied ONNX model failed verification")
    if sha256_file(output / "model.engine") != engine_sha256:
        raise EngineRegistrationError("Copied TensorRT engine failed verification")
    for source in source_files:
        if source.name == "manifest.json":
            continue
        if sha256_file(output / source.name) != source_file_sha256[source.name]:
            raise EngineRegistrationError(
                f"Copied source artifact failed verification: {source.name}"
            )
    if (
        sha256_file(output / "manifest.json")
        != registration["registered_manifest_sha256"]
    ):
        raise EngineRegistrationError(
            "Registered manifest failed verification"
        )
    return registration


def _validated_source_files(source_dir: Path) -> list[Path]:
    if not source_dir.is_dir() or source_dir.is_symlink():
        raise EngineRegistrationError(
            f"Model directory is not a regular directory: {source_dir}"
        )
    manifest_path = source_dir / "manifest.json"
    model_path = source_dir / "model.onnx"
    missing = [
        str(path)
        for path in (manifest_path, model_path)
        if not path.is_file() or path.is_symlink()
    ]
    if missing:
        raise EngineRegistrationError(
            f"Required source model files are missing: {missing}"
        )
    if (source_dir / "model.engine").exists():
        raise EngineRegistrationError(
            "Source model directory already contains model.engine"
        )
    if (source_dir / "engine-registration.json").exists():
        raise EngineRegistrationError(
            "Source model directory is already an engine registration"
        )

    files: list[Path] = []
    for entry in sorted(source_dir.iterdir(), key=lambda path: path.name):
        if entry.is_symlink() or not entry.is_file():
            raise EngineRegistrationError(
                f"Source model bundle must contain regular files only: {entry}"
            )
        files.append(entry)
    return files


def _load_source_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EngineRegistrationError(
            f"Unable to read source manifest: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise EngineRegistrationError("Source manifest must be a JSON object")
    required = {
        "version": str,
        "model_sha256": str,
        "validation_status": str,
        "domain_status": str,
        "thresholds_validated": bool,
    }
    for field, expected_type in required.items():
        value = manifest.get(field)
        if type(value) is not expected_type:
            raise EngineRegistrationError(
                f"Source manifest field {field} has an invalid type"
            )
        if expected_type is str and not value.strip():
            raise EngineRegistrationError(
                f"Source manifest field {field} cannot be blank"
            )
    release_version = manifest.get("release_version", manifest["version"])
    if not isinstance(release_version, str) or not release_version.strip():
        raise EngineRegistrationError(
            "Source manifest field release_version has an invalid type"
        )
    _require_sha256(manifest["model_sha256"], "model_sha256")
    if "engine_sha256" in manifest or "deployment_status" in manifest:
        raise EngineRegistrationError(
            "Source manifest already contains engine deployment metadata"
        )
    return manifest


def _validated_target(
    *,
    platform_version: str,
    tensorrt_version: str,
    device_model: str,
    precision: str,
    host_system: str,
    host_machine: str,
) -> dict[str, str]:
    normalized = {
        "platform_version": platform_version.strip(),
        "tensorrt_version": tensorrt_version.strip(),
        "device_model": device_model.strip(),
        "precision": precision.strip().lower(),
        "host_system": host_system.strip(),
        "host_machine": host_machine.strip().lower(),
    }
    blank = [key for key, value in normalized.items() if not value]
    if blank:
        raise EngineRegistrationError(
            f"Target environment fields cannot be blank: {blank}"
        )
    if normalized["precision"] not in SUPPORTED_PRECISIONS:
        raise EngineRegistrationError(
            f"Unsupported TensorRT precision: {normalized['precision']}"
        )
    if normalized["host_system"] != "Linux":
        raise EngineRegistrationError(
            "TensorRT engine registration must run on the target Linux host"
        )
    if normalized["host_machine"] not in SUPPORTED_LINUX_ARCHITECTURES:
        raise EngineRegistrationError(
            "TensorRT engine registration requires a supported Linux architecture"
        )
    return normalized


def _verify_manifest_links(
    source_dir: Path,
    manifest: dict[str, Any],
) -> None:
    calibration_sha256 = manifest.get("calibration_sha256")
    if calibration_sha256 is not None:
        if not isinstance(calibration_sha256, str):
            raise EngineRegistrationError(
                "Source manifest calibration_sha256 has an invalid type"
            )
        _require_sha256(calibration_sha256, "calibration_sha256")
        calibration_path = source_dir / "calibration.json"
        if not calibration_path.is_file() or calibration_path.is_symlink():
            raise EngineRegistrationError(
                "Source calibration artifact is missing"
            )
        if sha256_file(calibration_path) != calibration_sha256:
            raise EngineRegistrationError(
                "Calibration hash does not match the source manifest"
            )

    dataset_sha256 = manifest.get("dataset_manifest_sha256")
    if dataset_sha256 is not None:
        if not isinstance(dataset_sha256, str):
            raise EngineRegistrationError(
                "Source manifest dataset_manifest_sha256 has an invalid type"
            )
        _require_sha256(dataset_sha256, "dataset_manifest_sha256")
        dataset_manifest = source_dir / "manifest.csv"
        dataset_digest = source_dir / "manifest.csv.sha256"
        if (
            not dataset_manifest.is_file()
            or dataset_manifest.is_symlink()
            or not dataset_digest.is_file()
            or dataset_digest.is_symlink()
        ):
            raise EngineRegistrationError(
                "Source dataset manifest bundle is incomplete"
            )
        recorded_digest = dataset_digest.read_text(
            encoding="ascii"
        ).strip()
        if recorded_digest != dataset_sha256:
            raise EngineRegistrationError(
                "Dataset digest does not match the source manifest"
            )
        try:
            actual_digest = canonical_manifest_sha256(
                read_manifest(dataset_manifest)
            )
        except Exception as exc:
            raise EngineRegistrationError(
                f"Unable to verify the source dataset manifest: {exc}"
            ) from exc
        if actual_digest != dataset_sha256:
            raise EngineRegistrationError(
                "Canonical dataset manifest hash does not match"
            )


def _require_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise EngineRegistrationError(
            f"Source manifest field {field} must be a lowercase SHA-256"
        )


def _render_json(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    registration = register_tensorrt_engine(
        model_dir=args.model_dir,
        engine_path=args.engine,
        output_dir=args.output_dir,
        platform_version=args.platform_version,
        tensorrt_version=args.tensorrt_version,
        device_model=args.device_model,
        precision=args.precision,
        host_system=platform.system(),
        host_machine=platform.machine(),
    )
    print(
        "Registered TensorRT engine "
        f"{registration['engine_sha256']} in {args.output_dir}"
    )
    print(
        "Deployment remains disabled until parity and target-runtime "
        "validation are complete."
    )


if __name__ == "__main__":
    main()
