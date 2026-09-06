from __future__ import annotations

import json
from pathlib import Path

import pytest

from express_derm_ml.common import sha256_file
from express_derm_ml.register_tensorrt_engine import (
    EngineRegistrationError,
    register_tensorrt_engine,
)


REGISTERED_AT = "2026-07-26T18:00:00+00:00"


def create_source_bundle(tmp_path: Path) -> Path:
    source = tmp_path / "source-model"
    source.mkdir()
    model_path = source / "model.onnx"
    model_path.write_bytes(b"deterministic synthetic ONNX fixture")
    manifest = {
        "schema_version": 1,
        "version": "research-model-v1",
        "release_version": "research-model-v1",
        "model_sha256": sha256_file(model_path),
        "validation_status": "research_only",
        "domain_status": "microscope_validation_pending",
        "thresholds_validated": False,
        "research_only": True,
        "low_threshold": 0.2,
        "high_threshold": 0.8,
    }
    (source / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (source / "calibration.json").write_text(
        '{"selection_split":"validation"}\n',
        encoding="utf-8",
    )
    return source


def register(
    *,
    source: Path,
    engine: Path,
    output: Path,
    host_system: str = "Linux",
    host_machine: str = "x86_64",
) -> dict:
    return register_tensorrt_engine(
        model_dir=source,
        engine_path=engine,
        output_dir=output,
        platform_version="Ubuntu 24.04",
        tensorrt_version="10.3.0",
        device_model="NVIDIA GPU Host",
        precision="fp16",
        host_system=host_system,
        host_machine=host_machine,
        registered_at=REGISTERED_AT,
    )


def test_registration_creates_a_new_auditable_pending_package(
    tmp_path: Path,
) -> None:
    source = create_source_bundle(tmp_path)
    source_manifest_before = (source / "manifest.json").read_bytes()
    engine = tmp_path / "built.engine"
    engine.write_bytes(b"deterministic synthetic TensorRT engine fixture")
    output = tmp_path / "registered-model"

    registration = register(source=source, engine=engine, output=output)

    registered_manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert registered_manifest["version"] == "research-model-v1"
    assert registered_manifest["release_version"] == "research-model-v1"
    assert registration["model_release"] == "research-model-v1"
    assert registered_manifest["model_sha256"] == sha256_file(
        output / "model.onnx"
    )
    assert registered_manifest["engine_sha256"] == sha256_file(
        output / "model.engine"
    )
    assert registered_manifest["deployment_status"] == "engine_registered"
    assert registered_manifest["validation_status"] == "research_only"
    assert (
        registered_manifest["domain_status"]
        == "microscope_validation_pending"
    )
    assert registered_manifest["thresholds_validated"] is False
    assert registered_manifest["research_only"] is True
    assert (
        registered_manifest["engine_registration"]["parity_status"]
        == "pending"
    )
    assert registration["inference_enabled"] is False
    assert registration["target_runtime_status"] == "pending"
    assert registration["target"] == {
        "platform_version": "Ubuntu 24.04",
        "tensorrt_version": "10.3.0",
        "device_model": "NVIDIA GPU Host",
        "precision": "fp16",
        "host_system": "Linux",
        "host_machine": "x86_64",
    }
    assert registration["registered_manifest_sha256"] == sha256_file(
        output / "manifest.json"
    )
    assert registration["source_file_sha256"]["model.onnx"] == sha256_file(
        source / "model.onnx"
    )
    assert registration["source_file_sha256"][
        "calibration.json"
    ] == sha256_file(source / "calibration.json")
    assert (output / "calibration.json").is_file()
    assert (source / "manifest.json").read_bytes() == source_manifest_before
    assert not (source / "model.engine").exists()


def test_registration_rejects_a_tampered_onnx_before_creating_output(
    tmp_path: Path,
) -> None:
    source = create_source_bundle(tmp_path)
    (source / "model.onnx").write_bytes(b"tampered")
    engine = tmp_path / "built.engine"
    engine.write_bytes(b"engine")
    output = tmp_path / "registered-model"

    with pytest.raises(EngineRegistrationError, match="ONNX model hash"):
        register(source=source, engine=engine, output=output)

    assert not output.exists()


def test_registration_rejects_tampered_linked_calibration(
    tmp_path: Path,
) -> None:
    source = create_source_bundle(tmp_path)
    calibration_path = source / "calibration.json"
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["calibration_sha256"] = sha256_file(calibration_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    calibration_path.write_text('{"tampered":true}\n', encoding="utf-8")
    engine = tmp_path / "built.engine"
    engine.write_bytes(b"engine")
    output = tmp_path / "registered-model"

    with pytest.raises(EngineRegistrationError, match="Calibration hash"):
        register(source=source, engine=engine, output=output)

    assert not output.exists()


def test_registration_never_reuses_an_output_directory(
    tmp_path: Path,
) -> None:
    source = create_source_bundle(tmp_path)
    engine = tmp_path / "built.engine"
    engine.write_bytes(b"engine")
    output = tmp_path / "registered-model"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("original", encoding="utf-8")

    with pytest.raises(EngineRegistrationError, match="will not be reused"):
        register(source=source, engine=engine, output=output)

    assert marker.read_text(encoding="utf-8") == "original"
    assert list(output.iterdir()) == [marker]


def test_registration_rejects_already_registered_source(
    tmp_path: Path,
) -> None:
    source = create_source_bundle(tmp_path)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["engine_sha256"] = "a" * 64
    manifest["deployment_status"] = "engine_registered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    engine = tmp_path / "built.engine"
    engine.write_bytes(b"engine")
    output = tmp_path / "registered-model"

    with pytest.raises(
        EngineRegistrationError,
        match="already contains engine deployment metadata",
    ):
        register(source=source, engine=engine, output=output)

    assert not output.exists()


def test_registration_rejects_a_non_target_host(tmp_path: Path) -> None:
    source = create_source_bundle(tmp_path)
    engine = tmp_path / "built.engine"
    engine.write_bytes(b"engine")
    output = tmp_path / "registered-model"

    with pytest.raises(
        EngineRegistrationError,
        match="target Linux host",
    ):
        register(
            source=source,
            engine=engine,
            output=output,
            host_system="Darwin",
            host_machine="arm64",
        )

    assert not output.exists()


def test_registration_keeps_engine_build_outside_source_bundle(
    tmp_path: Path,
) -> None:
    source = create_source_bundle(tmp_path)
    engine = source / "built.engine"
    engine.write_bytes(b"engine")
    output = tmp_path / "registered-model"

    with pytest.raises(
        EngineRegistrationError,
        match="outside the immutable source",
    ):
        register(source=source, engine=engine, output=output)

    assert not output.exists()
