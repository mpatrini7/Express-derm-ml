from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_tensorrt_engine.sh"
)


def fake_trtexec(bin_dir: Path, *, exit_code: int) -> None:
    executable = bin_dir / "trtexec"
    executable.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
engine_path=""
for argument in "$@"; do
  case "${argument}" in
    --saveEngine=*) engine_path="${argument#--saveEngine=}" ;;
  esac
done
printf 'trtexec arguments: %s\n' "$*"
printf 'synthetic TensorRT engine fixture' > "${engine_path}"
exit """
        + str(exit_code)
        + "\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)


def run_build(
    *,
    onnx_path: Path,
    engine_path: Path,
    fake_bin: Path,
    dynamic_image_size: str | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    command = ["bash", str(SCRIPT), str(onnx_path), str(engine_path)]
    if dynamic_image_size is not None:
        command.append(dynamic_image_size)
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_build_publishes_only_a_complete_engine(tmp_path: Path) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"synthetic ONNX fixture")
    engine_path = tmp_path / "model.engine"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_trtexec(fake_bin, exit_code=0)

    result = run_build(
        onnx_path=onnx_path,
        engine_path=engine_path,
        fake_bin=fake_bin,
    )

    assert result.returncode == 0
    assert engine_path.read_bytes() == b"synthetic TensorRT engine fixture"
    assert "Built TensorRT engine" in result.stdout
    assert "--minShapes" not in result.stdout
    assert not list(tmp_path.glob(".model.engine.building.*"))


def test_build_adds_shapes_only_for_an_explicit_dynamic_model(
    tmp_path: Path,
) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"synthetic ONNX fixture")
    engine_path = tmp_path / "model.engine"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_trtexec(fake_bin, exit_code=0)

    result = run_build(
        onnx_path=onnx_path,
        engine_path=engine_path,
        fake_bin=fake_bin,
        dynamic_image_size="384",
    )

    assert result.returncode == 0
    assert "--minShapes=image:1x3x384x384" in result.stdout
    assert "--optShapes=image:1x3x384x384" in result.stdout
    assert "--maxShapes=image:1x3x384x384" in result.stdout


def test_build_rejects_invalid_dynamic_image_size(tmp_path: Path) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"synthetic ONNX fixture")
    engine_path = tmp_path / "model.engine"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_trtexec(fake_bin, exit_code=0)

    result = run_build(
        onnx_path=onnx_path,
        engine_path=engine_path,
        fake_bin=fake_bin,
        dynamic_image_size="224x224",
    )

    assert result.returncode == 2
    assert "must be a positive integer" in result.stdout
    assert not engine_path.exists()


def test_build_rejects_unexpected_arguments(tmp_path: Path) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"synthetic ONNX fixture")
    engine_path = tmp_path / "model.engine"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_trtexec(fake_bin, exit_code=0)
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            str(onnx_path),
            str(engine_path),
            "224",
            "unexpected",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "Usage:" in result.stdout
    assert not engine_path.exists()


def test_build_never_overwrites_an_existing_engine(tmp_path: Path) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"synthetic ONNX fixture")
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"existing engine")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_trtexec(fake_bin, exit_code=0)

    result = run_build(
        onnx_path=onnx_path,
        engine_path=engine_path,
        fake_bin=fake_bin,
    )

    assert result.returncode == 1
    assert "will not be overwritten" in result.stdout
    assert engine_path.read_bytes() == b"existing engine"


def test_failed_build_cleans_partial_engine(tmp_path: Path) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"synthetic ONNX fixture")
    engine_path = tmp_path / "model.engine"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_trtexec(fake_bin, exit_code=1)

    result = run_build(
        onnx_path=onnx_path,
        engine_path=engine_path,
        fake_bin=fake_bin,
    )

    assert result.returncode == 1
    assert not engine_path.exists()
    assert not list(tmp_path.glob(".model.engine.building.*"))
