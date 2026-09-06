from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .artifacts import (
    copy_file_exclusive,
    create_new_directory,
    write_json_exclusive,
)
from .common import load_yaml, sha256_file
from .dataset import (
    DEPLOYMENT_PREPROCESSING_VERSION,
    checkpoint_preprocessing,
    validate_preprocessing_version,
)
from .model import LogitWrapper, create_model
from .runtime_manifest import build_research_runtime_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--model-release")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output = Path(args.output)
    if output.name != "model.onnx":
        raise ValueError("ONNX output filename must be model.onnx")

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if run.get("status") != "complete":
        raise RuntimeError("Training run is not complete")
    preprocessing = validate_preprocessing_version(
        str(
            run.get(
                "evaluation_preprocessing",
                DEPLOYMENT_PREPROCESSING_VERSION,
            )
        )
    )
    config_path = run_dir / "config.yaml"
    if sha256_file(config_path) != run.get("config_sha256"):
        raise RuntimeError("Config hash does not match run.json")
    calibration_path = run_dir / "calibration.json"
    calibration = json.loads(
        calibration_path.read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (run_dir / "metrics_test.json").read_text(encoding="utf-8")
    )
    checkpoint_path = run_dir / "best.pt"
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != run.get("checkpoint_sha256"):
        raise RuntimeError("Checkpoint hash does not match run.json")
    if checkpoint_sha256 != calibration.get("checkpoint_sha256"):
        raise RuntimeError("Calibration checkpoint hash mismatch")
    if run.get("manifest_sha256") != calibration.get("manifest_sha256"):
        raise RuntimeError("Calibration manifest hash mismatch")
    if calibration.get("evaluation_preprocessing") != preprocessing:
        raise RuntimeError("Deployment preprocessing contract mismatch")
    if metrics.get("evaluation_preprocessing") != preprocessing:
        raise RuntimeError("Test-metrics preprocessing contract mismatch")
    if metrics.get("selection_split") != "validation":
        raise RuntimeError("Test metrics do not use validation calibration")
    calibration_sha256 = sha256_file(calibration_path)
    if metrics.get("calibration_sha256") != calibration_sha256:
        raise RuntimeError("Test metrics calibration hash mismatch")
    if metrics.get("manifest_sha256") != run.get("manifest_sha256"):
        raise RuntimeError("Test metrics manifest hash mismatch")
    if metrics.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError("Test metrics checkpoint hash mismatch")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint_preprocessing(checkpoint) != preprocessing:
        raise RuntimeError("Run and checkpoint preprocessing do not match")
    config = load_yaml(config_path)
    model = create_model(checkpoint["architecture"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model = LogitWrapper(model).eval()

    image_size = int(checkpoint["image_size"])
    dummy = torch.zeros(1, 3, image_size, image_size)
    input_name = config["deployment"]["input_name"]
    output_name = config["deployment"]["output_name"]

    build_research_runtime_manifest(
        model_version=args.model_version,
        model_release=args.model_release,
        checkpoint=checkpoint,
        config=config,
        run=run,
        calibration=calibration,
        model_sha256="pending",
        checkpoint_sha256=checkpoint_sha256,
        calibration_sha256=calibration_sha256,
    )
    create_new_directory(output.parent)
    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=[input_name],
        output_names=[output_name],
        opset_version=int(config["deployment"]["opset"]),
        do_constant_folding=True,
        dynamic_axes=None,
    )

    manifest = build_research_runtime_manifest(
        model_version=args.model_version,
        model_release=args.model_release,
        checkpoint=checkpoint,
        config=config,
        run=run,
        calibration=calibration,
        model_sha256=sha256_file(output),
        checkpoint_sha256=checkpoint_sha256,
        calibration_sha256=calibration_sha256,
    )
    write_json_exclusive(output.parent / "manifest.json", manifest)

    for filename in (
        "calibration.json",
        "metrics_test.json",
        "run.json",
        "config.yaml",
        "history.json",
        "manifest.csv",
        "manifest.csv.sha256",
        "manifest.report.json",
        "manifest.near_duplicates.csv",
        "preflight.json",
    ):
        source = run_dir / filename
        copy_file_exclusive(source, output.parent / filename)

    for filename in ("teacher_targets.json",):
        source = run_dir / filename
        if source.is_file():
            copy_file_exclusive(source, output.parent / filename)

    print(f"Exported ONNX model to {output}")
    print(f"Wrote runtime manifest to {output.parent / 'manifest.json'}")


if __name__ == "__main__":
    main()
