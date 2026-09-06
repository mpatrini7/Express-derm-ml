from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from .artifacts import require_absent, write_json_exclusive
from .common import sha256_file
from .dataset import (
    LesionDataset,
    validate_preprocessing_version,
)
from .manifest import read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare an exported ONNX graph with frozen PyTorch test logits."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-4)
    return parser.parse_args()


def comparison_summary(
    reference: np.ndarray,
    candidate: np.ndarray,
    absolute_tolerance: float,
) -> dict[str, float | bool]:
    if reference.shape != candidate.shape:
        raise ValueError("Reference and candidate output shapes differ")
    absolute_error = np.abs(reference - candidate)
    maximum = float(absolute_error.max(initial=0.0))
    return {
        "maximum_absolute_error": maximum,
        "mean_absolute_error": float(absolute_error.mean()),
        "absolute_tolerance": float(absolute_tolerance),
        "passed": maximum <= absolute_tolerance,
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output)
    require_absent([output_path])
    if args.sample_size <= 0:
        raise ValueError("Sample size must be positive")
    if args.absolute_tolerance <= 0:
        raise ValueError("Absolute tolerance must be positive")

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    runtime = json.loads(
        (model_dir / "manifest.json").read_text(encoding="utf-8")
    )
    model_path = model_dir / "model.onnx"
    predictions_path = run_dir / "predictions_test.npz"

    if run.get("status") != "complete":
        raise RuntimeError("Training run is not complete")
    if sha256_file(model_path) != runtime.get("model_sha256"):
        raise RuntimeError("ONNX model hash does not match manifest")
    if run.get("manifest_sha256") != runtime.get(
        "dataset_manifest_sha256"
    ):
        raise RuntimeError("Dataset manifest hash mismatch")
    if run.get("checkpoint_sha256") != runtime.get("checkpoint_sha256"):
        raise RuntimeError("Checkpoint hash mismatch")
    if sha256_file(model_dir / "calibration.json") != runtime.get(
        "calibration_sha256"
    ):
        raise RuntimeError("Calibration hash mismatch")
    preprocessing = validate_preprocessing_version(
        str(runtime.get("preprocessing", {}).get("version", ""))
    )
    if run.get("evaluation_preprocessing") != preprocessing:
        raise RuntimeError("Run and runtime preprocessing do not match")

    onnx.checker.check_model(onnx.load(model_path))
    frame = read_manifest(run_dir / "manifest.csv")
    test_frame = frame.loc[frame["split"] == "test"].reset_index(drop=True)
    with np.load(predictions_path, allow_pickle=False) as predictions:
        reference_logits = predictions["logits"].astype(np.float32)
        prediction_names = predictions["image_names"].astype(str)
    manifest_names = test_frame["image_name"].to_numpy(dtype=str)
    if not np.array_equal(prediction_names, manifest_names):
        raise RuntimeError("Frozen test predictions do not match manifest order")

    sample_size = min(args.sample_size, len(test_frame))
    indices = np.linspace(
        0,
        len(test_frame) - 1,
        num=sample_size,
        dtype=np.int64,
    )
    sample_frame = test_frame.iloc[indices].reset_index(drop=True)
    dataset = LesionDataset(
        sample_frame,
        args.images_dir,
        int(runtime["image_size"]),
        training=False,
        preprocessing=preprocessing,
    )
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    candidate_logits = []
    for index in range(len(dataset)):
        tensor, _, _ = dataset[index]
        output = session.run(
            [runtime["output_name"]],
            {runtime["input_name"]: tensor.numpy()[None, ...]},
        )[0]
        candidate_logits.append(float(np.asarray(output).reshape(-1)[0]))

    reference = reference_logits[indices]
    candidate = np.asarray(candidate_logits, dtype=np.float32)
    summary = comparison_summary(
        reference,
        candidate,
        args.absolute_tolerance,
    )
    report = {
        "schema_version": 1,
        "model_version": runtime["version"],
        "model_sha256": runtime["model_sha256"],
        "checkpoint_sha256": runtime["checkpoint_sha256"],
        "dataset_manifest_sha256": runtime["dataset_manifest_sha256"],
        "reference_predictions_sha256": sha256_file(predictions_path),
        "reference_backend": "pytorch_cpu",
        "candidate_backend": "onnxruntime_cpu",
        "preprocessing": preprocessing,
        "selection": "evenly_spaced_test_records",
        "sample_size": sample_size,
        **summary,
        "research_only": True,
        "inference_enabled": False,
    }
    write_json_exclusive(output_path, report)
    print(report)
    if not summary["passed"]:
        raise RuntimeError("ONNX parity tolerance exceeded")


if __name__ == "__main__":
    main()
