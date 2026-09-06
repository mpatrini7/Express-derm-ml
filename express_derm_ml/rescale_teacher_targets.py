from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .artifacts import create_new_directory, write_json_exclusive, write_npz_exclusive
from .common import sha256_file


def rescale_calibrated_logits(
    calibrated_logits: np.ndarray,
    *,
    logit_scale: float,
    logit_bias: float,
) -> np.ndarray:
    values = np.asarray(calibrated_logits, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Teacher logits must be finite")
    if not np.isfinite(logit_scale) or logit_scale <= 0.0:
        raise ValueError("Teacher logit scale must be finite and positive")
    if not np.isfinite(logit_bias):
        raise ValueError("Teacher logit bias must be finite")
    return (values - logit_bias) / logit_scale


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map calibrated teacher logits onto one component's raw-logit "
            "scale without changing ranking or weights."
        )
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scale-component", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    source_targets_path = input_dir / "teacher_targets.npz"
    source_receipt_path = input_dir / "teacher_targets.json"
    source_receipt = json.loads(
        source_receipt_path.read_text(encoding="utf-8")
    )
    if source_receipt.get("status") != "complete":
        raise RuntimeError("Source teacher target receipt is incomplete")
    source_targets_sha256 = sha256_file(source_targets_path)
    if source_targets_sha256 != source_receipt.get("teacher_targets_sha256"):
        raise RuntimeError("Source teacher target hash mismatch")
    components = source_receipt.get("components")
    if not isinstance(components, list):
        raise RuntimeError("Source teacher components are unavailable")
    matching = [
        component
        for component in components
        if str(component.get("name")) == args.scale_component
    ]
    if len(matching) != 1:
        raise RuntimeError("Requested teacher scale component is unavailable")
    scale_component = matching[0]
    logit_scale = float(scale_component["logit_scale"])
    logit_bias = float(scale_component["logit_bias"])

    with np.load(source_targets_path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    required = {
        "image_names",
        "hard_targets",
        "teacher_scores",
        "component_names",
        "component_logits",
    }
    if not required.issubset(arrays):
        raise RuntimeError("Source teacher target artifact is incomplete")
    rescaled_scores = rescale_calibrated_logits(
        arrays["teacher_scores"],
        logit_scale=logit_scale,
        logit_bias=logit_bias,
    ).astype(np.float32)

    output_dir = create_new_directory(args.output_dir)
    output_targets_path = output_dir / "teacher_targets.npz"
    arrays["teacher_scores"] = rescaled_scores
    write_npz_exclusive(output_targets_path, **arrays)
    scale_contract = {
        "source_component_set_sha256": source_receipt[
            "ensemble_component_set_sha256"
        ],
        "source_teacher_targets_sha256": source_targets_sha256,
        "scale_component": args.scale_component,
        "scale_component_checkpoint_sha256": scale_component[
            "checkpoint_sha256"
        ],
        "logit_scale": logit_scale,
        "logit_bias": logit_bias,
        "transform": "(weighted_calibrated_logit-bias)/scale",
    }
    receipt = dict(source_receipt)
    receipt.update(
        {
            "target_kind": (
                "validation_selected_weighted_generalist_scale_logit"
            ),
            "teacher_score_scale": scale_contract,
            "source_teacher_receipt_sha256": sha256_file(
                source_receipt_path
            ),
            "ensemble_component_set_sha256": _canonical_sha256(
                scale_contract
            ),
            "teacher_targets_sha256": sha256_file(output_targets_path),
        }
    )
    write_json_exclusive(output_dir / "teacher_targets.json", receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
