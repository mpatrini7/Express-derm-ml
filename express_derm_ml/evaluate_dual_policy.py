from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import require_absent, write_json_exclusive, write_npz_exclusive
from .common import load_yaml, sha256_file
from .dual_policy import (
    DUAL_POLICY_VERSION,
    apply_dual_policy,
    high_confirmation_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen two-model center-scale policy."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--melanoma-report", required=True)
    parser.add_argument("--melanoma-predictions", required=True)
    parser.add_argument("--broad-report", required=True)
    parser.add_argument("--broad-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-melanoma-superset",
        action="store_true",
        help="Allow the melanoma artifact to contain excluded external rows.",
    )
    return parser.parse_args()


def _load_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "calibrated_probabilities",
            "targets",
            "image_names",
            "view_names",
        }
        missing = required - set(payload.files)
        if missing:
            raise RuntimeError(f"Predictions are missing arrays: {sorted(missing)}")
        return {name: np.asarray(payload[name]) for name in required}


def _aligned_melanoma_indices(
    melanoma_names: np.ndarray,
    broad_names: np.ndarray,
    *,
    allow_superset: bool,
) -> np.ndarray:
    melanoma_text = np.asarray(melanoma_names, dtype=str)
    broad_text = np.asarray(broad_names, dtype=str)
    if len(set(melanoma_text)) != len(melanoma_text):
        raise RuntimeError("Melanoma predictions contain duplicate image names")
    if len(set(broad_text)) != len(broad_text):
        raise RuntimeError("Broad predictions contain duplicate image names")
    if not allow_superset and set(melanoma_text) != set(broad_text):
        raise RuntimeError("Dual prediction image sets do not match")
    lookup = {name: index for index, name in enumerate(melanoma_text)}
    try:
        return np.asarray([lookup[name] for name in broad_text], dtype=np.int64)
    except KeyError as error:
        raise RuntimeError(
            f"Broad prediction image is absent from melanoma artifact: {error}"
        ) from error


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    predictions_output = output_path.with_suffix(".npz")
    require_absent([output_path, predictions_output])

    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    if config.get("version") != DUAL_POLICY_VERSION:
        raise RuntimeError("Dual policy version is unsupported")
    if config.get("research_only") is not True:
        raise RuntimeError("Dual policy must remain explicitly research-only")
    if config.get("thresholds_validated") is not False:
        raise RuntimeError("Dual policy thresholds must remain unvalidated")

    melanoma_report_path = Path(args.melanoma_report).resolve()
    broad_report_path = Path(args.broad_report).resolve()
    melanoma_report = json.loads(melanoma_report_path.read_text(encoding="utf-8"))
    broad_report = json.loads(broad_report_path.read_text(encoding="utf-8"))
    melanoma_config = config["melanoma_attention"]
    broad_config = config["broad_malignancy_attention"]
    if melanoma_report.get("checkpoint_sha256") != melanoma_config[
        "checkpoint_sha256"
    ]:
        raise RuntimeError("Melanoma checkpoint identity does not match policy")
    if broad_report.get("checkpoint_sha256") != broad_config[
        "checkpoint_sha256"
    ]:
        raise RuntimeError("Broad checkpoint identity does not match policy")

    melanoma_predictions_path = Path(args.melanoma_predictions).resolve()
    broad_predictions_path = Path(args.broad_predictions).resolve()
    melanoma = _load_predictions(melanoma_predictions_path)
    broad = _load_predictions(broad_predictions_path)
    if not np.array_equal(melanoma["view_names"], broad["view_names"]):
        raise RuntimeError("Dual prediction view protocols do not match")
    view_names = np.asarray(melanoma["view_names"], dtype=str)
    configured_views = [str(name) for name in config["view_names"]]
    try:
        view_indices = np.asarray(
            [int(np.flatnonzero(view_names == name)[0]) for name in configured_views],
            dtype=np.int64,
        )
    except IndexError as error:
        raise RuntimeError("Configured center-scale view is missing") from error

    melanoma_indices = _aligned_melanoma_indices(
        melanoma["image_names"],
        broad["image_names"],
        allow_superset=args.allow_melanoma_superset,
    )
    result = apply_dual_policy(
        melanoma["calibrated_probabilities"][melanoma_indices][:, view_indices],
        broad["calibrated_probabilities"][:, view_indices],
        melanoma_low_threshold=float(melanoma_config["low_threshold"]),
        melanoma_high_threshold=float(melanoma_config["high_threshold"]),
        broad_low_threshold=float(broad_config["low_threshold"]),
        broad_high_threshold=float(broad_config["high_threshold"]),
        broad_confirmation_threshold=float(
            broad_config["confirmation_threshold"]
        ),
    )
    targets = np.asarray(broad["targets"], dtype=np.int64)
    report = {
        "schema_version": 1,
        "policy_version": DUAL_POLICY_VERSION,
        "config_sha256": sha256_file(config_path),
        "melanoma_report_sha256": sha256_file(melanoma_report_path),
        "melanoma_predictions_sha256": sha256_file(
            melanoma_predictions_path
        ),
        "broad_report_sha256": sha256_file(broad_report_path),
        "broad_predictions_sha256": sha256_file(broad_predictions_path),
        "records": int(len(targets)),
        "view_names": configured_views,
        "attention_counts": {
            label: int(np.sum(result.levels == label))
            for label in (
                "no_elevated_signal",
                "review",
                "high_confirmed",
            )
        },
        "high_confirmation_metrics": high_confirmation_metrics(
            targets,
            result.levels,
        ),
        "confirmation_selection": broad_config["confirmation_selection"],
        "thresholds_validated": False,
        "microscope_validation_pending": True,
        "research_only": True,
    }
    write_npz_exclusive(
        predictions_output,
        image_names=np.asarray(broad["image_names"], dtype=str),
        targets=targets,
        attention_levels=result.levels,
        melanoma_attention_levels=result.melanoma_levels,
        broad_attention_levels=result.broad_levels,
        melanoma_minimum_probability=result.melanoma_minimum,
        melanoma_maximum_probability=result.melanoma_maximum,
        broad_minimum_probability=result.broad_minimum,
        broad_maximum_probability=result.broad_maximum,
    )
    write_json_exclusive(output_path, report)
    print(report)


if __name__ == "__main__":
    main()
