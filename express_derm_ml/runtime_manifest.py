from __future__ import annotations

from typing import Any

from .dataset import (
    DEPLOYMENT_PREPROCESSING_VERSION,
    checkpoint_preprocessing,
    preprocessing_manifest_fields,
    validate_preprocessing_version,
)


class RuntimeManifestError(RuntimeError):
    pass


MODEL_FAMILY = "express-derm"


def build_research_runtime_manifest(
    *,
    model_version: str,
    model_release: str | None = None,
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    run: dict[str, Any],
    calibration: dict[str, Any],
    model_sha256: str,
    checkpoint_sha256: str,
    calibration_sha256: str,
) -> dict[str, Any]:
    if not model_version.strip():
        raise RuntimeManifestError("Model version cannot be blank")
    release_version = model_release or model_version
    if not release_version.strip():
        raise RuntimeManifestError("Model release cannot be blank")
    if calibration.get("selection_split") != "validation":
        raise RuntimeManifestError(
            "Runtime calibration must originate from validation"
        )
    if calibration.get("thresholds_validated") is not False:
        raise RuntimeManifestError(
            "Development calibration must remain explicitly unvalidated"
        )
    preprocessing = validate_preprocessing_version(
        str(
            config.get("deployment", {}).get(
                "preprocessing",
                DEPLOYMENT_PREPROCESSING_VERSION,
            )
        )
    )
    if checkpoint_preprocessing(checkpoint) != preprocessing:
        raise RuntimeManifestError(
            "Checkpoint preprocessing does not match the deployment contract"
        )
    if config.get("deployment", {}).get(
        "preprocessing",
        DEPLOYMENT_PREPROCESSING_VERSION,
    ) != preprocessing:
        raise RuntimeManifestError(
            "Config preprocessing does not match the deployment runtime contract"
        )
    if run.get("evaluation_preprocessing") != preprocessing:
        raise RuntimeManifestError(
            "Run preprocessing does not match the deployment runtime contract"
        )
    if calibration.get("evaluation_preprocessing") != preprocessing:
        raise RuntimeManifestError(
            "Calibration preprocessing does not match the deployment runtime "
            "contract"
        )
    low_threshold = float(calibration["low_threshold"])
    high_threshold = float(calibration["high_threshold"])
    if not 0.0 <= low_threshold < high_threshold <= 1.0:
        raise RuntimeManifestError("Calibration thresholds are not ordered")

    calibration_method = calibration.get("method")
    calibration_fields: dict[str, Any]
    if calibration_method == "affine_logistic_scaling":
        logit_scale = float(calibration["logit_scale"])
        logit_bias = float(calibration["logit_bias"])
        if logit_scale <= 0:
            raise RuntimeManifestError("Calibration scale must be positive")
        calibration_fields = {
            "logit_scale": logit_scale,
            "logit_bias": logit_bias,
        }
    elif calibration_method == "temperature_scaling":
        temperature = float(calibration["temperature"])
        if temperature <= 0:
            raise RuntimeManifestError(
                "Calibration temperature must be positive"
            )
        calibration_fields = {"temperature": temperature}
    else:
        raise RuntimeManifestError(
            f"Unsupported calibration method: {calibration_method}"
        )

    runtime_manifest = {
        "schema_version": 3,
        "version": model_version,
        "release_version": release_version,
        "model_family": MODEL_FAMILY,
        "runtime_identity": MODEL_FAMILY,
        "runtime_model_count": 1,
        "external_runtime_models": [],
        "architecture": checkpoint["architecture"],
        "input_name": config["deployment"]["input_name"],
        "output_name": config["deployment"]["output_name"],
        "image_size": int(checkpoint["image_size"]),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "preprocessing": preprocessing_manifest_fields(preprocessing),
        **calibration_fields,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "abstention_margin": float(
            config["calibration"]["development_abstention_margin"]
        ),
        "validation_status": "research_only",
        "domain_status": "microscope_validation_pending",
        "model_sha256": model_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_sha256": calibration_sha256,
        "dataset_manifest_sha256": run["manifest_sha256"],
        "calibration_method": calibration_method,
        "threshold_status": "development_frozen",
        "development_thresholds_frozen": True,
        "thresholds_validated": False,
        "research_only": True,
    }
    if run.get("candidate_type") == "distilled_student":
        training_provenance = {
            "candidate_type": "distilled_student",
            "teacher_ensemble_component_set_sha256": run[
                "teacher_ensemble_component_set_sha256"
            ],
            "teacher_targets_sha256": run["teacher_targets_sha256"],
            "initial_checkpoint_sha256": run[
                "initial_checkpoint_sha256"
            ],
        }
        optional_teacher_fields = (
            "teacher_target_kind",
            "teacher_ensemble_json_sha256",
            "teacher_score_scale",
            "teacher_source_receipt_sha256",
        )
        training_provenance.update(
            {
                field: run[field]
                for field in optional_teacher_fields
                if run.get(field) is not None
            }
        )
        runtime_manifest["training_provenance"] = training_provenance
    return runtime_manifest
