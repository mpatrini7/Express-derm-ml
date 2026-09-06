from __future__ import annotations

import cv2
import numpy as np
import pytest

from express_derm_ml.dataset import STRETCH_PREPROCESSING_VERSION
from express_derm_ml.evaluate_multiview import _validate_reference_predictions
from express_derm_ml.multiview import (
    POLICY_VIEW_NAMES,
    VIEW_NAMES,
    attention_from_consensus,
    multiview_resize_rgb,
    normalize_multiview_rgb,
    policy_view_indices,
)


def test_multiview_pixels_are_deterministic_and_original_has_runtime_parity(
    tmp_path,
) -> None:
    bgr = np.zeros((8, 12, 3), dtype=np.uint8)
    bgr[:, :6] = (10, 20, 30)
    bgr[:, 6:] = (100, 110, 120)
    path = tmp_path / "sample.png"
    assert cv2.imwrite(str(path), bgr)

    first = multiview_resize_rgb(
        path,
        6,
        preprocessing=STRETCH_PREPROCESSING_VERSION,
    )
    second = multiview_resize_rgb(
        path,
        6,
        preprocessing=STRETCH_PREPROCESSING_VERSION,
    )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    expected_original = cv2.resize(
        rgb,
        (6, 6),
        interpolation=cv2.INTER_AREA,
    )
    assert first.shape == (len(VIEW_NAMES), 6, 6, 3)
    assert np.array_equal(first, second)
    assert np.array_equal(first[0], expected_original)
    assert np.array_equal(first[1], cv2.flip(expected_original, 1))
    assert np.array_equal(first[2], cv2.flip(expected_original, 0))
    assert np.array_equal(first[3], cv2.flip(expected_original, -1))
    assert normalize_multiview_rgb(first).shape == (len(VIEW_NAMES), 3, 6, 6)


def test_multiview_rejects_non_deployment_preprocessing(tmp_path) -> None:
    path = tmp_path / "sample.png"
    assert cv2.imwrite(str(path), np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="express-derm-1"):
        multiview_resize_rgb(
            path,
            4,
            preprocessing=(
                "opencv_imread_bgr2rgb_inter_area_"
                "letterbox_imagenetmean_nchw_float32_v1"
            ),
        )


def test_consensus_requires_every_selected_view_to_agree() -> None:
    probabilities = np.asarray(
        [
            [0.9, 0.8, 0.7, 0.6, 0.9, 0.9],
            [0.9, 0.8, 0.1, 0.6, 0.9, 0.9],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ]
    )
    levels, minimum, maximum = attention_from_consensus(
        probabilities,
        policy_name="orientation4_consensus",
        low_threshold=0.2,
        high_threshold=0.5,
    )
    assert levels.tolist() == ["high", "inconclusive", "low"]
    assert minimum.tolist() == [0.6, 0.1, 0.1]
    assert maximum.tolist() == [0.9, 0.9, 0.1]


def test_policy_view_indices_match_declared_names() -> None:
    for policy_name, names in POLICY_VIEW_NAMES.items():
        indices = policy_view_indices(policy_name)
        assert tuple(VIEW_NAMES[index] for index in indices) == names
    with pytest.raises(ValueError, match="Unknown multi-view policy"):
        policy_view_indices("missing")


def test_reference_parity_has_a_bounded_mps_tolerance(tmp_path) -> None:
    reference_path = tmp_path / "reference.npz"
    np.savez_compressed(
        reference_path,
        logits=np.asarray([0.0, 1.0]),
        targets=np.asarray([0, 1]),
        image_names=np.asarray(["a", "b"]),
    )
    evidence = _validate_reference_predictions(
        reference_path,
        logits=np.asarray([[0.0013], [1.0013]]),
        targets=np.asarray([0, 1]),
        image_names=np.asarray(["a", "b"]),
        atol=0.002,
    )
    assert evidence["status"] == "pass"
    with pytest.raises(RuntimeError, match="Original-view parity failed"):
        _validate_reference_predictions(
            reference_path,
            logits=np.asarray([[0.003], [1.003]]),
            targets=np.asarray([0, 1]),
            image_names=np.asarray(["a", "b"]),
            atol=0.002,
        )


def test_reference_parity_accepts_external_melanoma_target_key(tmp_path) -> None:
    reference_path = tmp_path / "external.npz"
    np.savez_compressed(
        reference_path,
        logits=np.asarray([0.0, 1.0]),
        melanoma_targets=np.asarray([0, 1]),
        image_names=np.asarray(["a", "b"]),
    )
    evidence = _validate_reference_predictions(
        reference_path,
        logits=np.asarray([[0.0], [1.0]]),
        targets=np.asarray([0, 1]),
        image_names=np.asarray(["a", "b"]),
        atol=0.0,
    )
    assert evidence["reference_target_key"] == "melanoma_targets"
