from __future__ import annotations

import numpy as np
import pytest

from express_derm_ml.rescale_teacher_targets import (
    rescale_calibrated_logits,
)


def test_rescale_calibrated_logits_inverts_affine_calibration() -> None:
    raw = np.asarray([-2.0, 0.0, 3.0])
    scale = 1.75
    bias = -2.5
    calibrated = scale * raw + bias

    actual = rescale_calibrated_logits(
        calibrated,
        logit_scale=scale,
        logit_bias=bias,
    )

    assert np.allclose(actual, raw)


def test_rescale_calibrated_logits_rejects_invalid_scale() -> None:
    with pytest.raises(ValueError, match="positive"):
        rescale_calibrated_logits(
            np.asarray([0.0]),
            logit_scale=0.0,
            logit_bias=0.0,
        )
