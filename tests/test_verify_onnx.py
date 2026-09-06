from __future__ import annotations

import numpy as np
import pytest

from express_derm_ml.verify_onnx import comparison_summary


def test_comparison_summary_accepts_outputs_within_tolerance() -> None:
    summary = comparison_summary(
        np.asarray([0.0, 1.0], dtype=np.float32),
        np.asarray([0.0, 1.00001], dtype=np.float32),
        absolute_tolerance=1e-4,
    )

    assert summary["passed"] is True
    assert summary["maximum_absolute_error"] == pytest.approx(
        1e-5,
        abs=1e-7,
    )


def test_comparison_summary_rejects_different_shapes() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        comparison_summary(
            np.asarray([0.0], dtype=np.float32),
            np.asarray([[0.0]], dtype=np.float32),
            absolute_tolerance=1e-4,
        )
