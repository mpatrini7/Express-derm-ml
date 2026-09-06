from __future__ import annotations

import numpy as np
import pytest

from express_derm_ml.generate_weighted_teacher_targets import (
    select_two_component_weights,
)


def test_weight_selection_uses_validation_pr_auc() -> None:
    targets = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    logits = {
        "specialist": np.asarray([-4.0, -3.0, 2.0, 1.0, 0.5]),
        "generalist": np.asarray([-1.0, 1.0, -2.0, 3.0, 2.0]),
    }

    weights, results = select_two_component_weights(
        logits,
        targets,
        generalist_name="generalist",
        grid_step=0.5,
    )

    assert len(results) == 3
    assert weights == {"generalist": 1.0, "specialist": 0.0}


def test_weight_selection_rejects_invalid_grid() -> None:
    with pytest.raises(ValueError, match="divide one"):
        select_two_component_weights(
            {
                "specialist": np.asarray([-1.0, 1.0]),
                "generalist": np.asarray([-2.0, 2.0]),
            },
            np.asarray([0, 1]),
            generalist_name="generalist",
            grid_step=0.3,
        )
