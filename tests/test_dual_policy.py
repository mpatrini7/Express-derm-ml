import numpy as np

from express_derm_ml.dual_policy import (
    apply_dual_policy,
    high_confirmation_metrics,
)


def test_dual_policy_confirms_high_and_requires_both_for_low() -> None:
    melanoma = np.asarray(
        [
            [0.9, 0.91, 0.92],
            [0.9, 0.91, 0.92],
            [0.01, 0.02, 0.03],
            [0.01, 0.02, 0.03],
        ]
    )
    broad = np.asarray(
        [
            [0.85, 0.86, 0.87],
            [0.75, 0.76, 0.77],
            [0.01, 0.02, 0.03],
            [0.01, 0.2, 0.03],
        ]
    )

    result = apply_dual_policy(
        melanoma,
        broad,
        melanoma_low_threshold=0.1,
        melanoma_high_threshold=0.8,
        broad_low_threshold=0.1,
        broad_high_threshold=0.7,
        broad_confirmation_threshold=0.8,
    )

    assert result.levels.tolist() == [
        "high_confirmed",
        "review",
        "no_elevated_signal",
        "review",
    ]


def test_high_confirmation_metrics_do_not_treat_review_as_low_evidence() -> None:
    metrics = high_confirmation_metrics(
        np.asarray([1, 1, 0, 0]),
        np.asarray(
            ["high_confirmed", "review", "high_confirmed", "review"]
        ),
    )

    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5
