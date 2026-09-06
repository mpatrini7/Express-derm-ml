from __future__ import annotations

import numpy as np
import pytest
import torch

from express_derm_ml.device import (
    resolve_training_device,
    uses_cuda_transfer_optimizations,
)
from express_derm_ml.train import (
    EpochBalancedSampler,
    SourceClassBalancedSampler,
    collapse_sampling_sources,
    positive_class_weight,
)


def test_auto_device_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert resolve_training_device("auto").type == "cuda"


def test_auto_device_uses_mps_before_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert resolve_training_device("auto").type == "mps"


def test_auto_device_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    device = resolve_training_device("auto")
    assert device.type == "cpu"
    assert uses_cuda_transfer_optimizations(device) is False


@pytest.mark.parametrize(
    ("preference", "message"),
    (
        ("cuda", "CUDA was requested"),
        ("mps", "Apple Metal"),
    ),
)
def test_explicit_unavailable_accelerator_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    preference: str,
    message: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match=message):
        resolve_training_device(preference)


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported training device"):
        resolve_training_device("silicon")


def test_cuda_transfer_optimizations_are_cuda_only() -> None:
    assert uses_cuda_transfer_optimizations(torch.device("cuda")) is True
    assert uses_cuda_transfer_optimizations(torch.device("mps")) is False
    assert uses_cuda_transfer_optimizations(torch.device("cpu")) is False


def test_positive_class_weight_modes_are_explicit() -> None:
    labels = np.array([0] * 99 + [1], dtype=np.int64)

    assert positive_class_weight(labels, "none") == 1.0
    assert positive_class_weight(labels, "sqrt_balanced") == pytest.approx(
        np.sqrt(99.0)
    )
    assert positive_class_weight(labels, "balanced") == pytest.approx(99.0)


def test_epoch_balanced_sampler_keeps_positives_and_rotates_negatives() -> None:
    labels = np.array([0] * 20 + [1] * 2, dtype=np.int64)
    sampler = EpochBalancedSampler(
        labels,
        negative_to_positive_ratio=3,
        seed=12,
    )

    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)

    assert len(first) == len(second) == 8
    assert set(np.flatnonzero(labels == 1)).issubset(first)
    assert set(np.flatnonzero(labels == 1)).issubset(second)
    assert len(set(first)) == len(first)
    assert set(first) != set(second)


def test_epoch_balanced_sampler_always_keeps_priority_negatives() -> None:
    labels = np.array([0] * 20 + [1] * 2, dtype=np.int64)
    required = np.array([2, 7, 11], dtype=np.int64)
    sampler = EpochBalancedSampler(
        labels,
        negative_to_positive_ratio=3,
        seed=12,
        always_include_negative_indices=required,
    )

    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)

    assert len(first) == len(second) == 8
    assert set(required).issubset(first)
    assert set(required).issubset(second)
    assert set(first) != set(second)


def test_epoch_balanced_sampler_rejects_non_negative_priority() -> None:
    labels = np.array([0, 0, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="must all be negatives"):
        EpochBalancedSampler(
            labels,
            negative_to_positive_ratio=1,
            seed=12,
            always_include_negative_indices=np.array([2]),
        )


def test_epoch_balanced_sampler_repeats_priority_within_fixed_capacity() -> None:
    labels = np.array([0] * 20 + [1] * 2, dtype=np.int64)
    sampler = EpochBalancedSampler(
        labels,
        negative_to_positive_ratio=3,
        seed=12,
        always_include_negative_indices=np.array([2, 7]),
        always_include_negative_repeats=2,
    )

    sampled = list(sampler)

    assert len(sampled) == 8
    assert sampled.count(2) == 2
    assert sampled.count(7) == 2
    assert int(labels[sampled].sum()) == 2


def test_source_balanced_sampler_removes_source_label_correlation() -> None:
    labels = np.array(
        [1] * 2 + [0] * 12 + [1] * 5 + [0] * 6,
        dtype=np.int64,
    )
    sources = np.array(["primary"] * 14 + ["added"] * 11)
    sampler = SourceClassBalancedSampler(
        labels,
        sources,
        negative_to_positive_ratio=2,
        seed=12,
    )

    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)

    assert len(first) == len(second) == 12
    assert len(set(first)) == len(first)
    assert set(first) != set(second)
    for source in ("primary", "added"):
        source_indices = [index for index in first if sources[index] == source]
        selected_labels = labels[source_indices]
        assert int(selected_labels.sum()) == 2
        assert int((selected_labels == 0).sum()) == 4


def test_source_balanced_sampler_rejects_single_class_source() -> None:
    labels = np.array([0, 1, 0, 0], dtype=np.int64)
    sources = np.array(["primary", "primary", "added", "added"])

    with pytest.raises(ValueError, match="must contain both targets"):
        SourceClassBalancedSampler(
            labels,
            sources,
            negative_to_positive_ratio=2,
            seed=12,
        )


def test_sampling_sources_can_be_collapsed_into_declared_groups() -> None:
    sources = np.array(["isic-2020", "isic-2019", "dicm"])

    grouped = collapse_sampling_sources(
        sources,
        {
            "historical": ["isic-2020", "isic-2019"],
            "added": ["dicm"],
        },
    )

    assert grouped.tolist() == ["historical", "historical", "added"]


def test_sampling_source_groups_reject_unmapped_sources() -> None:
    with pytest.raises(ValueError, match="not grouped"):
        collapse_sampling_sources(
            np.array(["isic-2020", "dicm"]),
            {"historical": ["isic-2020"]},
        )
