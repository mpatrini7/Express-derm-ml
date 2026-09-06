from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .dataset import (
    STRETCH_PREPROCESSING_VERSION,
    normalize_deployment_rgb,
    validate_preprocessing_version,
)
from .path_safety import resolve_manifest_image_path


MULTIVIEW_PROTOCOL_VERSION = "orientation4_center_scale3_consensus_v1"
VIEW_NAMES = (
    "original",
    "horizontal_flip",
    "vertical_flip",
    "rotate_180",
    "center_crop_90",
    "center_crop_80",
)
POLICY_VIEW_NAMES = {
    "orientation4_consensus": VIEW_NAMES[:4],
    "center_scale3_consensus": (
        "original",
        "center_crop_90",
        "center_crop_80",
    ),
    "combined6_consensus": VIEW_NAMES,
}


def _center_crop(rgb: np.ndarray, fraction: float) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Center-crop fraction must be in (0, 1]")
    height, width = rgb.shape[:2]
    crop_height = max(1, min(height, round(height * fraction)))
    crop_width = max(1, min(width, round(width * fraction)))
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return np.ascontiguousarray(
        rgb[top : top + crop_height, left : left + crop_width]
    )


def multiview_resize_rgb(
    image_path: str | Path,
    image_size: int,
    *,
    preprocessing: str,
) -> np.ndarray:
    """Create immutable, deterministic views for the selected 224px model."""
    preprocessing = validate_preprocessing_version(preprocessing)
    if preprocessing != STRETCH_PREPROCESSING_VERSION:
        raise ValueError(
            "The v1 multi-view protocol is defined only for the historical "
            "stretch preprocessing used by express-derm-1"
        )
    if image_size <= 0:
        raise ValueError("Image size must be positive")
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Unable to decode image with OpenCV: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    original = cv2.resize(
        rgb,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )
    transformed = (
        original,
        cv2.flip(original, 1),
        cv2.flip(original, 0),
        cv2.flip(original, -1),
        cv2.resize(
            _center_crop(rgb, 0.90),
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        ),
        cv2.resize(
            _center_crop(rgb, 0.80),
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        ),
    )
    return np.stack(transformed, axis=0)


def normalize_multiview_rgb(views: np.ndarray) -> torch.Tensor:
    if views.ndim != 4 or views.shape[-1] != 3:
        raise ValueError("Multi-view pixels must have shape [views, H, W, 3]")
    return torch.stack(
        [normalize_deployment_rgb(view) for view in views],
        dim=0,
    )


def policy_view_indices(policy_name: str) -> tuple[int, ...]:
    try:
        names = POLICY_VIEW_NAMES[policy_name]
    except KeyError as error:
        raise ValueError(f"Unknown multi-view policy: {policy_name}") from error
    return tuple(VIEW_NAMES.index(name) for name in names)


def attention_from_consensus(
    calibrated_probabilities: np.ndarray,
    *,
    policy_name: str,
    low_threshold: float,
    high_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.asarray(calibrated_probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(VIEW_NAMES):
        raise ValueError(
            f"Multi-view probabilities must have shape [N, {len(VIEW_NAMES)}]"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("Multi-view probabilities must be finite")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Multi-view probabilities must be in [0, 1]")
    if not 0.0 <= low_threshold < high_threshold <= 1.0:
        raise ValueError("Multi-view thresholds must be ordered in [0, 1]")

    selected = probabilities[:, policy_view_indices(policy_name)]
    minimum = selected.min(axis=1)
    maximum = selected.max(axis=1)
    levels = np.full(len(selected), "inconclusive", dtype="<U12")
    levels[maximum < low_threshold] = "low"
    levels[minimum >= high_threshold] = "high"
    return levels, minimum, maximum


class MultiViewDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        images_dir: str | Path,
        image_size: int,
        *,
        preprocessing: str,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.images_dir = Path(images_dir).resolve()
        if not self.images_dir.is_dir():
            raise ValueError(
                f"Images directory does not exist: {self.images_dir}"
            )
        self.image_size = image_size
        self.preprocessing = validate_preprocessing_version(preprocessing)

    def __len__(self) -> int:
        return len(self.frame)

    def resolve_image_path(self, index: int) -> Path:
        raw_path = str(self.frame.iloc[index]["image_path"])
        return resolve_manifest_image_path(self.images_dir, raw_path)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        views = multiview_resize_rgb(
            self.resolve_image_path(index),
            self.image_size,
            preprocessing=self.preprocessing,
        )
        return (
            normalize_multiview_rgb(views),
            torch.tensor(float(row["target"]), dtype=torch.float32),
            str(row["image_name"]),
        )
