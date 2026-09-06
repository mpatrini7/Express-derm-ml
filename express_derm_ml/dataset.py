from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from .path_safety import resolve_manifest_image_path


STRETCH_PREPROCESSING_VERSION = (
    "opencv_imread_bgr2rgb_inter_area_stretch_imagenet_nchw_float32_v1"
)
LETTERBOX_PREPROCESSING_VERSION = (
    "opencv_imread_bgr2rgb_inter_area_letterbox_imagenetmean_nchw_float32_v1"
)
# Backward-compatible name for every already-versioned model artifact.
DEPLOYMENT_PREPROCESSING_VERSION = STRETCH_PREPROCESSING_VERSION
SUPPORTED_PREPROCESSING_VERSIONS = frozenset(
    {STRETCH_PREPROCESSING_VERSION, LETTERBOX_PREPROCESSING_VERSION}
)
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
LETTERBOX_FILL_RGB = tuple(
    int(round(float(channel) * 255.0)) for channel in IMAGENET_MEAN
)


def validate_preprocessing_version(preprocessing: str) -> str:
    preprocessing = str(preprocessing)
    if preprocessing not in SUPPORTED_PREPROCESSING_VERSIONS:
        raise ValueError(
            f"Unsupported deployment preprocessing contract: {preprocessing}"
        )
    return preprocessing


def checkpoint_preprocessing(checkpoint: dict[str, object]) -> str:
    """Resolve preprocessing while retaining compatibility with old runs."""
    return validate_preprocessing_version(
        str(
            checkpoint.get(
                "preprocessing",
                STRETCH_PREPROCESSING_VERSION,
            )
        )
    )


def preprocessing_manifest_fields(preprocessing: str) -> dict[str, str]:
    preprocessing = validate_preprocessing_version(preprocessing)
    geometry = (
        "stretch_square"
        if preprocessing == STRETCH_PREPROCESSING_VERSION
        else "letterbox_square_imagenet_mean"
    )
    return {
        "version": preprocessing,
        "decoder": "opencv_imread_color",
        "color_conversion": "bgr_to_rgb",
        "resize_interpolation": "area",
        "resize_geometry": geometry,
        "pixel_scale": "uint8_div_255",
        "layout": "nchw",
        "dtype": "float32",
    }


def deployment_preprocess(
    image_path: str | Path,
    image_size: int,
    preprocessing: str = DEPLOYMENT_PREPROCESSING_VERSION,
) -> torch.Tensor:
    """Match the OpenCV/C++ deployment preprocessing contract exactly."""
    resized = deployment_resize_rgb(
        image_path,
        image_size,
        preprocessing=preprocessing,
    )
    return normalize_deployment_rgb(resized)


def deployment_resize_rgb(
    image_path: str | Path,
    image_size: int,
    preprocessing: str = DEPLOYMENT_PREPROCESSING_VERSION,
) -> np.ndarray:
    """Decode and resize to the exact uint8 RGB pixels used at runtime."""
    preprocessing = validate_preprocessing_version(preprocessing)
    if image_size <= 0:
        raise ValueError("Image size must be positive")
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Unable to decode image with OpenCV: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if preprocessing == STRETCH_PREPROCESSING_VERSION:
        return cv2.resize(
            rgb,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )

    source_height, source_width = rgb.shape[:2]
    scale = min(image_size / source_width, image_size / source_height)
    target_width = max(1, min(image_size, round(source_width * scale)))
    target_height = max(1, min(image_size, round(source_height * scale)))
    resized = cv2.resize(
        rgb,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.empty((image_size, image_size, 3), dtype=np.uint8)
    canvas[...] = LETTERBOX_FILL_RGB
    left = (image_size - target_width) // 2
    top = (image_size - target_height) // 2
    canvas[top : top + target_height, left : left + target_width] = resized
    return canvas


def normalize_deployment_rgb(resized: np.ndarray) -> torch.Tensor:
    tensor = resized.astype(np.float32) / np.float32(255.0)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    nchw = np.ascontiguousarray(np.transpose(tensor, (2, 0, 1)))
    return torch.from_numpy(nchw)


class _PILLetterboxSquare:
    def __init__(self, image_size: int):
        if image_size <= 0:
            raise ValueError("Image size must be positive")
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> Image.Image:
        source_width, source_height = image.size
        scale = min(
            self.image_size / source_width,
            self.image_size / source_height,
        )
        target_width = max(
            1,
            min(self.image_size, round(source_width * scale)),
        )
        target_height = max(
            1,
            min(self.image_size, round(source_height * scale)),
        )
        resized = image.resize(
            (target_width, target_height),
            resample=Image.Resampling.BOX,
        )
        canvas = Image.new(
            "RGB",
            (self.image_size, self.image_size),
            color=LETTERBOX_FILL_RGB,
        )
        canvas.paste(
            resized,
            (
                (self.image_size - target_width) // 2,
                (self.image_size - target_height) // 2,
            ),
        )
        return canvas


def validate_center_crop_scales(scales: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(float(scale) for scale in scales)
    if not normalized:
        raise ValueError("Training center-crop scales cannot be empty")
    if any(not 0.0 < scale <= 1.0 for scale in normalized):
        raise ValueError("Training center-crop scales must be in (0, 1]")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Training center-crop scales must be unique")
    if 1.0 not in normalized:
        raise ValueError("Training center-crop scales must include 1.0")
    return normalized


class _RandomCenterScaleCrop:
    def __init__(self, scales: Sequence[float]):
        self.scales = validate_center_crop_scales(scales)

    def __call__(self, image: Image.Image) -> Image.Image:
        choice = int(torch.randint(len(self.scales), size=(1,)).item())
        scale = self.scales[choice]
        if scale == 1.0:
            return image
        width, height = image.size
        crop_width = max(1, min(width, round(width * scale)))
        crop_height = max(1, min(height, round(height * scale)))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        return image.crop((left, top, left + crop_width, top + crop_height))


def build_transforms(
    image_size: int,
    training: bool,
    preprocessing: str = DEPLOYMENT_PREPROCESSING_VERSION,
    center_crop_scales: Sequence[float] = (1.0,),
):
    preprocessing = validate_preprocessing_version(preprocessing)
    transforms = []
    normalized_scales = validate_center_crop_scales(center_crop_scales)
    if training and normalized_scales != (1.0,):
        transforms.append(_RandomCenterScaleCrop(normalized_scales))
    if preprocessing == STRETCH_PREPROCESSING_VERSION:
        transforms.extend(
            [
                v2.ToImage(),
                v2.Resize((image_size, image_size), antialias=True),
            ]
        )
    else:
        transforms.extend([_PILLetterboxSquare(image_size), v2.ToImage()])
    if training:
        transforms.extend(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(degrees=180),
                v2.ColorJitter(
                    brightness=0.12,
                    contrast=0.12,
                    saturation=0.08,
                    hue=0.02,
                ),
            ]
        )
    transforms.extend(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return v2.Compose(transforms)


class LesionDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        images_dir: str | Path,
        image_size: int,
        training: bool,
        preprocessing: str = DEPLOYMENT_PREPROCESSING_VERSION,
        center_crop_scales: Sequence[float] = (1.0,),
    ):
        self.frame = frame.reset_index(drop=True)
        self.images_dir = Path(images_dir).resolve()
        if not self.images_dir.is_dir():
            raise ValueError(
                f"Images directory does not exist: {self.images_dir}"
            )
        self.image_size = image_size
        self.training = training
        self.preprocessing = validate_preprocessing_version(preprocessing)
        self.center_crop_scales = validate_center_crop_scales(
            center_crop_scales
        )
        self.transform = (
            build_transforms(
                image_size,
                training,
                preprocessing=self.preprocessing,
                center_crop_scales=self.center_crop_scales,
            )
            if training
            else None
        )
        self._deployment_cache: list[np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self.frame)

    def _resolve_image_path(self, index: int) -> Path:
        row = self.frame.iloc[index]
        raw_path = str(row["image_path"])
        return resolve_manifest_image_path(self.images_dir, raw_path)

    def cache_evaluation_images(self, *, workers: int = 2) -> None:
        """Cache exact resized uint8 pixels once for repeated validation."""
        if self.training:
            raise RuntimeError("Training images cannot use evaluation cache")
        if workers <= 0:
            raise ValueError("Evaluation cache workers must be positive")
        paths = [self._resolve_image_path(index) for index in range(len(self))]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            self._deployment_cache = list(
                executor.map(
                    lambda path: deployment_resize_rgb(
                        path,
                        self.image_size,
                        preprocessing=self.preprocessing,
                    ),
                    paths,
                )
            )

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_path = self._resolve_image_path(index)
        if self.training:
            with Image.open(image_path) as image:
                draft_size = math.ceil(
                    self.image_size / min(self.center_crop_scales)
                )
                image.draft("RGB", (draft_size, draft_size))
                image = image.convert("RGB")
                if self.transform is None:  # pragma: no cover - invariant
                    raise RuntimeError("Training transform is unavailable")
                tensor = self.transform(image)
        else:
            tensor = (
                normalize_deployment_rgb(self._deployment_cache[index])
                if self._deployment_cache is not None
                else deployment_preprocess(
                    image_path,
                    self.image_size,
                    preprocessing=self.preprocessing,
                )
            )
        target = torch.tensor(float(row["target"]), dtype=torch.float32)
        return tensor, target, str(row["image_name"])


class DistillationDataset(LesionDataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        images_dir: str | Path,
        image_size: int,
        training: bool,
        teacher_scores: np.ndarray,
        preprocessing: str = DEPLOYMENT_PREPROCESSING_VERSION,
        center_crop_scales: Sequence[float] = (1.0,),
    ):
        super().__init__(
            frame,
            images_dir,
            image_size,
            training,
            preprocessing=preprocessing,
            center_crop_scales=center_crop_scales,
        )
        scores = np.asarray(teacher_scores, dtype=np.float32)
        if scores.ndim != 1 or len(scores) != len(self.frame):
            raise ValueError("Teacher scores must align with the dataset frame")
        if not np.isfinite(scores).all():
            raise ValueError("Teacher scores must be finite")
        self.teacher_scores = scores

    def __getitem__(self, index: int):
        image, target, image_name = super().__getitem__(index)
        teacher_score = torch.tensor(
            self.teacher_scores[index],
            dtype=torch.float32,
        )
        return image, target, teacher_score, image_name
