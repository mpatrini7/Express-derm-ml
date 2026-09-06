from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from express_derm_ml.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    LETTERBOX_FILL_RGB,
    LETTERBOX_PREPROCESSING_VERSION,
    DistillationDataset,
    LesionDataset,
    deployment_preprocess,
    deployment_resize_rgb,
    _RandomCenterScaleCrop,
    validate_center_crop_scales,
)


def test_evaluation_dataset_matches_opencv_deployment_contract(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_path = images_dir / "sample.png"
    pixels = np.zeros((5, 7, 3), dtype=np.uint8)
    pixels[..., 0] = np.arange(7, dtype=np.uint8)
    pixels[..., 1] = np.arange(5, dtype=np.uint8)[:, None] * 10
    pixels[..., 2] = 200
    Image.fromarray(pixels).save(image_path)
    frame = pd.DataFrame(
        [
            {
                "image_path": image_path.name,
                "image_name": "sample",
                "target": 0,
            }
        ]
    )
    dataset = LesionDataset(
        frame,
        images_dir,
        image_size=4,
        training=False,
    )
    tensor, target, image_name = dataset[0]

    import cv2

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (4, 4), interpolation=cv2.INTER_AREA)
    expected = resized.astype(np.float32) / np.float32(255.0)
    expected = (expected - IMAGENET_MEAN) / IMAGENET_STD
    expected = torch.from_numpy(
        np.ascontiguousarray(np.transpose(expected, (2, 0, 1)))
    )

    assert tensor.shape == (3, 4, 4)
    assert torch.equal(tensor, expected)
    assert float(target) == 0.0
    assert image_name == "sample"

    uncached = tensor.clone()
    dataset.cache_evaluation_images(workers=1)
    cached, _, _ = dataset[0]
    assert torch.equal(cached, uncached)


def test_deployment_preprocess_rejects_undecodable_image(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "not-an-image.jpg"
    image_path.write_text("invalid", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Unable to decode"):
        deployment_preprocess(image_path, 16)


def test_letterbox_preprocessing_preserves_geometry_and_padding(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "wide.png"
    pixels = np.zeros((4, 8, 3), dtype=np.uint8)
    pixels[...] = (210, 30, 60)
    Image.fromarray(pixels).save(image_path)

    resized = deployment_resize_rgb(
        image_path,
        8,
        preprocessing=LETTERBOX_PREPROCESSING_VERSION,
    )

    assert resized.shape == (8, 8, 3)
    assert np.all(resized[2:6] == np.asarray((210, 30, 60)))
    assert np.all(resized[:2] == np.asarray(LETTERBOX_FILL_RGB))
    assert np.all(resized[6:] == np.asarray(LETTERBOX_FILL_RGB))


def test_letterbox_dataset_cache_matches_uncached_pixels(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (31, 17), color=(10, 80, 170)).save(
        images_dir / "sample.png"
    )
    frame = pd.DataFrame(
        [{"image_path": "sample.png", "image_name": "sample", "target": 0}]
    )
    dataset = LesionDataset(
        frame,
        images_dir,
        image_size=32,
        training=False,
        preprocessing=LETTERBOX_PREPROCESSING_VERSION,
    )

    uncached, _, _ = dataset[0]
    dataset.cache_evaluation_images(workers=1)
    cached, _, _ = dataset[0]

    assert uncached.shape == (3, 32, 32)
    assert torch.equal(cached, uncached)


def test_distillation_dataset_aligns_finite_teacher_scores(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    Image.new("RGB", (32, 32), color=(30, 80, 120)).save(
        images_dir / "sample.jpg"
    )
    frame = pd.DataFrame(
        [{"image_path": "sample.jpg", "image_name": "sample", "target": 1}]
    )
    dataset = DistillationDataset(
        frame,
        images_dir,
        image_size=16,
        training=False,
        teacher_scores=np.array([0.75], dtype=np.float32),
    )

    _, target, teacher_score, image_name = dataset[0]

    assert float(target) == 1.0
    assert float(teacher_score) == 0.75
    assert image_name == "sample"

    with pytest.raises(ValueError, match="align"):
        DistillationDataset(
            frame,
            images_dir,
            image_size=16,
            training=False,
            teacher_scores=np.array([], dtype=np.float32),
        )


def test_training_center_crop_preserves_center_and_declares_full_view() -> None:
    image = Image.new("RGB", (20, 10), color=(0, 0, 255))
    for x in range(5, 15):
        for y in range(2, 8):
            image.putpixel((x, y), (255, 0, 0))
    crop = _RandomCenterScaleCrop((1.0, 0.5))
    crop.scales = (0.5,)

    result = crop(image)

    assert result.size == (10, 5)
    assert result.getpixel((5, 2)) == (255, 0, 0)
    assert validate_center_crop_scales((1.0, 0.9, 0.8)) == (
        1.0,
        0.9,
        0.8,
    )
    with pytest.raises(ValueError, match="include 1.0"):
        validate_center_crop_scales((0.9, 0.8))


def test_dataset_allows_confined_top_level_data_symlink(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    Image.new("RGB", (16, 16), color=(20, 40, 60)).save(
        external / "sample.jpg"
    )
    images_dir = tmp_path / "composite"
    images_dir.mkdir()
    (images_dir / "source").symlink_to(external, target_is_directory=True)
    frame = pd.DataFrame(
        [
            {
                "image_path": "source/sample.jpg",
                "image_name": "sample",
                "target": 0,
            }
        ]
    )

    dataset = LesionDataset(
        frame,
        images_dir,
        image_size=8,
        training=False,
    )

    tensor, _, _ = dataset[0]
    assert tensor.shape == (3, 8, 8)
    frame.loc[0, "image_path"] = "source/../outside.jpg"
    unsafe = LesionDataset(frame, images_dir, image_size=8, training=False)
    with pytest.raises(ValueError, match="safe and relative"):
        unsafe[0]
