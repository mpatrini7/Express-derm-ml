from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    MobileNet_V3_Large_Weights,
    efficientnet_b0,
    efficientnet_b2,
    mobilenet_v3_large,
)


def create_model(architecture: str, pretrained: bool) -> nn.Module:
    if architecture == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(input_features, 1)
        return model

    if architecture == "efficientnet_b2":
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = efficientnet_b2(weights=weights)
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(input_features, 1)
        return model

    if architecture == "mobilenet_v3_large":
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_large(weights=weights)
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(input_features, 1)
        return model

    raise ValueError(f"Unsupported architecture: {architecture}")


class LogitWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image).flatten(1)
