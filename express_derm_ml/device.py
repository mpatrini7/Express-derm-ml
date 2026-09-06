from __future__ import annotations

import torch

TRAINING_DEVICES = {"auto", "cpu", "cuda", "mps"}


def resolve_training_device(preference: str = "auto") -> torch.device:
    if preference not in TRAINING_DEVICES:
        raise ValueError(f"Unsupported training device: {preference}")
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if preference == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple Metal (MPS) was requested but is not available")
    return torch.device(preference)


def uses_cuda_transfer_optimizations(device: torch.device) -> bool:
    return device.type == "cuda"
