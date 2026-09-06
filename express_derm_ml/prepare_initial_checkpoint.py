from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision

from .artifacts import create_new_directory, write_json_exclusive
from .common import sha256_file
from .model import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an immutable ImageNet-initialized student checkpoint."
    )
    parser.add_argument("--architecture", choices=("efficientnet_b2",), required=True)
    parser.add_argument("--image-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image_size <= 0:
        raise ValueError("Image size must be positive")
    output_dir = create_new_directory(args.output_dir)
    torch.manual_seed(args.seed)
    model = create_model(args.architecture, pretrained=True)
    checkpoint_path = output_dir / "best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": args.architecture,
            "image_size": args.image_size,
            "initialization": "torchvision_imagenet1k_default",
        },
        checkpoint_path,
    )
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "architecture": args.architecture,
        "image_size": args.image_size,
        "initialization": "torchvision_imagenet1k_default",
        "seed": args.seed,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "research_only": True,
    }
    write_json_exclusive(output_dir / "initialization.json", receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
