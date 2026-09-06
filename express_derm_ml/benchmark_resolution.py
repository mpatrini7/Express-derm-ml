from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from .artifacts import write_json_exclusive
from .device import resolve_training_device
from .model import create_model


def parse_sizes(value: str) -> list[int]:
    try:
        sizes = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Sizes must be comma-separated integers"
        ) from error
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("Every image size must be positive")
    if len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("Image sizes must be unique")
    return sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure forward and training-step cost at multiple square input "
            "resolutions before committing to a long model run."
        )
    )
    parser.add_argument("--architecture", default="efficientnet_b0")
    parser.add_argument("--sizes", type=parse_sizes, default=[224, 512, 1024])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Skip backward and optimizer steps.",
    )
    parser.add_argument("--output")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_snapshot(device: torch.device) -> dict[str, int]:
    if device.type == "cuda":
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
        }
    if device.type == "mps":
        return {
            "allocated_bytes": int(torch.mps.current_allocated_memory()),
            "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        }
    return {}


def timed_inference(
    model: nn.Module,
    images: torch.Tensor,
    *,
    device: torch.device,
    warmup_steps: int,
    steps: int,
) -> list[float]:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_steps):
            model(images)
        synchronize(device)
        durations = []
        for _ in range(steps):
            started = time.perf_counter()
            model(images)
            synchronize(device)
            durations.append(time.perf_counter() - started)
    return durations


def timed_training(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device,
    warmup_steps: int,
    steps: int,
) -> list[float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        logits = model(images).flatten()
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

    for _ in range(warmup_steps):
        step()
    synchronize(device)
    durations = []
    for _ in range(steps):
        started = time.perf_counter()
        step()
        synchronize(device)
        durations.append(time.perf_counter() - started)
    return durations


def summarize(durations: list[float], batch_size: int) -> dict[str, float]:
    milliseconds = [duration * 1000.0 for duration in durations]
    mean_ms = statistics.fmean(milliseconds)
    return {
        "mean_step_ms": mean_ms,
        "minimum_step_ms": min(milliseconds),
        "maximum_step_ms": max(milliseconds),
        "mean_per_image_ms": mean_ms / batch_size,
        "images_per_second": batch_size * 1000.0 / mean_ms,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if args.warmup_steps < 0 or args.steps <= 0:
        raise ValueError("Warmup must be non-negative and steps must be positive")
    device = resolve_training_device(args.device)
    results = []
    for image_size in args.sizes:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model = create_model(args.architecture, pretrained=False).to(device)
        images = torch.randn(
            args.batch_size,
            3,
            image_size,
            image_size,
            device=device,
        )
        targets = torch.zeros(args.batch_size, device=device)
        inference = timed_inference(
            model,
            images,
            device=device,
            warmup_steps=args.warmup_steps,
            steps=args.steps,
        )
        training = None
        if not args.inference_only:
            training = timed_training(
                model,
                images,
                targets,
                device=device,
                warmup_steps=args.warmup_steps,
                steps=args.steps,
            )
        results.append(
            {
                "image_size": image_size,
                "input_pixels": image_size * image_size,
                "pixel_multiple_vs_224": (image_size / 224.0) ** 2,
                "inference": summarize(inference, args.batch_size),
                "training": (
                    None
                    if training is None
                    else summarize(training, args.batch_size)
                ),
                "memory": memory_snapshot(device),
            }
        )
        del targets, images, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    report = {
        "schema_version": 1,
        "architecture": args.architecture,
        "device": str(device),
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.steps,
        "inference_only": bool(args.inference_only),
        "torch_version": str(torch.__version__),
        "platform": platform.platform(),
        "results": results,
        "warning": (
            "Synthetic throughput estimate only; full training also includes "
            "image decoding, augmentation, validation, and artifact checks."
        ),
    }
    if args.output:
        write_json_exclusive(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
