from __future__ import annotations

from pathlib import Path, PureWindowsPath


def resolve_manifest_image_path(
    images_root: str | Path,
    raw_path: str,
) -> Path:
    root = Path(images_root).resolve()
    relative = Path(raw_path)
    if (
        not raw_path.strip()
        or relative.is_absolute()
        or PureWindowsPath(raw_path).is_absolute()
        or ".." in relative.parts
    ):
        raise ValueError("Manifest image paths must be safe and relative")

    top_level = root / relative.parts[0]
    resolved = (root / relative).resolve()
    allowed_root = top_level.resolve() if top_level.is_symlink() else root
    try:
        resolved.relative_to(allowed_root)
    except ValueError as error:
        raise ValueError(
            "Manifest image path escapes its configured data source"
        ) from error
    return resolved
