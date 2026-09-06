from __future__ import annotations

import hashlib
from collections.abc import Iterable


def canonical_id_set_sha256(values: Iterable[str]) -> str:
    normalized = sorted({str(value).strip() for value in values})
    if not normalized or any(not value for value in normalized):
        raise ValueError("Canonical ID sets cannot be empty or blank")
    payload = "".join(f"{len(value)}:{value}\n" for value in normalized)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deterministic_group_order(group_id: str, seed: int) -> str:
    payload = f"express-derm-oof-v1\0{int(seed)}\0{group_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
