"""Canonical JSON and atomic sidecar persistence helpers."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping


def canonical_json(value: object) -> bytes:
    """Serialize JSON deterministically for content identities."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unreadable JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_hashed_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    hash_field: str,
) -> None:
    """Write a self-hashed JSON object atomically without overwriting."""

    if path.exists():
        raise FileExistsError(f"destination already exists: {path}")
    payload = dict(value)
    recorded_hash = payload.pop(hash_field, None)
    expected_hash = canonical_sha256(payload)
    if recorded_hash != expected_hash:
        raise ValueError(
            f"{hash_field} mismatch: recorded={recorded_hash!r}, "
            f"expected={expected_hash!r}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
