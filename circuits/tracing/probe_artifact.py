"""Atomic JSON persistence for graph-free ADAG target probes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from circuits.tracing.trace import (
    TeacherForcedProbeResult,
    validate_teacher_forced_probe_result,
)

SCHEMA_VERSION = "adag.probe-artifact.v1"
PROBE_FILENAME = "probe.json"
MANIFEST_FILENAME = "manifest.json"
METRICS_FILENAME = "metrics.json"


@dataclass(frozen=True)
class ProbeArtifact:
    path: Path
    probe: dict[str, object]
    manifest: dict[str, object]
    metrics: dict[str, object]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_probe_artifact(
    path: str | os.PathLike[str],
    probe: TeacherForcedProbeResult | Mapping[str, object],
    *,
    metrics: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically create a self-validating JSON probe directory."""

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"probe artifact destination already exists: {target}")
    canonical_probe = validate_teacher_forced_probe_result(probe)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        probe_path = temporary / PROBE_FILENAME
        metrics_path = temporary / METRICS_FILENAME
        _write_json(probe_path, canonical_probe)
        _write_json(metrics_path, dict(metrics or {}))
        canonical_manifest: dict[str, Any] = dict(manifest or {})
        canonical_manifest.update(
            {
                "schema_version": SCHEMA_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "probe_file": PROBE_FILENAME,
                "probe_schema_version": canonical_probe["schema_version"],
                "probe_sha256": _sha256_file(probe_path),
                "probe_size_bytes": probe_path.stat().st_size,
                "metrics_file": METRICS_FILENAME,
                "metrics_sha256": _sha256_file(metrics_path),
                "metrics_size_bytes": metrics_path.stat().st_size,
                "target_provenance": canonical_probe["target_provenance"],
                "occurrence_signature": canonical_probe["occurrence_signature"],
                "feature_basis_signature": canonical_probe[
                    "feature_basis_signature"
                ],
                "selected_occurrence_count": len(
                    canonical_probe["selected_occurrences"]
                ),
            }
        )
        _write_json(temporary / MANIFEST_FILENAME, canonical_manifest)
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def validate_probe_artifact_integrity(
    path: str | os.PathLike[str],
) -> dict[str, object]:
    """Validate JSON schemas and checksum without loading any pickle payload."""

    artifact_path = Path(path)
    if not artifact_path.is_dir():
        raise ValueError(f"probe artifact path is not a directory: {artifact_path}")
    manifest_path = artifact_path / MANIFEST_FILENAME
    metrics_path = artifact_path / METRICS_FILENAME
    probe_path = artifact_path / PROBE_FILENAME
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        with metrics_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        with probe_path.open(encoding="utf-8") as handle:
            probe = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"probe artifact is missing or unreadable: {artifact_path}") from error
    if not isinstance(manifest, dict) or not isinstance(metrics, dict):
        raise ValueError("probe artifact manifest and metrics must be JSON objects")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported probe artifact schema: {manifest.get('schema_version')!r}"
        )
    if manifest.get("probe_file") != PROBE_FILENAME:
        raise ValueError(f"probe artifact probe_file must be {PROBE_FILENAME!r}")
    expected_size = manifest.get("probe_size_bytes")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int):
        raise ValueError("probe artifact probe_size_bytes is invalid")
    if probe_path.stat().st_size != expected_size:
        raise ValueError("probe artifact data size mismatch")
    expected_hash = manifest.get("probe_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("probe artifact probe_sha256 is invalid")
    if _sha256_file(probe_path) != expected_hash:
        raise ValueError("probe artifact checksum mismatch")
    if manifest.get("metrics_file") != METRICS_FILENAME:
        raise ValueError(f"probe artifact metrics_file must be {METRICS_FILENAME!r}")
    expected_metrics_size = manifest.get("metrics_size_bytes")
    if (
        isinstance(expected_metrics_size, bool)
        or not isinstance(expected_metrics_size, int)
    ):
        raise ValueError("probe artifact metrics_size_bytes is invalid")
    if metrics_path.stat().st_size != expected_metrics_size:
        raise ValueError("probe artifact metrics size mismatch")
    expected_metrics_hash = manifest.get("metrics_sha256")
    if not isinstance(expected_metrics_hash, str) or len(expected_metrics_hash) != 64:
        raise ValueError("probe artifact metrics_sha256 is invalid")
    if _sha256_file(metrics_path) != expected_metrics_hash:
        raise ValueError("probe artifact metrics checksum mismatch")
    canonical_probe = validate_teacher_forced_probe_result(probe)
    if manifest.get("occurrence_signature") != canonical_probe[
        "occurrence_signature"
    ]:
        raise ValueError("probe artifact occurrence signature disagrees with payload")
    if manifest.get("feature_basis_signature") != canonical_probe[
        "feature_basis_signature"
    ]:
        raise ValueError("probe artifact feature basis disagrees with payload")
    if manifest.get("target_provenance") != canonical_probe["target_provenance"]:
        raise ValueError("probe artifact target provenance disagrees with payload")
    if manifest.get("selected_occurrence_count") != len(
        canonical_probe["selected_occurrences"]
    ):
        raise ValueError("probe artifact selected occurrence count mismatch")
    return manifest


def load_probe_artifact(path: str | os.PathLike[str]) -> ProbeArtifact:
    artifact_path = Path(path)
    manifest = validate_probe_artifact_integrity(artifact_path)
    with (artifact_path / PROBE_FILENAME).open(encoding="utf-8") as handle:
        probe = json.load(handle)
    with (artifact_path / METRICS_FILENAME).open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    return ProbeArtifact(
        path=artifact_path,
        probe=probe,
        manifest=manifest,
        metrics=metrics,
    )
