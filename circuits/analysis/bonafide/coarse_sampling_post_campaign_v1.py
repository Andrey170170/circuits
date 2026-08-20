"""Frozen post-campaign analysis for coarse trace-target sampling metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256

ANALYSIS_SCHEMA = "adag.process-witness.coarse-post-campaign-analysis.v1"
REPORT_SCHEMA = "adag.process-witness.coarse-post-campaign-report.v1"
INVENTORY_SCHEMA = "adag.process-witness.coarse-post-campaign-inventory.v1"
ANALYSIS_STATUS = "frozen_sampling_metadata_not_truth"


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_self_hash(value: dict[str, Any], field: str, label: str) -> None:
    payload = dict(value)
    observed = payload.pop(field, None)
    if observed != canonical_sha256(payload):
        raise ValueError(f"{label} self-hash drift")


def _validate_readonly_modes(root: Path) -> None:
    if root.stat().st_mode & 0o777 != 0o555:
        raise ValueError("analysis root mode drift")
    for path in root.rglob("*"):
        expected = 0o555 if path.is_dir() else 0o444
        if path.stat().st_mode & 0o777 != expected:
            raise ValueError(f"analysis mode drift: {path.relative_to(root)}")


def load_frozen_post_campaign_analysis(root: Path) -> dict[str, Any]:
    """Validate a frozen analysis using only evidence beneath ``root``."""

    _validate_readonly_modes(root)
    manifest = _load_object(root / "manifest.json")
    _verify_self_hash(manifest, "manifest_sha256", "analysis manifest")
    if (
        manifest.get("schema_version") != ANALYSIS_SCHEMA
        or manifest.get("status") != ANALYSIS_STATUS
    ):
        raise ValueError("analysis manifest semantic drift")
    inventory = _load_object(root / "evidence-inventory.json")
    _verify_self_hash(inventory, "inventory_sha256", "analysis inventory")
    if inventory.get("schema_version") != INVENTORY_SCHEMA or inventory.get(
        "inventory_sha256"
    ) != manifest.get("inventory_sha256"):
        raise ValueError("analysis inventory binding drift")
    files = inventory.get("files")
    if not isinstance(files, list):
        raise ValueError("analysis inventory files absent")
    expected = {str(row["path"]): row for row in files}
    if len(expected) != len(files):
        raise ValueError("analysis inventory path collision")
    excluded = {root / "manifest.json", root / "evidence-inventory.json"}
    observed = {
        str(path.relative_to(root)): path
        for path in root.rglob("*")
        if path.is_file() and path not in excluded
    }
    if set(observed) != set(expected):
        raise ValueError("analysis inventory coverage drift")
    for relative, path in observed.items():
        row = expected[relative]
        if path.is_symlink():
            raise ValueError(f"analysis artifact contains symlink: {relative}")
        if path.stat().st_size != row.get("bytes") or file_sha256(path) != row.get(
            "sha256"
        ):
            raise ValueError(f"analysis evidence file drift: {relative}")
    report = _load_object(root / "completion-report.json")
    if report.get("schema_version") != REPORT_SCHEMA or file_sha256(
        root / "completion-report.json"
    ) != manifest.get("completion_report_sha256"):
        raise ValueError("analysis completion report drift")
    return {"manifest": manifest, "completion_report": report}
