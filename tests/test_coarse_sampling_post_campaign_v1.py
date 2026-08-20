from __future__ import annotations

import json
from pathlib import Path

import pytest
from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_post_campaign_v1 import (
    load_frozen_post_campaign_analysis,
)


def _write_minimal_artifact(root: Path) -> dict[str, object]:
    root.mkdir()
    report = {
        "schema_version": "adag.process-witness.coarse-post-campaign-report.v1",
        "claim_boundary": "sampling metadata only",
        "census": {
            "physical_requests": 37_671,
            "effective_success": 37_656,
            "residual_invalid_output": 15,
            "units": 94_546,
        },
    }
    (root / "completion-report.json").write_text(json.dumps(report) + "\n")
    files = [
        {
            "path": "completion-report.json",
            "bytes": (root / "completion-report.json").stat().st_size,
            "sha256": file_sha256(root / "completion-report.json"),
        }
    ]
    inventory = {
        "schema_version": "adag.process-witness.coarse-post-campaign-inventory.v1",
        "files": files,
    }
    inventory["inventory_sha256"] = canonical_sha256(inventory)
    (root / "evidence-inventory.json").write_text(json.dumps(inventory) + "\n")
    manifest = {
        "schema_version": "adag.process-witness.coarse-post-campaign-analysis.v1",
        "status": "frozen_sampling_metadata_not_truth",
        "inventory_sha256": inventory["inventory_sha256"],
        "completion_report_sha256": files[0]["sha256"],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n")
    for path in root.rglob("*"):
        path.chmod(0o444 if path.is_file() else 0o555)
    root.chmod(0o555)
    return manifest


def test_frozen_loader_validates_without_source_roots(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    manifest = _write_minimal_artifact(root)

    loaded = load_frozen_post_campaign_analysis(root)

    assert loaded["manifest"] == manifest
    assert loaded["completion_report"]["census"]["physical_requests"] == 37_671


def test_frozen_loader_rejects_tamper_and_mode_drift(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _write_minimal_artifact(root)
    report = root / "completion-report.json"
    report.chmod(0o644)
    with pytest.raises(ValueError, match="mode drift"):
        load_frozen_post_campaign_analysis(root)
    report.chmod(0o444)
    root.chmod(0o755)
    with pytest.raises(ValueError, match="mode drift"):
        load_frozen_post_campaign_analysis(root)
