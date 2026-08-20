from __future__ import annotations

import ctypes
import errno
import json
from pathlib import Path

import pytest
from circuits.analysis.bonafide import coarse_sampling_post_campaign_v1 as post_campaign
from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_post_campaign_v1 import (
    load_frozen_post_campaign_analysis,
)


def _write_minimal_artifact(root: Path) -> dict[str, object]:
    root.mkdir()
    (root / "source-evidence/bundle").mkdir(parents=True)
    report = {
        "schema_version": "adag.process-witness.coarse-post-campaign-report.v1",
        "claim_boundary": "sampling metadata only",
        "census": {
            "physical_requests": 37_671,
            "effective_success": 37_656,
            "residual_invalid_output": 15,
            "responses": 188,
            "units": 94_546,
            "openai_pending_units": 74_860,
            "deterministic_surface_units": 19_500,
            "deterministic_terminal_units": 186,
        },
        "strict_proposals": {
            "provider_vote_coverage": {"0": 6, "1": 6, "2": 60, "3": 74_788},
            "broad_counts": {
                "contextual": 55_966,
                "process_bearing": 38_366,
                "unresolved": 142,
                "missing_proposal": 72,
            },
            "fine_agreement": {
                "unanimous": 59_812,
                "two_one": 14_300,
                "one_one_one": 676,
            },
            "broad_agreement": {
                "unanimous": 67_113,
                "two_one": 7_647,
                "one_one_one": 28,
            },
        },
        "conservative_exact_id_salvage": {
            "provider_vote_coverage": {"0": 1, "1": 1, "2": 10, "3": 74_848},
            "broad_counts": {
                "contextual": 55_994,
                "process_bearing": 38_398,
                "unresolved": 142,
                "missing_proposal": 12,
            },
            "fine_agreement": {
                "unanimous": 59_864,
                "two_one": 14_308,
                "one_one_one": 676,
            },
        },
    }
    (root / "completion-report.json").write_text(json.dumps(report) + "\n")
    units = [
        {
            "unit_id": "u0",
            "response_id": "r0",
            "sequence_index": 0,
            "fragment_of": "f0",
        },
        {
            "unit_id": "u1",
            "response_id": "r0",
            "sequence_index": 1,
            "fragment_of": "f0",
        },
    ]
    (root / "source-evidence/bundle/units.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in units)
    )
    psu = {
        "psu_id": "p0",
        "response_id": "r0",
        "response_psu_index": 0,
        "member_unit_ids": ["u0", "u1"],
        "fragment_of": "f0",
        "incomplete_hard_barrier": False,
        "correlation_run_id": "r0:run-0",
    }
    (root / "sampling-psus.jsonl").write_text(json.dumps(psu) + "\n")
    candidates = [
        {
            "psu_id": "p0",
            "unit_id": unit_id,
            "designs": [
                {
                    "policy": "balanced",
                    "nominal_expected_budget": budget,
                    "group_inclusion_probability": pi,
                    "atom_conditional_probability": 0.5,
                    "each_position_conditional_probability": 1.0,
                    "each_position_marginal_inclusion_probability": pi * 0.5,
                    "each_position_inverse_probability_weight": 1.0 / (pi * 0.5),
                    "selected_or_frozen_trace_policy": False,
                    "exact_integer_sample_selected": False,
                }
                for budget, pi in ((30_000, 0.3), (35_000, 0.35), (40_000, 0.4))
            ],
        }
        for unit_id in ("u0", "u1")
    ]
    (root / "candidate-inclusion-probabilities.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in candidates)
    )
    files = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
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
        "completion_report_sha256": file_sha256(root / "completion-report.json"),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n")
    for path in root.rglob("*"):
        path.chmod(0o444 if path.is_file() else 0o555)
    root.chmod(0o555)
    return manifest


def _rehash_artifact(root: Path) -> None:
    inventory_path = root / "evidence-inventory.json"
    inventory = json.loads(inventory_path.read_text())
    inventory["files"] = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path not in {root / "manifest.json", inventory_path}
    ]
    inventory.pop("inventory_sha256")
    inventory["inventory_sha256"] = canonical_sha256(inventory)
    inventory_path.write_text(json.dumps(inventory) + "\n")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["inventory_sha256"] = inventory["inventory_sha256"]
    manifest["completion_report_sha256"] = file_sha256(root / "completion-report.json")
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest) + "\n")


def test_frozen_loader_validates_without_source_roots(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    manifest = _write_minimal_artifact(root)

    loaded = load_frozen_post_campaign_analysis(root)

    assert loaded["manifest"] == manifest
    assert loaded["completion_report"]["census"]["physical_requests"] == 37_671


def test_frozen_loader_streams_large_candidate_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifact"
    _write_minimal_artifact(root)
    original = post_campaign.read_jsonl

    def reject_bulk_candidate_read(path: Path) -> list[dict[str, object]]:
        if path.name == "candidate-inclusion-probabilities.jsonl":
            raise AssertionError("candidate table must be validated as a stream")
        return original(path)

    monkeypatch.setattr(post_campaign, "read_jsonl", reject_bulk_candidate_read)

    load_frozen_post_campaign_analysis(root)


def test_publish_falls_back_without_replacing_existing_nested_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class UnsupportedRename:
        def __call__(self, *_args: object) -> int:
            ctypes.set_errno(errno.EINVAL)
            return -1

    class FakeLibc:
        renameat2 = UnsupportedRename()

    monkeypatch.setattr(
        post_campaign.ctypes, "CDLL", lambda *_args, **_kwargs: FakeLibc()
    )
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "nested/evidence.json").write_text("evidence\n")
    (source / "manifest.json").write_text("manifest\n")
    destination = tmp_path / "published"

    post_campaign._publish_no_replace(source, destination)

    assert not source.exists()
    assert (destination / "nested/evidence.json").read_text() == "evidence\n"
    assert (destination / "manifest.json").read_text() == "manifest\n"
    second = tmp_path / "second"
    second.mkdir()
    (second / "manifest.json").write_text("replacement\n")
    with pytest.raises(FileExistsError, match="destination exists"):
        post_campaign._publish_no_replace(second, destination)
    assert (destination / "manifest.json").read_text() == "manifest\n"


def test_publish_fallback_removes_interrupted_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class UnsupportedRename:
        def __call__(self, *_args: object) -> int:
            ctypes.set_errno(errno.EINVAL)
            return -1

    class FakeLibc:
        renameat2 = UnsupportedRename()

    monkeypatch.setattr(
        post_campaign.ctypes, "CDLL", lambda *_args, **_kwargs: FakeLibc()
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "evidence.json").write_text("evidence\n")
    (source / "manifest.json").write_text("manifest\n")
    destination = tmp_path / "published"
    original_rename = post_campaign.os.rename
    move_count = 0

    def interrupt_second_move(source_path: Path, destination_path: Path) -> None:
        nonlocal move_count
        move_count += 1
        if move_count == 2:
            raise OSError("injected publication interruption")
        original_rename(source_path, destination_path)

    monkeypatch.setattr(post_campaign.os, "rename", interrupt_second_move)

    with pytest.raises(OSError, match="injected publication interruption"):
        post_campaign._publish_no_replace(source, destination)

    assert not destination.exists()


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


def test_frozen_loader_rejects_rehashed_literal_census_drift(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _write_minimal_artifact(root)
    for path in root.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    root.chmod(0o755)
    report_path = root / "completion-report.json"
    report = json.loads(report_path.read_text())
    report["strict_proposals"]["provider_vote_coverage"]["3"] = 74_787
    report_path.write_text(json.dumps(report) + "\n")
    _rehash_artifact(root)
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)

    with pytest.raises(ValueError, match="literal census drift"):
        load_frozen_post_campaign_analysis(root)


def test_frozen_loader_rejects_split_fragment_psu(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _write_minimal_artifact(root)
    for path in root.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    root.chmod(0o755)
    (root / "sampling-psus.jsonl").write_text(
        json.dumps(
            {
                "psu_id": "p0",
                "response_id": "r0",
                "response_psu_index": 0,
                "member_unit_ids": ["u0"],
                "fragment_of": "f0",
                "incomplete_hard_barrier": False,
                "correlation_run_id": "r0:run-0",
            }
        )
        + "\n"
        + json.dumps(
            {
                "psu_id": "p1",
                "response_id": "r0",
                "response_psu_index": 1,
                "member_unit_ids": ["u1"],
                "fragment_of": "f0",
                "incomplete_hard_barrier": False,
                "correlation_run_id": "r0:run-0",
            }
        )
        + "\n"
    )
    _rehash_artifact(root)
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)

    with pytest.raises(ValueError, match="fragment PSU partition drift"):
        load_frozen_post_campaign_analysis(root)


def test_frozen_loader_rejects_non_nested_or_nonpositive_candidate_pi(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _write_minimal_artifact(root)
    for path in root.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    root.chmod(0o755)
    path = root / "candidate-inclusion-probabilities.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["designs"][1]["group_inclusion_probability"] = 0.2
    rows[0]["designs"][1]["each_position_marginal_inclusion_probability"] = 0.1
    rows[0]["designs"][1]["each_position_inverse_probability_weight"] = 10.0
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _rehash_artifact(root)
    for child in root.rglob("*"):
        child.chmod(0o555 if child.is_dir() else 0o444)
    root.chmod(0o555)

    with pytest.raises(ValueError, match="candidate nesting drift"):
        load_frozen_post_campaign_analysis(root)
