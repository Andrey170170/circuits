from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from circuits.analysis.bonafide.canonical import canonical_sha256
from circuits.analysis.bonafide.coarse_sampling_commentary_audit_v1 import (
    ARTIFACT_BINDING_PLACEHOLDER,
    EXPORT_SCHEMA,
    _runtime_source_paths,
    _validate_item_semantics,
    build_commentary_audit_packet,
    load_commentary_audit_packet,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _proposal(unit_id: str, *, complete: bool = True) -> dict:
    votes = (
        [
            {
                "tag": "active_task_work",
                "confidence": "high",
                "boundary_concerns": ["merge_next"],
                "boundary_note": "Continues in the next unit.",
                "replica_index": index,
                "request_id": f"request-{unit_id}-{index}",
                "unit_id": unit_id,
                "vote_origin": "provider_schema_valid",
            }
            for index in range(3)
        ]
        if complete
        else []
    )
    return {
        "unit_id": unit_id,
        "response_id": "response-1",
        "proposal_status": "complete" if complete else "insufficient_exact_votes",
        "source": "openai_replica_votes",
        "assignment_route": "openai_pending",
        "fine_votes": [vote["tag"] for vote in votes],
        "fine_vote_histogram": {"active_task_work": 3} if complete else {},
        "fine_agreement_pattern": "unanimous" if complete else "not_available",
        "broad_votes": ["process_bearing"] * len(votes),
        "broad_vote_histogram": {"process_bearing": 3} if complete else {},
        "broad_majority": "process_bearing" if complete else "missing_proposal",
        "broad_agreement_pattern": "unanimous" if complete else "not_available",
        "physical_votes": votes,
        "replica_coverage": len(votes),
        "missing_replica_indices": [] if complete else [0, 1, 2],
        "fragment_of": "fragment-1",
        "sequence_index": 0 if unit_id == "unit-1" else 1,
    }


def _source_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    analysis = tmp_path / "analysis"
    sampling = tmp_path / "sampling"
    analysis.mkdir()
    sampling.mkdir()
    (analysis / "source-evidence").mkdir()
    (analysis / "audit-plan.json").write_text("{}\n")
    (analysis / "source-evidence/response-contexts.jsonl").write_text("{}\n")
    (sampling / "audit-supplement-plan.json").write_text("{}\n")
    blind = [
        {
            "schema_version": "adag.process-witness.coarse-blind-audit-item.v1",
            "audit_id": "audit-1",
            "response_id": "response-1",
            "task_prompt": "Compute carefully.",
            "full_response": "First atom. Second atom.",
            "targets": [
                {
                    "unit_id": "unit-1",
                    "text": "First atom.",
                    "token_span": [0, 3],
                    "core_character_span": [0, 11],
                    "covering_character_span": [0, 12],
                },
                {
                    "unit_id": "unit-2",
                    "text": "Second atom.",
                    "token_span": [3, 6],
                    "core_character_span": [12, 24],
                    "covering_character_span": [12, 24],
                },
            ],
        }
    ]
    reveal = [
        {
            "schema_version": "adag.process-witness.coarse-audit-reveal-item.v1",
            "audit_id": "audit-1",
            "psu_id": "psu-1",
            "route": "provider_process",
            "strata": ["diagnostic:fragment", "probability_base:provider_process"],
            "probability_base_inclusion_probability": 0.25,
            "proposals": [_proposal("unit-1"), _proposal("unit-2", complete=False)],
        }
    ]
    _write_jsonl(analysis / "blind-audit.jsonl", blind)
    _write_jsonl(analysis / "audit-reveal.jsonl", reveal)
    _write_jsonl(
        sampling / "audit-supplement-pools.jsonl",
        [
            {
                "schema_version": (
                    "adag.process-witness.coarse-audit-supplement-pool.v2"
                ),
                "psu_id": "psu-1",
                "response_id": "response-1",
                "pool_ids": [
                    "long_unit_at_96_token_segmentation_cap",
                    "rare_fine:active_task_work",
                ],
            }
        ],
    )
    monkeypatch.setattr(
        "circuits.analysis.bonafide.coarse_sampling_commentary_audit_v1."
        "load_frozen_post_campaign_analysis",
        lambda root: {"manifest": {"manifest_sha256": "analysis-sha"}},
    )
    monkeypatch.setattr(
        "circuits.analysis.bonafide.coarse_sampling_commentary_audit_v1."
        "load_frozen_post_campaign_sampling_v2",
        lambda root, parent_v1_root: {
            "manifest": {
                "manifest_sha256": "sampling-sha",
                "parent_v1_manifest_sha256": "analysis-sha",
            }
        },
    )
    monkeypatch.setattr(
        "circuits.analysis.bonafide.coarse_sampling_commentary_audit_v1."
        "_execution_source_revision",
        lambda: {
            "repository_commit": "a" * 40,
            "repository_tree": "b" * 40,
            "tracked_source_clean": True,
            "files": [
                {
                    "path": path,
                    "bytes": 1,
                    "sha256": "c" * 64,
                    "git_blob_sha1": "d" * 40,
                }
                for path in (
                    "circuits/analysis/bonafide/coarse_sampling_commentary_audit_v1.py",
                    "scripts/bonafide/"
                    "build_process_witness_coarse_commentary_audit_v1.py",
                )
            ],
        },
    )
    return analysis, sampling


def test_builds_nonblind_per_atom_packet_and_strict_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis, sampling = _source_fixture(tmp_path, monkeypatch)
    destination = tmp_path / "packet"

    manifest = build_commentary_audit_packet(
        analysis_root=analysis,
        sampling_root=sampling,
        destination=destination,
    )
    loaded = load_commentary_audit_packet(destination)

    assert manifest["status"] == "frozen_nonblind_qualitative_commentary_packet"
    assert manifest["counts"] == {
        "audit_draws": 1,
        "complete_proposals": 1,
        "documents": 1,
        "insufficient_proposals": 1,
        "items": 2,
        "multi_atom_draws": 1,
        "physical_votes": 3,
        "single_atom_draws": 0,
        "vote_origins": {"provider_schema_valid": 3},
    }
    assert [item["unit_id"] for item in loaded["items"]] == ["unit-1", "unit-2"]
    first = loaded["items"][0]
    assert first["audit_id"] == "audit-1"
    assert first["psu_id"] == "psu-1"
    assert first["strata"] == [
        "diagnostic:fragment",
        "probability_base:provider_process",
    ]
    assert first["supplement_pool_ids"] == [
        "long_unit_at_96_token_segmentation_cap",
        "rare_fine:active_task_work",
    ]
    assert first["model_proposal"]["fine_majority"] == "active_task_work"
    assert first["model_proposal"]["physical_votes"][0]["boundary_note"]
    assert loaded["packet"]["qualitative_only"] is True
    assert loaded["packet"]["completion_required"] is False
    assert len(loaded["packet"]["artifact_binding_sha256"]) == 64
    assert (
        loaded["packet"]["artifact_binding_sha256"]
        in (destination / "review.html").read_text()
    )
    assert "accuracy" in loaded["packet"]["claim_boundary"].lower()


def test_standalone_loader_rejects_packet_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis, sampling = _source_fixture(tmp_path, monkeypatch)
    destination = tmp_path / "packet"
    build_commentary_audit_packet(
        analysis_root=analysis,
        sampling_root=sampling,
        destination=destination,
    )
    (destination / "items.jsonl").chmod(0o644)
    with (destination / "items.jsonl").open("a") as handle:
        handle.write("{}\n")
    (destination / "items.jsonl").chmod(0o444)

    with pytest.raises(ValueError, match="file drift"):
        load_commentary_audit_packet(destination)


def test_browser_is_self_contained_and_exports_all_items_with_review_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis, sampling = _source_fixture(tmp_path, monkeypatch)
    destination = tmp_path / "packet"
    build_commentary_audit_packet(
        analysis_root=analysis,
        sampling_root=sampling,
        destination=destination,
    )

    html = (destination / "review.html").read_text()
    packet = json.loads((destination / "packet.json").read_text())
    assert "const DATA=" in html
    assert "Array.from(doc.full_response)" in html
    assert "localStorage" in html
    assert "Export commentary JSONL" in html
    assert "reviewed:!!decision.reviewed" in html
    assert EXPORT_SCHEMA in html
    assert "artifact_binding_sha256:packet.artifact_binding_sha256" in html
    assert "captureCurrent();const exportedAt=" in html
    assert "vote.vote_origin" in html
    assert "PSU-level probability-base inclusion" in html
    assert "packet.decision_precedence.join" in html
    assert "composite unit" in html
    assert "global seal" not in html.lower()
    assert "Model proposal" in html
    assert "visible.includes(original)?old+1:old" in html
    assert packet["packet_binding_sha256"] == canonical_sha256(
        {
            key: packet[key]
            for key in (
                "schema_version",
                "source_bindings",
                "ui_version",
                "item_ids_in_order",
                "tag_definitions",
                "decision_precedence",
                "boundary_definitions",
                "dispositions",
                "claim_boundary",
            )
        }
    )


def test_standalone_loader_rejects_mode_and_root_membership_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis, sampling = _source_fixture(tmp_path, monkeypatch)
    destination = tmp_path / "packet"
    build_commentary_audit_packet(
        analysis_root=analysis,
        sampling_root=sampling,
        destination=destination,
    )
    (destination / "review.html").chmod(0o644)
    with pytest.raises(ValueError, match="file drift"):
        load_commentary_audit_packet(destination)
    (destination / "review.html").chmod(0o444)
    destination.chmod(0o755)
    (destination / "extra").mkdir()
    destination.chmod(0o555)
    with pytest.raises(ValueError, match="root membership drift"):
        load_commentary_audit_packet(destination)


def test_rendered_browser_javascript_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable")
    analysis, sampling = _source_fixture(tmp_path, monkeypatch)
    destination = tmp_path / "packet"
    build_commentary_audit_packet(
        analysis_root=analysis,
        sampling_root=sampling,
        destination=destination,
    )
    html = (destination / "review.html").read_text()
    script = re.search(r"<script>(.*)</script>", html, re.DOTALL)
    assert script is not None
    javascript = tmp_path / "review.js"
    javascript.write_text(script.group(1))
    subprocess.run([node, "--check", javascript], check=True, capture_output=True)


def test_execution_source_closure_includes_runtime_transitives() -> None:
    paths = set(_runtime_source_paths())
    assert {
        "circuits/analysis/bonafide/canonical.py",
        "circuits/analysis/bonafide/coarse_sampling_annotation.py",
        "circuits/analysis/bonafide/coarse_sampling_openai_batch_continuation_v1.py",
        "circuits/analysis/bonafide/coarse_sampling_post_campaign_v1.py",
        "circuits/analysis/bonafide/coarse_sampling_post_campaign_v2.py",
        "circuits/__init__.py",
        "circuits/analysis/__init__.py",
        "circuits/analysis/bonafide/__init__.py",
        "circuits/analysis/bonafide/features.py",
        "circuits/analysis/bonafide/identity.py",
        "circuits/analysis/bonafide/index.py",
        "circuits/analysis/bonafide/partition.py",
        "circuits/labeling/__init__.py",
        "circuits/labeling/io.py",
        "pyproject.toml",
        "uv.lock",
    }.issubset(paths)


def test_packet_uses_exact_frozen_production_ontology() -> None:
    config = json.loads(
        Path(
            "scripts/bonafide/configs/process_witness_coarse_production_v1.json"
        ).read_text()
    )
    from circuits.analysis.bonafide.coarse_sampling_commentary_audit_v1 import (
        TAG_DEFINITIONS,
    )

    assert config["tags"] == TAG_DEFINITIONS
    assert loaded_precedence() == config["decision_precedence"]


def loaded_precedence() -> list[str]:
    from circuits.analysis.bonafide.coarse_sampling_commentary_audit_v1 import (
        DECISION_PRECEDENCE,
    )

    return list(DECISION_PRECEDENCE)


def test_builder_validates_readonly_staging_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis, sampling = _source_fixture(tmp_path, monkeypatch)
    published = False

    def publish(source: Path, destination: Path) -> None:
        nonlocal published
        published = True

    monkeypatch.setattr(
        "circuits.analysis.bonafide.coarse_sampling_commentary_audit_v1."
        "_publish_no_replace",
        publish,
    )
    monkeypatch.setattr(
        "circuits.analysis.bonafide.coarse_sampling_commentary_audit_v1."
        "load_commentary_audit_packet",
        lambda root: (_ for _ in ()).throw(ValueError("staging rejected")),
    )
    with pytest.raises(ValueError, match="staging rejected"):
        build_commentary_audit_packet(
            analysis_root=analysis,
            sampling_root=sampling,
            destination=tmp_path / "packet",
        )
    assert published is False


def test_item_semantics_reject_identity_and_codepoint_span_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis, sampling = _source_fixture(tmp_path, monkeypatch)
    destination = tmp_path / "packet"
    build_commentary_audit_packet(
        analysis_root=analysis,
        sampling_root=sampling,
        destination=destination,
    )
    loaded = load_commentary_audit_packet(destination)
    item = dict(loaded["items"][0])
    document = loaded["documents"][0]
    _validate_item_semantics(item, document, loaded["packet"]["source_bindings"])
    item["item_id"] = "wrong"
    with pytest.raises(ValueError, match="item identity drift"):
        _validate_item_semantics(item, document, loaded["packet"]["source_bindings"])
    item = dict(loaded["items"][0])
    item["text"] = "wrong"
    with pytest.raises(ValueError, match="span/text drift"):
        _validate_item_semantics(item, document, loaded["packet"]["source_bindings"])


def test_artifact_binding_placeholder_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis, sampling = _source_fixture(tmp_path, monkeypatch)
    destination = tmp_path / "packet"
    build_commentary_audit_packet(
        analysis_root=analysis,
        sampling_root=sampling,
        destination=destination,
    )
    assert ARTIFACT_BINDING_PLACEHOLDER not in (destination / "review.html").read_text()
