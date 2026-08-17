from __future__ import annotations

import copy
from pathlib import Path

import scripts.bonafide.build_process_witness_coarse_compatibility_v4 as builder
from circuits.analysis.bonafide.coarse_sampling_annotation_v4 import (
    load_coarse_v4_config,
)
from circuits.analysis.bonafide.coarse_sampling_compatibility_v4 import (
    build_compatibility_report,
)
from circuits.analysis.bonafide.coarse_sampling_review_v4 import EXPORT_SCHEMA

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts/bonafide/configs/process_witness_coarse_openai_v4.json"


def _inputs() -> dict:
    unit_ids = [f"unit-{index}" for index in range(24)]
    roles = ["repair"] * 14 + ["unchanged_short"] * 6 + ["long_diagnostic"] * 4
    packet = {
        "packet_id": "packet-v4",
        "packet_binding_sha256": "b" * 64,
        "qualification_manifest_sha256": "q" * 64,
    }
    items = [
        {"unit_id": unit_id, "item_id": f"item-{index}"}
        for index, unit_id in enumerate(unit_ids)
    ]
    human = [
        {
            "schema_version": EXPORT_SCHEMA,
            "packet_id": packet["packet_id"],
            "packet_binding_sha256": packet["packet_binding_sha256"],
            "item_id": item["item_id"],
            "unit_id": item["unit_id"],
            "primary_label": "active_task_work",
            "defensible_alternatives": [],
            "boundary_concerns": [],
            "note": "",
            "globally_sealed": True,
            "global_seal_id": "00000000-0000-4000-8000-000000000004",
            "global_sealed_at": "2026-08-17T00:00:00Z",
        }
        for item in items
    ]
    decisions = [
        {
            "unit_id": unit_id,
            "tag": "active_task_work",
            "confidence": "high",
            "boundary_concerns": [],
            "boundary_note": "",
        }
        for unit_id in unit_ids
    ]
    return {
        "qualification": {
            "manifest": {"manifest_sha256": "q" * 64},
            "config": load_coarse_v4_config(CONFIG),
            "focal_units": [{"unit_id": unit_id} for unit_id in unit_ids],
            "windows": [{"target_roles": dict(zip(unit_ids, roles, strict=True))}],
        },
        "review": {
            "manifest": {"manifest_sha256": "v" * 64},
            "packet": packet,
            "items": items,
        },
        "human": human,
        "events": [
            {"validation_status": "success", "decisions": copy.deepcopy(decisions)}
            for _ in range(3)
        ],
        "collection": {"collection_manifest_sha256": "c" * 64},
        "human_ledger_sha256": "h" * 64,
    }


def test_compatibility_gate_passes_exact_clean_panel() -> None:
    report = build_compatibility_report(_inputs())
    assert report["status"] == "passed"
    assert all(report["gate_checks"].values())
    assert report["metrics"]["gated_targets"] == 20
    assert report["metrics"]["long_diagnostic_targets"] == 4


def test_human_repair_boundary_fails_even_when_model_reports_none() -> None:
    inputs = _inputs()
    inputs["human"][0]["boundary_concerns"] = ["merge_next"]
    report = build_compatibility_report(inputs)
    assert report["status"] == "failed_closed"
    assert (
        report["gate_checks"]["maximum_merge_or_split_flags_on_20_gated_units"] is False
    )
    assert report["metrics"]["gated_human_merge_or_split_flags"] == 1
    assert report["metrics"]["gated_model_merge_or_split_majorities"] == 0


def test_compatibility_artifact_binds_source_and_copies_sealed_ledger(
    tmp_path: Path, monkeypatch
) -> None:
    ledger = tmp_path / "human.jsonl"
    ledger.write_text('{"sealed":true}\n', encoding="utf-8")
    human_hash = __import__("hashlib").sha256(ledger.read_bytes()).hexdigest()
    monkeypatch.setattr(builder, "_source_revision", lambda: {"git_commit": "a" * 40})
    monkeypatch.setattr(builder, "load_completed_v4_inputs", lambda **_: {})
    monkeypatch.setattr(
        builder,
        "build_compatibility_report",
        lambda _: {
            "schema_version": "test",
            "status": "passed",
            "human_ledger_sha256": human_hash,
            "rows": [{"unit_id": "u"}],
        },
    )
    destination = tmp_path / "artifact"
    manifest = builder.build(
        qualification_root=tmp_path / "qualification",
        run_root=tmp_path / "run",
        review_root=tmp_path / "review",
        human_ledger_path=ledger,
        destination=destination,
    )
    assert manifest["source_revision"] == {"git_commit": "a" * 40}
    assert manifest["human_ledger_sha256"] == human_hash
    assert (
        destination / "sealed-human-ledger.jsonl"
    ).read_bytes() == ledger.read_bytes()
    assert {
        "circuits/analysis/bonafide/coarse_sampling_compatibility_v4.py",
        "scripts/bonafide/build_process_witness_coarse_compatibility_v4.py",
        "scripts/bonafide/configs/process_witness_coarse_openai_v4.json",
    } <= set(builder._BOUND_SOURCE_FILES)
