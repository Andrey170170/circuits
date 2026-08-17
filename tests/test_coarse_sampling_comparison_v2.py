from __future__ import annotations

import copy
import json
from pathlib import Path

import circuits.analysis.bonafide.coarse_sampling_comparison_v2 as module
import pytest
from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_annotation_v2 import (
    ARM_FULL_UNIT,
    ARM_TARGET_ONLY,
    COMPARISON_PLAN,
)
from circuits.analysis.bonafide.coarse_sampling_comparison_v2 import (
    MANIFEST_SCHEMA,
    build_comparison_report,
    load_comparison_bundle,
    load_completed_comparison_inputs,
)


def _decision(unit_id: str, tag: str = "active_task_work") -> dict:
    return {
        "unit_id": unit_id,
        "tag": tag,
        "confidence": "high",
        "boundary_concerns": [],
        "boundary_note": "",
    }


def _fixture() -> dict:
    repeat_windows = {0, 5, 7, 9}
    units = [
        {
            "unit_id": f"unit-{window}-{offset}",
            "response_id": f"response-{window}",
            "text": f"text {window}:{offset}",
        }
        for window in range(12)
        for offset in range(6)
    ]
    requests = []
    events = []
    v1_events = []
    corrected = []

    def add_v1(request_id: str, window: int, *, repeat_of: str | None) -> None:
        event = {
            "request_id": request_id,
            "repeat_of_request_id": repeat_of,
            "status": "success",
            "decisions": [_decision(f"unit-{window}-{offset}") for offset in range(6)],
        }
        v1_events.append(event)
        corrected.append(
            {
                "request_id": request_id,
                "corrected_usage_from_raw_receipt": {
                    "input_tokens": 10,
                    "uncached_input_tokens": 1,
                    "cache_read_tokens": 4,
                    "cache_write_tokens": 5,
                    "output_tokens": 3,
                    "reasoning_tokens": 2,
                },
                "corrected_cost": {"total_cost": 0.01},
            }
        )

    for window in range(12):
        primary_id = f"v1-{window}"
        add_v1(primary_id, window, repeat_of=None)
        if window in repeat_windows:
            add_v1(f"v1-{window}-repeat", window, repeat_of=primary_id)

    for arm in (ARM_TARGET_ONLY, ARM_FULL_UNIT):
        for window in range(12):
            source_id = f"v1-{window}"
            primary_id = f"v2-{arm}-{window}"
            focal = [f"unit-{window}-{offset}" for offset in range(6)]
            request = {
                "request_id": primary_id,
                "arm_id": arm,
                "source_v1_request_id": source_id,
                "repeat_of_request_id": None,
                "window_index": window,
                "response_id": f"response-{window}",
                "focal_unit_ids": focal,
            }
            decisions = [_decision(unit_id) for unit_id in focal]
            if arm == ARM_FULL_UNIT and window == 0:
                decisions[0]["tag"] = "evaluation_or_revision"
            event = {
                **{
                    key: request[key]
                    for key in (
                        "request_id",
                        "arm_id",
                        "source_v1_request_id",
                        "repeat_of_request_id",
                    )
                },
                "validation_status": "success",
                "decisions": decisions,
                "usage": {
                    "input_tokens": 100,
                    "uncached_input_tokens": 1,
                    "cache_read_tokens": 40,
                    "cache_write_tokens": 59,
                    "output_tokens": 20,
                    "reasoning_tokens": 10,
                },
                "cost": {"total_cost": 0.1},
            }
            requests.append(request)
            events.append(event)
            if window in repeat_windows:
                repeat = {
                    **request,
                    "request_id": f"{primary_id}-repeat",
                    "source_v1_request_id": f"v1-{window}-repeat",
                    "repeat_of_request_id": primary_id,
                }
                repeat_decisions = copy.deepcopy(decisions)
                if arm == ARM_TARGET_ONLY and window == 0:
                    repeat_decisions[0]["tag"] = "other_semantic_text"
                repeat_event = {
                    **{
                        key: repeat[key]
                        for key in (
                            "request_id",
                            "arm_id",
                            "source_v1_request_id",
                            "repeat_of_request_id",
                        )
                    },
                    "validation_status": "success",
                    "decisions": repeat_decisions,
                    "usage": copy.deepcopy(event["usage"]),
                    "cost": {"total_cost": 0.1},
                }
                requests.append(repeat)
                events.append(repeat_event)

    qualification = {
        "manifest": {"manifest_sha256": "q" * 64},
        "config": {"comparison_plan": copy.deepcopy(COMPARISON_PLAN)},
        "requests": requests,
        "focal_units": units,
        "v1_comparison_baseline": {
            "manifest": {"run_manifest_sha256": "v" * 64},
            "events": v1_events,
        },
        "v1_cost_correction_audit": {
            "cost_correction_audit_sha256": "a" * 64,
            "corrected_total_cost_usd": 0.16,
            "requests": corrected,
        },
    }
    qualification["config"]["source"] = {"v1_completed_events_sha256": "e" * 64}
    return {
        "qualification": qualification,
        "run_intent": {"run_intent_sha256": "r" * 64},
        "collection": {
            "collection_manifest_sha256": "c" * 64,
            "events_jsonl_sha256": "f" * 64,
            "actual_total_cost_usd": 3.2,
            "usage_totals": {
                "input_tokens": 3200,
                "uncached_input_tokens": 32,
                "cache_read_tokens": 1280,
                "cache_write_tokens": 1888,
                "output_tokens": 640,
                "reasoning_tokens": 320,
            },
        },
        "events": events,
    }


def test_report_executes_all_predeclared_pairings_and_matrices() -> None:
    report, review_rows = build_comparison_report(_fixture())
    summaries = {
        (item["comparison_id"], item["stratum"]): item
        for item in report["comparison_summaries"]
    }
    assert summaries[("within_arm_repeat", ARM_TARGET_ONLY)]["counts"] == {
        "expected": 24,
        "observed": 24,
        "eligible": 24,
        "missing": 0,
        "invalid": 0,
    }
    repeat = summaries[("within_arm_repeat", ARM_TARGET_ONLY)]
    assert repeat["agreement"]["tag_agreement"]["disagree"] == 1
    assert repeat["row_side"] == f"{ARM_TARGET_ONLY}:repeat"
    assert repeat["column_side"] == f"{ARM_TARGET_ONLY}:primary"
    assert (
        repeat["tag_confusion"]["counts"]["other_semantic_text"]["active_task_work"]
        == 1
    )
    cross = summaries[("cross_arm_primary", "all_primary_targets")]
    assert cross["counts"]["expected"] == 72
    assert cross["agreement"]["tag_agreement"]["disagree"] == 1
    assert (
        cross["tag_confusion"]["counts"]["active_task_work"]["evaluation_or_revision"]
        == 1
    )
    assert cross["confidence_confusion"] is not None
    assert cross["boundary_concern_confusions"] is not None
    assert (
        summaries[("each_arm_vs_v1_primary", ARM_FULL_UNIT)]["counts"]["expected"] == 72
    )
    assert (
        summaries[("each_arm_vs_v1_repeat", ARM_TARGET_ONLY)]["counts"]["expected"]
        == 24
    )
    assert report["formal_pass_threshold"] is None
    assert report["human_blind_review_status"] == "deferred"
    assert review_rows
    assert "row_source" not in review_rows[0]
    assert len(report["review_unblinding_bindings"]) == len(review_rows)


def test_usage_cost_is_exactly_stratified_and_uses_corrected_v1_receipts() -> None:
    report, _ = build_comparison_report(_fixture())
    target = report["usage_cost"]["v2_by_arm"][ARM_TARGET_ONLY]
    assert target["all"]["request_count"] == 16
    assert target["primary"]["request_count"] == 12
    assert target["repeat"]["request_count"] == 4
    assert target["all"]["token_totals"]["cache_read_tokens"] == 640
    assert target["all"]["token_totals"]["cache_write_tokens"] == 944
    assert target["all"]["total_cost_usd"] == pytest.approx(1.6)
    v1 = report["usage_cost"]["v1_baseline"]
    assert v1["all"]["request_count"] == 16
    assert v1["primary"]["request_count"] == 12
    assert v1["repeat"]["request_count"] == 4
    assert v1["all"]["token_totals"]["cache_write_tokens"] == 80
    assert v1["all"]["total_cost_usd"] == pytest.approx(0.16)


def test_usage_missingness_fails_closed() -> None:
    inputs = _fixture()
    inputs["events"][0]["usage"]["cache_write_tokens"] = None
    with pytest.raises(ValueError, match="missingness is nonzero"):
        build_comparison_report(inputs)


def test_completed_input_loader_binds_hashes_order_and_exact_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture()
    qualification_root = tmp_path / "qualification"
    qualification_root.mkdir()
    run_root = tmp_path / "run"
    run_root.mkdir()
    monkeypatch.setattr(
        module, "load_v2_qualification", lambda _: fixture["qualification"]
    )
    intent = {
        "schema_version": "adag.process-witness.coarse-openai-batch-run.v2",
        "qualification_manifest_sha256": "q" * 64,
        "qualification_root": str(qualification_root.resolve()),
    }
    intent["run_intent_sha256"] = canonical_sha256(intent)
    (run_root / "run-intent.json").write_text(json.dumps(intent))
    event_bindings = []
    for event in fixture["events"]:
        event["event_sha256"] = canonical_sha256(event)
        event_bindings.append(
            {
                "request_id": event["request_id"],
                "event_sha256": event["event_sha256"],
            }
        )
    (run_root / "events.jsonl").write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in fixture["events"])
    )
    collection = {
        "schema_version": "adag.process-witness.coarse-openai-batch-collection.v2",
        "status": "complete",
        "qualification_decisions_ready": True,
        "cost_complete": True,
        "authorization_exceeded": False,
        "request_count": 32,
        "success_count": 32,
        "failure_count": 0,
        "exact_target_coverage": True,
        "unique_arm_target_coverage": 144,
        "run_intent_sha256": intent["run_intent_sha256"],
        "events_jsonl_sha256": file_sha256(run_root / "events.jsonl"),
        "event_bindings_in_order": event_bindings,
        "actual_total_cost_usd": 3.2,
        "usage_totals": fixture["collection"]["usage_totals"],
    }
    collection["collection_manifest_sha256"] = canonical_sha256(collection)
    (run_root / "collection-manifest.json").write_text(json.dumps(collection))

    loaded = load_completed_comparison_inputs(
        run_root=run_root, qualification_root=qualification_root
    )
    assert len(loaded["events"]) == 32

    truncated = (run_root / "events.jsonl").read_text().splitlines()[:-1]
    (run_root / "events.jsonl").write_text("\n".join(truncated) + "\n")
    with pytest.raises(ValueError, match="events file drift"):
        load_completed_comparison_inputs(
            run_root=run_root, qualification_root=qualification_root
        )


def test_immutable_bundle_loader_rejects_payload_tampering(tmp_path: Path) -> None:
    report, examples = build_comparison_report(_fixture())
    (tmp_path / "comparison-report.json").write_text(
        json.dumps(report, sort_keys=True) + "\n"
    )
    (tmp_path / "examples-disagreements.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in examples)
    )
    files = [
        {
            "path": path.name,
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(tmp_path.iterdir())
    ]
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "complete_offline_immutable_comparison",
        "network_calls_made": 0,
        "report_sha256": report["report_sha256"],
        "comparison_plan_sha256": report["comparison_plan_sha256"],
        "example_row_count": len(examples),
        "files": files,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert len(load_comparison_bundle(tmp_path)["examples"]) == len(examples)

    with (tmp_path / "examples-disagreements.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="payload drift"):
        load_comparison_bundle(tmp_path)
