from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_annotation import (
    cost_plan,
    load_coarse_config,
)
from circuits.analysis.bonafide.coarse_sampling_review import (
    build_review_packet,
    merge_review_rows,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts/bonafide/configs/process_witness_coarse_openai_v1.json"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    qualification = tmp_path / "qualification"
    run = tmp_path / "run"
    qualification.mkdir()
    for directory in ("intents", "raw", "records"):
        (run / directory).mkdir(parents=True, exist_ok=True)

    requests: list[dict[str, object]] = []
    templates: list[dict[str, object]] = []
    primary_ids: list[str] = []
    repeat_indices = {0, 5, 7, 9}
    physical_index = 0
    for window_index in range(12):
        request_id = f"request-{physical_index}"
        primary_ids.append(request_id)
        focal = [f"unit-{window_index}-{index}" for index in range(6)]
        context = [f"context-{window_index}", *focal, f"tail-{window_index}"]
        body = {
            "model": "gpt-5.6-luna",
            "input": [{"role": "user", "content": f"window {window_index}"}],
            "max_output_tokens": 100,
            "store": False,
        }
        request = {
            "request_id": request_id,
            "body_sha256": canonical_sha256(body),
            "repeat_of_request_id": None,
            "focal_unit_ids": focal,
            "context_unit_ids": context,
            "provider_body": body,
            "response_id": f"response-{window_index}",
            "prompt_sha256": f"prompt-{window_index}",
            "window_index": window_index,
        }
        requests.append(request)
        physical_index += 1
        if window_index in repeat_indices:
            requests.append(
                {
                    **request,
                    "request_id": f"request-{physical_index}",
                    "repeat_of_request_id": request_id,
                }
            )
            physical_index += 1
        bounded = [
            {
                "unit_id": unit_id,
                "role": "target" if unit_id in focal else "context",
                "text": f"text for {unit_id}",
            }
            for unit_id in context
        ]
        for unit_index, unit_id in enumerate(focal):
            templates.append(
                {
                    "schema_version": "adag.process-witness.coarse-human-review-template.v1",
                    "window_index": window_index,
                    "unit_id": unit_id,
                    "response_id": f"response-{window_index}",
                    "prompt_sha256": f"prompt-{window_index}",
                    "source_type_stratum": "simple",
                    "position_stratum": "early",
                    "task_prompt": f"prompt text {window_index} </script>",
                    "bounded_response_units": bounded,
                    "token_span": [unit_index, unit_index + 1],
                    "text": f"text for {unit_id}",
                    "human_tag": None,
                    "boundary_concerns": [],
                    "notes": "",
                }
            )

    _write_jsonl(qualification / "requests.jsonl", requests)
    _write_jsonl(qualification / "human-review-template.jsonl", templates)
    (qualification / "units.jsonl").write_text("")
    _write_json(qualification / "windows.json", [])
    prices = {
        "schema_version": "adag.labeling.prices.v1",
        "snapshot_id": "test-prices",
        "rates": {
            "openai": {
                "gpt-5.6-luna": {
                    "live": {
                        "input_per_million": 0.2,
                        "cache_read_per_million": 0.02,
                        "cache_write_per_million": 0.25,
                        "output_per_million": 1.2,
                    }
                }
            }
        },
    }
    price_path = qualification / "prices.json"
    _write_json(price_path, prices)
    plan = cost_plan(requests, load_coarse_config(CONFIG), prices)
    plan.update(
        {
            "price_snapshot_path": str(price_path),
            "price_snapshot_sha256": file_sha256(price_path),
        }
    )
    plan["cost_plan_sha256"] = canonical_sha256(plan)
    _write_json(qualification / "cost-plan.json", plan)
    required = (
        "cost-plan.json",
        "human-review-template.jsonl",
        "requests.jsonl",
        "units.jsonl",
        "windows.json",
    )
    qualification_manifest = {
        "schema_version": "adag.process-witness.coarse-qualification-bundle.v1",
        "status": "prepared_offline_no_provider_calls",
        "network_calls_made": 0,
        "claim_boundary": "selection only",
        "qualification_claim_boundary": "smoke only",
        "config_path": str(CONFIG),
        "config_sha256": file_sha256(CONFIG),
        "files": [
            {
                "path": name,
                "bytes": (qualification / name).stat().st_size,
                "sha256": file_sha256(qualification / name),
            }
            for name in required
        ],
        "request_bindings_in_order": [
            {
                "request_id": request["request_id"],
                "body_sha256": request["body_sha256"],
                "repeat_of_request_id": request["repeat_of_request_id"],
            }
            for request in requests
        ],
    }
    qualification_manifest["manifest_sha256"] = canonical_sha256(qualification_manifest)
    _write_json(qualification / "manifest.json", qualification_manifest)

    request_ids = [str(request["request_id"]) for request in requests]
    run_intent = {
        "schema_version": "adag.process-witness.coarse-openai-run.v1",
        "status": "intent_persisted_before_provider_calls",
        "qualification_root": str(qualification),
        "qualification_manifest_sha256": qualification_manifest["manifest_sha256"],
        "cost_plan_sha256": plan["cost_plan_sha256"],
        "request_ids_in_order": request_ids,
    }
    run_intent["run_intent_sha256"] = canonical_sha256(run_intent)
    _write_json(run / "run-intent.json", run_intent)

    events: list[dict[str, object]] = []
    total_cost = 0.0
    for index, request in enumerate(requests):
        request_id = str(request["request_id"])
        intent = {
            "schema_version": "adag.process-witness.coarse-openai-attempt-intent.v1",
            "run_intent_sha256": run_intent["run_intent_sha256"],
            "request_id": request_id,
            "body_sha256": request["body_sha256"],
            "repeat_of_request_id": request["repeat_of_request_id"],
        }
        intent["intent_sha256"] = canonical_sha256(intent)
        _write_json(run / "intents" / f"{request_id}.json", intent)
        raw_usage = {
            "input_tokens": 100,
            "input_tokens_details": {
                "cached_tokens": 20,
                "cache_write_tokens": 30,
            },
            "output_tokens": 25,
            "output_tokens_details": {"reasoning_tokens": 5},
            "total_tokens": 125,
        }
        raw = {"id": f"response-{index}", "usage": raw_usage}
        _write_json(run / "raw" / f"{request_id}.json", raw)
        decisions = []
        for unit_index, unit_id in enumerate(request["focal_unit_ids"]):
            tag = "active_task_work"
            if request["repeat_of_request_id"] is not None and unit_index == 0:
                tag = "evaluation_or_revision"
            decisions.append(
                {
                    "unit_id": unit_id,
                    "tag": tag,
                    "confidence": "high",
                    "boundary_concerns": [],
                    "boundary_note": "",
                }
            )
        raw_text = json.dumps({"decisions": decisions})
        recorded_request_cost = 0.0000464
        total_cost += recorded_request_cost
        event = {
            "schema_version": "adag.process-witness.coarse-openai-event.v1",
            "status": "success",
            "request_id": request_id,
            "body_sha256": request["body_sha256"],
            "repeat_of_request_id": request["repeat_of_request_id"],
            "intent_sha256": intent["intent_sha256"],
            "model_resolved": "gpt-5.6-luna",
            "raw_response_path": f"raw/{request_id}.json",
            "raw_response_sha256": file_sha256(run / "raw" / f"{request_id}.json"),
            "raw_text": raw_text,
            "raw_text_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
            "decisions": decisions,
            "usage": {
                "input_tokens": 100,
                "uncached_input_tokens": 80,
                "cache_read_tokens": 20,
                "cache_write_tokens": 0,
                "output_tokens": 25,
                "reasoning_tokens": 5,
            },
            "cost": {
                "currency": "USD",
                "price_snapshot_id": "test-prices",
                "input_cost": 0.000016,
                "cache_read_cost": 0.0000004,
                "cache_write_cost": 0.0,
                "output_cost": 0.00003,
                "total_cost": recorded_request_cost,
                "complete": True,
                "missing_components": [],
            },
            "cumulative_cost_usd": total_cost,
        }
        event["event_sha256"] = canonical_sha256(event)
        events.append(event)
        _write_json(run / "records" / f"{request_id}.json", event)
    _write_jsonl(run / "events.jsonl", events)
    run_manifest = {
        **{
            key: value
            for key, value in run_intent.items()
            if key != "run_intent_sha256"
        },
        "status": "complete",
        "event_count": 16,
        "success_count": 16,
        "invalid_output_count": 0,
        "provider_incomplete_count": 0,
        "cost_complete": True,
        "known_priced_cost_usd": total_cost,
        "actual_total_cost_usd": total_cost,
        "events_jsonl_sha256": file_sha256(run / "events.jsonl"),
        "record_bindings_in_order": [
            {"request_id": event["request_id"], "event_sha256": event["event_sha256"]}
            for event in events
        ],
    }
    run_manifest["run_manifest_sha256"] = canonical_sha256(run_manifest)
    _write_json(run / "run-manifest.json", run_manifest)
    return qualification, run


def test_merge_review_rows_keeps_unique_units_and_exposes_repeats(
    tmp_path: Path,
) -> None:
    qualification, run = _fixture(tmp_path)
    rows, sources = merge_review_rows(qualification_root=qualification, run_root=run)
    assert len(rows) == 72
    assert len({row["unit_id"] for row in rows}) == 72
    assert sum(row["machine_repeat"] is not None for row in rows) == 24
    assert sum(row["repeat_tag_agreement"] is False for row in rows) == 4
    assert all(row["human_tag"] is None for row in rows)
    assert sources["packet_id"].startswith("process-witness-coarse-review-v1-")
    assert rows[0]["bounded_response_units"][1]["is_reviewed_focal"] is True


def test_merge_review_rows_rejects_tampered_record(tmp_path: Path) -> None:
    qualification, run = _fixture(tmp_path)
    record = run / "records" / "request-0.json"
    value = json.loads(record.read_text())
    value["decisions"][0]["tag"] = "uncertain"
    _write_json(record, value)
    with pytest.raises(ValueError, match="event/record content drift"):
        merge_review_rows(qualification_root=qualification, run_root=run)


def test_build_review_packet_binds_sources_and_embeds_safe_html(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification, run = _fixture(tmp_path)
    monkeypatch.setattr(
        "circuits.analysis.bonafide.coarse_sampling_review._source_revision",
        lambda: {
            "git_commit": "a" * 40,
            "tracked_worktree_clean": True,
            "files": [{"path": "builder.py", "sha256": "b" * 64}],
        },
    )
    monkeypatch.setattr(
        "circuits.analysis.bonafide.coarse_sampling_review._readonly_tree",
        lambda _: None,
    )
    output = tmp_path / "review-v1"
    manifest = build_review_packet(
        qualification_root=qualification, run_root=run, destination=output
    )
    assert manifest["status"] == "frozen_offline_human_review_packet"
    assert manifest["network_calls_made"] == 0
    assert manifest["counts"] == {
        "review_rows": 72,
        "rows_with_repeat": 24,
        "repeat_tag_disagreements": 4,
    }
    assert manifest["ui_sha256"] == file_sha256(output / "review.html")
    assert manifest["ui_version"].endswith("blind-first")
    assert manifest["original_recorded_total_cost_usd"] == pytest.approx(0.0007424)
    assert manifest["corrected_total_cost_usd"] == pytest.approx(0.0007664)
    assert manifest["cost_delta_usd"] == pytest.approx(0.000024)
    assert manifest["run_events_jsonl_sha256"] == file_sha256(run / "events.jsonl")
    assert {path.name for path in output.iterdir()} == {
        "cost-correction-audit.json",
        "manifest.json",
        "review-rows.jsonl",
        "review.html",
    }
    audit = json.loads((output / "cost-correction-audit.json").read_text())
    audit_payload = {
        key: value
        for key, value in audit.items()
        if key != "cost_correction_audit_sha256"
    }
    assert audit["cost_correction_audit_sha256"] == canonical_sha256(audit_payload)
    assert (
        manifest["cost_correction_audit_sha256"]
        == audit["cost_correction_audit_sha256"]
    )
    assert manifest["cost_correction_audit_file_sha256"] == file_sha256(
        output / "cost-correction-audit.json"
    )
    assert len(audit["requests"]) == 16
    assert audit["requests"][0]["corrected_usage_from_raw_receipt"] == {
        "input_tokens": 100,
        "uncached_input_tokens": 50,
        "cache_read_tokens": 20,
        "cache_write_tokens": 30,
        "output_tokens": 25,
        "reasoning_tokens": 5,
    }
    html = (output / "review.html").read_text()
    assert "Export JSONL" in html
    assert "Lock judgment and reveal labels" in html
    assert "pre_reveal_judgment_locked" in html
    assert "model_labels_revealed_at" in html
    assert manifest["ui_template_sha256"] in html
    assert "prompt text 0 </script>" not in html
    assert "adag.process-witness.coarse-human-review-decision.v1" in html


def test_merge_review_rows_rejects_invalid_raw_usage(tmp_path: Path) -> None:
    qualification, run = _fixture(tmp_path)
    raw_path = run / "raw" / "request-0.json"
    raw = json.loads(raw_path.read_text())
    raw["usage"]["input_tokens_details"]["cache_write_tokens"] = 200
    _write_json(raw_path, raw)
    event_path = run / "records" / "request-0.json"
    event = json.loads(event_path.read_text())
    event["raw_response_sha256"] = file_sha256(raw_path)
    event["event_sha256"] = canonical_sha256(
        {key: value for key, value in event.items() if key != "event_sha256"}
    )
    _write_json(event_path, event)
    events_path = run / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    events[0] = event
    _write_jsonl(events_path, events)
    manifest_path = run / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["events_jsonl_sha256"] = file_sha256(events_path)
    manifest["record_bindings_in_order"][0]["event_sha256"] = event["event_sha256"]
    manifest["run_manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "run_manifest_sha256"}
    )
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="input buckets exceed total"):
        merge_review_rows(qualification_root=qualification, run_root=run)


def test_build_refuses_existing_destination(tmp_path: Path) -> None:
    qualification, run = _fixture(tmp_path)
    output = tmp_path / "review-v1"
    output.mkdir()
    with pytest.raises(FileExistsError):
        build_review_packet(
            qualification_root=qualification, run_root=run, destination=output
        )
