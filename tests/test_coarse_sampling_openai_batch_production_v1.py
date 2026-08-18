from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import circuits.analysis.bonafide.coarse_sampling_openai_batch_production_v1 as module
import pytest
from circuits.analysis.bonafide.coarse_sampling_openai_batch_production_v1 import (
    _copy_campaign_evidence,
    _parse_row,
    _submission_gate,
    collect_recovery_shard,
    collect_shard,
    initialize_campaign_run,
    load_frozen_proposal_bank,
    prepare_failed_only_recovery,
    recover_shard_submission,
    submit_shard,
)


def _provider(metadata: dict[str, str]) -> dict:
    return {
        "schema_version": "adag.labeling.provider-batch.v1",
        "provider": "openai",
        "batch_id": "batch-1",
        "input_file_id": "file-1",
        "endpoint": "/v1/responses",
        "completion_window": "24h",
        "metadata": metadata,
        "status": "validating",
        "output_file_id": None,
        "error_file_id": None,
        "request_counts": None,
        "model": None,
        "usage": None,
    }


def test_initialize_requires_queue_capacity_for_requested_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    shards = []
    for index, forecast in enumerate((60, 50, 20)):
        path = bundle_root / f"shard-{index}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        shards.append(
            {
                "shard_id": f"shard-{index}",
                "path": path.name,
                "sha256": module.file_sha256(path),
                "bytes": path.stat().st_size,
                "request_count": 1,
                "direct_v4_cost_forecast_usd": 1.0,
                "queued_input_tokens_empirical_forecast": forecast,
            }
        )
    loaded = {
        "manifest": {"manifest_sha256": "m" * 64},
        "cost_plan": {"cost_plan_sha256": "c" * 64},
        "shards": shards,
    }
    monkeypatch.setattr(
        module, "load_production_bundle", lambda *args, **kwargs: loaded
    )
    with pytest.raises(ValueError, match="queue limit"):
        initialize_campaign_run(
            bundle_root=bundle_root,
            run_root=tmp_path / "too-small",
            maximum_authorized_cost_usd=20,
            authorization_note="test authorization",
            provider_queued_input_token_limit=100,
            maximum_concurrent_shards=2,
        )
    intent = initialize_campaign_run(
        bundle_root=bundle_root,
        run_root=tmp_path / "run",
        maximum_authorized_cost_usd=20,
        authorization_note="test authorization",
        provider_queued_input_token_limit=110,
        maximum_concurrent_shards=2,
    )
    assert intent["maximum_concurrent_shards"] == 2
    assert intent["provider_queued_input_token_limit"] == 110
    assert intent["network_calls_made"] == 0


def test_submit_persists_intent_and_fails_closed_on_ambiguous_provider_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    shard_root = run / "shards" / "shard-000"
    shard_root.mkdir(parents=True)
    (shard_root / "input.jsonl").write_text("{}\n", encoding="utf-8")
    intent = {
        "campaign_run_sha256": "c" * 64,
        "maximum_authorized_cost_usd": 20.0,
        "maximum_concurrent_shards": 1,
        "provider_queued_input_token_limit": 1000,
        "shards": [
            {
                "shard_id": "shard-000",
                "direct_v4_cost_forecast_usd": 1.0,
                "queued_input_tokens_empirical_forecast": 100,
            }
        ],
    }
    bundle = {
        "shards": [
            {
                "shard_id": "shard-000",
                "request_count": 1,
            }
        ]
    }
    monkeypatch.setattr(module, "_campaign", lambda _: (intent, bundle))

    def upload(_path: Path) -> dict:
        return {
            "schema_version": module.UPLOAD_SCHEMA,
            "provider": "openai",
            "input_file_id": "file-1",
            "purpose": "batch",
        }

    def fail(*args, **kwargs):
        raise TimeoutError("create response lost")

    with pytest.raises(RuntimeError, match="indeterminate"):
        submit_shard(
            run_root=run, shard_id="shard-000", uploader=upload, creator=fail
        )
    assert (shard_root / "submission-intent.json").is_file()
    assert (shard_root / "provider-upload-response.json").is_file()
    failure = json.loads((shard_root / "submission-failure.json").read_text())
    assert failure["automatic_retry_permitted"] is False
    assert failure["upload_receipt_persisted"] is True
    recovered = recover_shard_submission(
        run_root=run,
        shard_id="shard-000",
        discoverer=lambda **_: [_provider(module._metadata(intent, "shard-000", "primary"))],
    )
    assert recovered["recovered_by"] == "unique_provider_metadata_discovery"
    assert recovered["provider_response"]["input_file_id"] == "file-1"


def test_submit_reserves_completed_but_uncollected_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    for shard_id in ("shard-000", "shard-001"):
        root = run / "shards" / shard_id
        root.mkdir(parents=True)
        (root / "input.jsonl").write_text("{}\n", encoding="utf-8")
    (run / "shards" / "shard-000" / "submission-intent.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (run / "shards" / "shard-000" / "submission.json").write_text(
        json.dumps({"provider_response": {"status": "completed"}}), encoding="utf-8"
    )
    intent = {
        "campaign_run_sha256": "c" * 64,
        "maximum_authorized_cost_usd": 1.5,
        "maximum_concurrent_shards": 2,
        "provider_queued_input_token_limit": 1000,
        "shards": [
            {
                "shard_id": shard_id,
                "direct_v4_cost_forecast_usd": 1.0,
                "queued_input_tokens_empirical_forecast": 100,
            }
            for shard_id in ("shard-000", "shard-001")
        ],
    }
    bundle = {
        "shards": [
            {"shard_id": shard_id, "request_count": 1}
            for shard_id in ("shard-000", "shard-001")
        ]
    }
    monkeypatch.setattr(module, "_campaign", lambda _: (intent, bundle))
    with pytest.raises(ValueError, match="prospective actual-or-reserved"):
        submit_shard(
            run_root=run,
            shard_id="shard-001",
            uploader=lambda _: (_ for _ in ()).throw(AssertionError("uploaded")),
        )


def test_submit_reserves_cross_shard_queue_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    for shard_id in ("shard-000", "shard-001"):
        root = run / "shards" / shard_id
        root.mkdir(parents=True)
        (root / "input.jsonl").write_text("{}\n")
    active = run / "shards" / "shard-000"
    (active / "submission-intent.json").write_text("{}\n")
    (active / "submission.json").write_text(
        json.dumps({"provider_response": {"status": "in_progress"}})
    )
    intent = {
        "campaign_run_sha256": "c" * 64,
        "maximum_authorized_cost_usd": 10.0,
        "maximum_concurrent_shards": 2,
        "provider_queued_input_token_limit": 150,
        "shards": [
            {
                "shard_id": shard_id,
                "direct_v4_cost_forecast_usd": 1.0,
                "queued_input_tokens_empirical_forecast": 100,
            }
            for shard_id in ("shard-000", "shard-001")
        ],
    }
    bundle = {
        "shards": [
            {"shard_id": shard_id, "request_count": 1}
            for shard_id in ("shard-000", "shard-001")
        ]
    }
    monkeypatch.setattr(module, "_campaign", lambda _: (intent, bundle))
    with pytest.raises(ValueError, match="queued-input-token capacity"):
        submit_shard(run_root=run, shard_id="shard-001")


def test_submission_gate_is_exclusive(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    with _submission_gate(run), pytest.raises(
        RuntimeError, match="already held or stale"
    ), _submission_gate(run):
        pass
    assert not (run / ".submission-gate").exists()


def test_simultaneous_same_shard_submit_calls_provider_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    attempt = run / "shards" / "shard-000"
    attempt.mkdir(parents=True)
    (attempt / "input.jsonl").write_text("{}\n")
    intent = {
        "campaign_run_sha256": "c" * 64,
        "maximum_authorized_cost_usd": 2.0,
        "maximum_concurrent_shards": 1,
        "provider_queued_input_token_limit": 1000,
        "shards": [
            {
                "shard_id": "shard-000",
                "direct_v4_cost_forecast_usd": 1.0,
                "queued_input_tokens_empirical_forecast": 100,
            }
        ],
    }
    bundle = {"shards": [{"shard_id": "shard-000", "request_count": 1}]}
    monkeypatch.setattr(module, "_campaign", lambda _: (intent, bundle))
    started = threading.Event()
    release = threading.Event()
    calls = []

    def upload(_path: Path) -> dict:
        calls.append("upload")
        started.set()
        assert release.wait(5)
        return {
            "schema_version": module.UPLOAD_SCHEMA,
            "provider": "openai",
            "input_file_id": "file-1",
            "purpose": "batch",
        }

    def creator(_file_id: str, *, metadata: dict[str, str]) -> dict:
        return _provider(metadata)

    errors = []

    def first() -> None:
        try:
            submit_shard(
                run_root=run,
                shard_id="shard-000",
                uploader=upload,
                creator=creator,
            )
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=first)
    thread.start()
    assert started.wait(5)
    with pytest.raises(RuntimeError, match="already held or stale"):
        submit_shard(
            run_root=run,
            shard_id="shard-000",
            uploader=upload,
            creator=creator,
        )
    release.set()
    thread.join(5)
    assert not thread.is_alive()
    assert errors == []
    assert calls == ["upload"]


def test_copied_campaign_evidence_survives_source_removal(tmp_path: Path) -> None:
    run = tmp_path / "run"
    source_bundle = tmp_path / "bundle"
    shard_id = "shard-000"
    batch_relative = f"batch-shards/{shard_id}.jsonl"
    (source_bundle / "batch-shards").mkdir(parents=True)
    (source_bundle / batch_relative).write_text('{"custom_id":"r"}\n')
    (source_bundle / "manifest.json").write_text("{}\n")
    attempt = run / "shards" / shard_id
    (attempt / "raw").mkdir(parents=True)
    (attempt / "input.jsonl").write_text('{"custom_id":"r"}\n')
    (attempt / "raw" / "output.jsonl").write_text('{"custom_id":"r"}\n')
    (run / "campaign-intent.json").write_text("{}\n")
    destination = tmp_path / "final"
    destination.mkdir()
    _copy_campaign_evidence(
        run_root=run,
        temporary=destination,
        intent={"bundle_root": str(source_bundle)},
        bundle={"shards": [{"shard_id": shard_id, "path": batch_relative}]},
    )
    shutil.rmtree(run)
    shutil.rmtree(source_bundle)
    assert (destination / "campaign-bundle" / batch_relative).is_file()
    assert (destination / "shards" / shard_id / "input.jsonl").read_text() == (
        '{"custom_id":"r"}\n'
    )
    assert (destination / "shards" / shard_id / "raw/output.jsonl").is_file()


def test_strict_final_loader_binds_raw_provider_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "final"
    attempt = root / "shards" / "shard-000"
    raw = attempt / "raw"
    raw.mkdir(parents=True)
    (root / "campaign-bundle").mkdir()
    bundle_hash = "b" * 64
    monkeypatch.setattr(
        module,
        "load_production_bundle",
        lambda *_args, **_kwargs: {"manifest": {"manifest_sha256": bundle_hash}},
    )
    campaign = module._hashed(
        {"schema_version": module.CAMPAIGN_RUN_SCHEMA}, "campaign_run_sha256"
    )
    module.atomic_write_json(root / "campaign-intent.json", campaign)
    (attempt / "input.jsonl").write_text("{}\n")
    submission_intent = module._hashed(
        {
            "schema_version": module.SUBMISSION_SCHEMA,
            "input_sha256": module.file_sha256(attempt / "input.jsonl"),
        },
        "submission_intent_sha256",
    )
    module.atomic_write_json(attempt / "submission-intent.json", submission_intent)
    upload = {
        "schema_version": module.UPLOAD_SCHEMA,
        "provider": "openai",
        "input_file_id": "file-1",
        "purpose": "batch",
    }
    module.atomic_write_json(attempt / "provider-upload-response.json", upload)
    metadata = {"campaign": "c", "shard": "shard-000", "generation": "primary"}
    provider = _provider(metadata)
    module.atomic_write_json(attempt / "provider-create-response.json", provider)
    submission = module._hashed(
        {
            "schema_version": module.SUBMISSION_SCHEMA,
            "provider_upload_response_sha256": module.file_sha256(
                attempt / "provider-upload-response.json"
            ),
            "provider_response": provider,
        },
        "submission_sha256",
    )
    module.atomic_write_json(attempt / "submission.json", submission)
    collection_intent = module._hashed(
        {"schema_version": module.COLLECTION_SCHEMA}, "collection_intent_sha256"
    )
    module.atomic_write_json(attempt / "collection-intent.json", collection_intent)
    raw_snapshot = {**provider, "status": "completed"}
    module.atomic_write_json(raw / "provider-snapshot.json", raw_snapshot)
    (raw / "output.jsonl").write_text('{"custom_id":"r"}\n')
    (attempt / "events.jsonl").write_text('{"request_id":"r"}\n')
    raw_bindings = [
        {
            "path": str(path.relative_to(root)),
            "sha256": module.file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in (raw / "provider-snapshot.json", raw / "output.jsonl")
    ]
    collection = module._hashed(
        {
            "schema_version": module.COLLECTION_SCHEMA,
            "collection_intent_sha256": collection_intent[
                "collection_intent_sha256"
            ],
            "events_sha256": module.file_sha256(attempt / "events.jsonl"),
            "raw_file_bindings": raw_bindings,
        },
        "collection_sha256",
    )
    module.atomic_write_json(attempt / "collection.json", collection)
    for name in ("effective-events.jsonl", "proposals.jsonl", "sampling-groups.jsonl"):
        (root / name).write_text("{}\n")
    inventory = module._write_evidence_inventory(root)
    manifest = module._hashed(
        {
            "campaign_run_sha256": campaign["campaign_run_sha256"],
            "bundle_manifest_sha256": bundle_hash,
            "evidence_inventory_sha256": inventory["evidence_inventory_sha256"],
            "effective_events_sha256": module.file_sha256(
                root / "effective-events.jsonl"
            ),
            "proposals_sha256": module.file_sha256(root / "proposals.jsonl"),
            "sampling_groups_sha256": module.file_sha256(
                root / "sampling-groups.jsonl"
            ),
            "collection_bindings": [
                {
                    "generation": "primary",
                    "shard_id": "shard-000",
                    "collection_sha256": collection["collection_sha256"],
                    "events_sha256": collection["events_sha256"],
                }
            ],
        },
        "proposal_bank_manifest_sha256",
    )
    module.atomic_write_json(root / "manifest.json", manifest)
    assert load_frozen_proposal_bank(root)["manifest"] == manifest
    (raw / "output.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError, match="evidence file drift"):
        load_frozen_proposal_bank(root)


def test_terminal_failed_batch_materializes_recovery_events_and_priced_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    shard_root = run / "shards" / "shard-000"
    shard_root.mkdir(parents=True)
    metadata = {
        "campaign": ("c" * 64)[:40],
        "shard": "shard-000",
        "generation": "primary",
    }
    provider = _provider(metadata)
    submission = module._hashed(
        {
            "schema_version": module.SUBMISSION_SCHEMA,
            "status": "submitted",
            "recorded_at": "test",
            "campaign_run_sha256": "c" * 64,
            "submission_intent_sha256": "i" * 64,
            "provider_response": provider,
        },
        "submission_sha256",
    )
    intent = {
        "campaign_run_sha256": "c" * 64,
        "bundle_root": str(tmp_path / "bundle"),
        "maximum_authorized_cost_usd": 10.0,
        "shards": [{"shard_id": "shard-000"}],
    }
    request = {
        "request_id": "request-1",
        "shard_id": "shard-000",
        "window_id": "window-1",
        "window_index": 0,
        "response_id": "response-1",
        "replica_index": 0,
        "body_sha256": "b" * 64,
        "focal_unit_ids": ["unit-1"],
    }
    prices = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "scripts/bonafide/configs/labeling/prices-2026-08-16-coarse-v2.json"
        ).read_text()
    )
    monkeypatch.setattr(module, "_submission", lambda *_: (intent, submission))
    monkeypatch.setattr(module, "iter_shard_requests", lambda *_: [request])
    monkeypatch.setattr(module, "load_price_snapshot", lambda *_: prices)
    failed = {**provider, "status": "failed", "request_counts": {"total": 1, "completed": 0, "failed": 1}}
    result = collect_shard(
        run_root=run,
        shard_id="shard-000",
        downloader=lambda _batch: (failed, {}),
    )
    events = module.read_jsonl(shard_root / "events.jsonl")
    assert result["status"] == "complete_with_failed_requests_recovery_eligible"
    assert result["cost_complete"] is True
    assert result["known_priced_cost_usd"] == 0.0
    assert events[0]["validation_status"] == "missing"
    assert events[0]["pricing_basis"] == "no_provider_result_or_usage_priced_zero"
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    source_input = bundle_root / "shard-000.jsonl"
    source_input.write_text(
        json.dumps({"custom_id": "request-1", "body": {"model": "gpt-5.6-luna"}})
        + "\n"
    )
    bundle = {
        "shards": [
            {
                "shard_id": "shard-000",
                "path": source_input.name,
                "request_ids_in_order": ["request-1"],
            }
        ],
        "config": {
            "empirical_calibration": {
                "source_actual_cost_usd": 0.02,
                "source_request_count": 40,
                "source_input_tokens": 100,
                "source_provider_body_utf8_bytes": 100,
            }
        },
    }
    intent["bundle_root"] = str(bundle_root)
    monkeypatch.setattr(module, "_campaign", lambda *_: (intent, bundle))
    recovery = prepare_failed_only_recovery(run_root=run)
    assert recovery["request_count"] == 1
    assert recovery["shards"][0]["request_ids_in_order"] == ["request-1"]
    recovery_root = run / "recovery-000" / "shards" / "shard-000"
    upload = {
        "schema_version": module.UPLOAD_SCHEMA,
        "provider": "openai",
        "input_file_id": "file-recovery",
        "purpose": "batch",
    }
    module.atomic_write_json(recovery_root / "provider-upload-response.json", upload)
    recovery_provider = {
        **_provider(module._metadata(intent, "shard-000", "recovery-000")),
        "input_file_id": "file-recovery",
        "batch_id": "batch-recovery",
        "status": "completed",
        "output_file_id": "output-recovery",
    }
    recovery_submission = module._hashed(
        {
            "schema_version": module.SUBMISSION_SCHEMA,
            "status": "submitted",
            "recorded_at": "test",
            "campaign_run_sha256": intent["campaign_run_sha256"],
            "submission_intent_sha256": "s" * 64,
            "provider_upload_response_sha256": module.file_sha256(
                recovery_root / "provider-upload-response.json"
            ),
            "provider_response": recovery_provider,
        },
        "submission_sha256",
    )
    module.atomic_write_json(recovery_root / "submission.json", recovery_submission)
    output_row = {
        "custom_id": "request-1",
        "response": {
            "status_code": 200,
            "request_id": "provider-request-1",
            "body": {
                "id": "response-1",
                "model": "gpt-5.6-luna-2026-08-01",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "decisions": [
                                            {
                                                "unit_id": "unit-1",
                                                "tag": "active_task_work",
                                                "confidence": "high",
                                                "boundary_concerns": [],
                                                "boundary_note": "",
                                            }
                                        ]
                                    }
                                ),
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 10,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 5,
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
        },
        "error": None,
    }
    raw_output = (json.dumps(output_row) + "\n").encode()
    recovered = collect_recovery_shard(
        run_root=run,
        shard_id="shard-000",
        downloader=lambda _batch: (
            recovery_provider,
            {
                "output": {
                    "file_id": "output-recovery",
                    "content": raw_output,
                }
            },
        ),
    )
    assert recovered["status"] == "complete"
    assert recovered["cost_complete"] is True
    assert recovered["cumulative_known_priced_cost_usd"] > 0


def test_parser_preserves_replica_and_all_seven_way_decision_fields() -> None:
    request = {
        "request_id": "request-1",
        "shard_id": "shard-000",
        "window_id": "window-1",
        "window_index": 2,
        "response_id": "response-1",
        "replica_index": 1,
        "body_sha256": "b" * 64,
        "focal_unit_ids": ["unit-1"],
    }
    decisions = {
        "decisions": [
            {
                "unit_id": "unit-1",
                "tag": "evaluation_or_revision",
                "confidence": "medium",
                "boundary_concerns": ["meaning_unclear"],
                "boundary_note": "defensible alternative",
            }
        ]
    }
    row = {
        "custom_id": "request-1",
        "response": {
            "status_code": 200,
            "request_id": "provider-1",
            "body": {
                "id": "response-provider-1",
                "model": "gpt-5.6-luna-2026-08-01",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": json.dumps(decisions)}
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 100,
                    "input_tokens_details": {"cached_tokens": 50},
                    "output_tokens": 20,
                    "output_tokens_details": {"reasoning_tokens": 5},
                },
            },
        },
        "error": None,
    }
    event = _parse_row(row, request)
    assert event["validation_status"] == "success"
    assert event["replica_index"] == 1
    assert event["decisions"] == decisions["decisions"]
