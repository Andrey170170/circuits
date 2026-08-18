from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from typing import Any, ClassVar

import circuits.analysis.bonafide.coarse_sampling_openai_batch_production_v1 as module
import pytest
from circuits.analysis.bonafide.coarse_sampling_openai_batch_production_v1 import (
    _copy_campaign_evidence,
    _parse_row,
    _submission_gate,
    authorize_recovery_wave,
    collect_recovery_shard,
    collect_shard,
    initialize_campaign_run,
    load_frozen_proposal_bank,
    prepare_failed_only_recovery,
    recover_recovery_submission,
    recover_shard_submission,
    submit_recovery_shard,
    submit_shard,
)


def _provider(metadata: dict[str, str]) -> dict[str, Any]:
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
                "strict_no_cache_full_output_exposure_usd": 10.0,
                "queued_input_tokens_empirical_forecast": forecast,
            }
        )
    loaded = {
        "manifest": {"manifest_sha256": "m" * 64},
        "cost_plan": {
            "cost_plan_sha256": "c" * 64,
            "strict_no_cache_full_output_exposure_usd": 514.0,
        },
        "shards": shards,
    }
    monkeypatch.setattr(
        module, "load_production_bundle", lambda *args, **kwargs: loaded
    )
    with pytest.raises(ValueError, match="queue limit"):
        initialize_campaign_run(
            bundle_root=bundle_root,
            run_root=tmp_path / "too-small",
            forecast_budget_usd=20,
            forecast_budget_authorization_note="test forecast budget",
            acknowledged_strict_worst_case_exposure_usd=514.0,
            strict_exposure_acknowledgement_note="actual may exceed forecast",
            provider_queued_input_token_limit=100,
            maximum_concurrent_shards=2,
        )
    with pytest.raises(ValueError, match="exact acknowledgement"):
        initialize_campaign_run(
            bundle_root=bundle_root,
            run_root=tmp_path / "wrong-exposure",
            forecast_budget_usd=20,
            forecast_budget_authorization_note="test forecast budget",
            acknowledged_strict_worst_case_exposure_usd=513.0,
            strict_exposure_acknowledgement_note="wrong exposure",
            provider_queued_input_token_limit=110,
            maximum_concurrent_shards=2,
        )
    intent = initialize_campaign_run(
        bundle_root=bundle_root,
        run_root=tmp_path / "run",
        forecast_budget_usd=20,
        forecast_budget_authorization_note="test forecast budget",
        acknowledged_strict_worst_case_exposure_usd=514.0,
        strict_exposure_acknowledgement_note="actual may exceed forecast",
        provider_queued_input_token_limit=110,
        maximum_concurrent_shards=2,
    )
    assert intent["maximum_concurrent_shards"] == 2
    assert intent["provider_queued_input_token_limit"] == 110
    assert intent["network_calls_made"] == 0
    assert intent["forecast_budget_is_hard_spend_cap"] is False
    assert intent["strict_worst_case_exposure_usd"] == 514.0


def test_runtime_environment_rejects_sdk_and_provider_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_PROJECT_ID", raising=False)
    monkeypatch.delenv("OPENAI_ORG_ID", raising=False)
    current = {
        "python_version": module.platform.python_version(),
        "openai_sdk_version": module.importlib.metadata.version("openai"),
        "openai_project_sha256": None,
        "openai_organization_sha256": None,
    }
    module._validate_runtime_environment(current)
    with pytest.raises(ValueError, match="runtime environment drift"):
        module._validate_runtime_environment(
            {**current, "openai_sdk_version": "different-sdk"}
        )
    monkeypatch.setenv("OPENAI_PROJECT_ID", "different-project")
    with pytest.raises(ValueError, match="runtime environment drift"):
        module._validate_runtime_environment(current)


def test_submit_persists_intent_and_fails_closed_on_ambiguous_provider_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    shard_root = run / "shards" / "shard-000"
    shard_root.mkdir(parents=True)
    (shard_root / "input.jsonl").write_text('{"body":{}}\n', encoding="utf-8")
    intent = {
        "campaign_run_sha256": "c" * 64,
        "bundle_root": str(tmp_path / "bundle"),
        "forecast_budget_usd": 20.0,
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
    production_config = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "scripts/bonafide/configs/process_witness_coarse_production_v1.json"
        ).read_text()
    )
    bundle = {
        "shards": [
            {
                "shard_id": "shard-000",
                "request_count": 1,
            }
        ],
        "config": production_config,
    }
    monkeypatch.setattr(module, "_campaign", lambda _: (intent, bundle))
    monkeypatch.setattr(
        module, "load_production_bundle", lambda *_args, **_kwargs: bundle
    )

    def upload(_path: Path) -> dict[str, Any]:
        return {
            "schema_version": module.UPLOAD_SCHEMA,
            "provider": "openai",
            "input_file_id": "file-1",
            "purpose": "batch",
        }

    def fail(*args, **kwargs):
        raise TimeoutError("create response lost")

    with pytest.raises(RuntimeError, match="indeterminate"):
        submit_shard(run_root=run, shard_id="shard-000", uploader=upload, creator=fail)
    assert (shard_root / "submission-intent.json").is_file()
    assert (shard_root / "provider-upload-response.json").is_file()
    failure = json.loads((shard_root / "submission-failure.json").read_text())
    assert failure["automatic_retry_permitted"] is False
    assert failure["upload_receipt_persisted"] is True
    recovered = recover_shard_submission(
        run_root=run,
        shard_id="shard-000",
        discoverer=lambda **_: [
            _provider(module._metadata(intent, "shard-000", "primary"))
        ],
    )
    assert recovered["recovered_by"] == "unique_provider_metadata_discovery"
    assert recovered["provider_response"]["input_file_id"] == "file-1"
    assert recovered["provider_upload_response_sha256"] == module.file_sha256(
        shard_root / "provider-upload-response.json"
    )
    _, loaded = module._submission(run, "shard-000")
    assert loaded["submission_sha256"] == recovered["submission_sha256"]
    completed = {**recovered["provider_response"], "status": "completed"}
    status = module.check_shard(
        run_root=run,
        shard_id="shard-000",
        retriever=lambda _batch: completed,
    )
    assert status["provider_response"]["status"] == "completed"
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
    monkeypatch.setattr(module, "iter_shard_requests", lambda *_: [request])
    monkeypatch.setattr(module, "load_price_snapshot", lambda *_: prices)
    with pytest.raises(TimeoutError, match="download interrupted"):
        collect_shard(
            run_root=run,
            shard_id="shard-000",
            downloader=lambda _batch: (_ for _ in ()).throw(
                TimeoutError("download interrupted")
            ),
        )
    assert (shard_root / "collection-intent.json").is_file()
    completed_with_output = {
        **completed,
        "output_file_id": "output-1",
    }
    raw_error = (
        json.dumps(
            {
                "custom_id": "request-1",
                "error": {"code": "provider_error", "message": "typed failure"},
            }
        )
        + "\n"
    ).encode()
    original_parse = module._parse_row
    monkeypatch.setattr(
        module,
        "_parse_row",
        lambda *_: (_ for _ in ()).throw(RuntimeError("parse interrupted")),
    )
    with pytest.raises(RuntimeError, match="parse interrupted"):
        collect_shard(
            run_root=run,
            shard_id="shard-000",
            downloader=lambda _batch: (
                completed_with_output,
                {"output": {"file_id": "output-1", "content": raw_error}},
            ),
        )
    assert (shard_root / "raw" / "provider-snapshot.json").is_file()
    assert (shard_root / "raw" / "output.jsonl").read_bytes() == raw_error
    monkeypatch.setattr(module, "_parse_row", original_parse)
    collection = collect_shard(
        run_root=run,
        shard_id="shard-000",
        downloader=lambda _batch: (
            completed_with_output,
            {"output": {"file_id": "output-1", "content": raw_error}},
        ),
    )
    assert collection["cost_complete"] is False
    assert collection["failure_count"] == 1


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
        "forecast_budget_usd": 1.5,
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
        "forecast_budget_usd": 10.0,
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


def test_submit_refuses_while_collection_materialization_is_reserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    for shard_id in ("shard-000", "shard-001"):
        root = run / "shards" / shard_id
        root.mkdir(parents=True)
        (root / "input.jsonl").write_text("{}\n")
    (run / "shards" / "shard-000" / ".collection-gate").mkdir()
    intent = {
        "campaign_run_sha256": "c" * 64,
        "forecast_budget_usd": 10.0,
        "maximum_concurrent_shards": 2,
        "provider_queued_input_token_limit": 1_000,
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
    with pytest.raises(RuntimeError, match="collection materialization is active"):
        submit_shard(
            run_root=run,
            shard_id="shard-001",
            uploader=lambda _: (_ for _ in ()).throw(AssertionError("uploaded")),
        )


def test_submission_gate_is_exclusive(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    with (
        _submission_gate(run),
        pytest.raises(RuntimeError, match="already held or stale"),
        _submission_gate(run),
    ):
        pass
    assert not (run / ".submission-gate").exists()


def test_simultaneous_collect_is_locked_through_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    (run / "shards" / "shard-000").mkdir(parents=True)
    started = threading.Event()
    release = threading.Event()
    errors = []

    def materialize(**_kwargs):
        started.set()
        assert release.wait(5)
        return {"status": "complete"}

    monkeypatch.setattr(module, "_collect_shard_locked", materialize)

    def first() -> None:
        try:
            collect_shard(run_root=run, shard_id="shard-000")
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=first)
    thread.start()
    assert started.wait(5)
    assert module._active_collection_locks(run) == ["shards/shard-000/.collection-gate"]
    with pytest.raises(RuntimeError, match="collection gate is already held"):
        collect_shard(run_root=run, shard_id="shard-000")
    release.set()
    thread.join(5)
    assert not thread.is_alive()
    assert errors == []
    assert module._active_collection_locks(run) == []


def test_simultaneous_same_shard_submit_calls_provider_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    attempt = run / "shards" / "shard-000"
    attempt.mkdir(parents=True)
    (attempt / "input.jsonl").write_text("{}\n")
    intent = {
        "campaign_run_sha256": "c" * 64,
        "forecast_budget_usd": 2.0,
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

    def upload(_path: Path) -> dict[str, Any]:
        calls.append("upload")
        started.set()
        assert release.wait(5)
        return {
            "schema_version": module.UPLOAD_SCHEMA,
            "provider": "openai",
            "input_file_id": "file-1",
            "purpose": "batch",
        }

    def creator(_file_id: str, *, metadata: dict[str, str]) -> dict[str, Any]:
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
        {
            "schema_version": module.CAMPAIGN_RUN_SCHEMA,
            "status": "initialized_no_provider_calls",
            "forecast_budget_usd": 1.0,
            "forecast_budget_is_hard_spend_cap": False,
            "strict_worst_case_exposure_usd": 10.0,
            "acknowledged_strict_worst_case_exposure_usd": 10.0,
            "strict_exposure_acknowledgement_note": "test strict exposure",
        },
        "campaign_run_sha256",
    )
    module.atomic_write_json(root / "campaign-intent.json", campaign)
    (attempt / "input.jsonl").write_text("{}\n")
    submission_intent = module._hashed(
        {
            "schema_version": module.SUBMISSION_SCHEMA,
            "generation": "primary",
            "metadata": module._metadata(campaign, "shard-000", "primary"),
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
    metadata = module._metadata(campaign, "shard-000", "primary")
    provider = _provider(metadata)
    module.atomic_write_json(attempt / "provider-create-response.json", provider)
    submission = module._hashed(
        {
            "schema_version": module.SUBMISSION_SCHEMA,
            "status": "submitted",
            "campaign_run_sha256": campaign["campaign_run_sha256"],
            "submission_intent_sha256": submission_intent["submission_intent_sha256"],
            "provider_upload_response_sha256": module.file_sha256(
                attempt / "provider-upload-response.json"
            ),
            "provider_response": provider,
        },
        "submission_sha256",
    )
    module.atomic_write_json(attempt / "submission.json", submission)
    status = module._hashed(
        {
            "schema_version": module.STATUS_SCHEMA,
            "campaign_run_sha256": campaign["campaign_run_sha256"],
            "submission_sha256": submission["submission_sha256"],
            "previous_status_sha256": None,
            "provider_response": provider,
        },
        "status_sha256",
    )
    (attempt / "status").mkdir()
    module.atomic_write_json(attempt / "status" / "receipt-0000.json", status)
    collection_intent = module._hashed(
        {
            "schema_version": module.COLLECTION_SCHEMA,
            "submission_sha256": submission["submission_sha256"],
            "batch_id": provider["batch_id"],
        },
        "collection_intent_sha256",
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
            "status": "complete",
            "cost_complete": True,
            "known_priced_cost_usd": 0.0,
            "provider_terminal_status": "completed",
            "collection_intent_sha256": collection_intent["collection_intent_sha256"],
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
            "schema_version": "adag.process-witness.coarse-proposal-bank.v1",
            "status": "frozen_sampling_proposals_not_semantic_truth",
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
            "primary_actual_cost_usd": 0.0,
            "primary_forecast_budget_usd": 1.0,
            "primary_forecast_budget_is_hard_spend_cap": False,
            "primary_strict_worst_case_exposure_usd": 10.0,
            "recovery_actual_cost_usd": 0.0,
            "actual_total_cost_usd": 0.0,
            "recovery_authorization": None,
        },
        "proposal_bank_manifest_sha256",
    )
    module.atomic_write_json(root / "manifest.json", manifest)
    module._readonly_tree(root)
    assert load_frozen_proposal_bank(root)["manifest"] == manifest
    (raw / "output.jsonl").chmod(0o644)
    with pytest.raises(ValueError, match="mode drift"):
        load_frozen_proposal_bank(root)
    (raw / "output.jsonl").chmod(0o444)
    (raw / "output.jsonl").chmod(0o644)
    (raw / "output.jsonl").write_text("tampered\n")
    (raw / "output.jsonl").chmod(0o444)
    with pytest.raises(ValueError, match="evidence file drift"):
        load_frozen_proposal_bank(root)


def test_terminal_failed_batch_materializes_recovery_events_and_priced_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    shard_root = run / "shards" / "shard-000"
    shard_root.mkdir(parents=True)
    (shard_root / "input.jsonl").write_text('{"body":{}}\n')
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
        "forecast_budget_usd": 10.0,
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
    config = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "scripts/bonafide/configs/process_witness_coarse_production_v1.json"
        ).read_text()
    )
    monkeypatch.setattr(module, "_submission", lambda *_: (intent, submission))
    monkeypatch.setattr(module, "iter_shard_requests", lambda *_: [request])
    monkeypatch.setattr(module, "load_price_snapshot", lambda *_: prices)
    monkeypatch.setattr(
        module,
        "load_production_bundle",
        lambda *_args, **_kwargs: {"config": config},
    )
    failed = {
        **provider,
        "status": "failed",
        "request_counts": {"total": 1, "completed": 0, "failed": 1},
        "usage": {
            "input_tokens": 0,
            "input_tokens_details": {
                "cached_tokens": 0,
                "cache_write_tokens": 0,
            },
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "errors": {"data": [{"code": "invalid_request", "message": "rejected"}]},
    }
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
    assert (
        events[0]["collection_pricing_basis"]
        == "aggregate_batch_usage_with_per_request_byte_upper_bound"
    )
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    source_input = bundle_root / "shard-000.jsonl"
    source_input.write_text(
        json.dumps({"custom_id": "request-1", "body": {"model": "gpt-5.6-luna"}}) + "\n"
    )
    bundle = {
        "shards": [
            {
                "shard_id": "shard-000",
                "path": source_input.name,
                "request_ids_in_order": ["request-1"],
            }
        ],
        "config": config,
    }
    intent["bundle_root"] = str(bundle_root)
    monkeypatch.setattr(module, "_campaign", lambda *_: (intent, bundle))
    recovery = prepare_failed_only_recovery(run_root=run)
    assert recovery["request_count"] == 1
    assert recovery["shards"][0]["request_ids_in_order"] == ["request-1"]
    authorization = authorize_recovery_wave(
        run_root=run,
        recovery_forecast_budget_usd=1.0,
        forecast_budget_authorization_note="test recovery forecast budget",
        acknowledged_strict_worst_case_exposure_usd=recovery[
            "strict_no_cache_full_output_exposure_usd"
        ],
        strict_exposure_acknowledgement_note="actual recovery may exceed forecast",
    )
    assert authorization["forecast_budget_is_hard_spend_cap"] is False
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
    with pytest.raises(TimeoutError, match="recovery download interrupted"):
        collect_recovery_shard(
            run_root=run,
            shard_id="shard-000",
            downloader=lambda _batch: (_ for _ in ()).throw(
                TimeoutError("recovery download interrupted")
            ),
        )
    assert (recovery_root / "collection-intent.json").is_file()
    original_write_jsonl = module._write_or_verify_jsonl
    monkeypatch.setattr(
        module,
        "_write_or_verify_jsonl",
        lambda *_: (_ for _ in ()).throw(RuntimeError("event publish interrupted")),
    )
    with pytest.raises(RuntimeError, match="event publish interrupted"):
        collect_recovery_shard(
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
    assert (recovery_root / "raw" / "output.jsonl").read_bytes() == raw_output
    monkeypatch.setattr(module, "_write_or_verify_jsonl", original_write_jsonl)
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


@pytest.mark.parametrize(
    ("status", "counts", "errors"),
    [
        (
            "expired",
            {"total": 1, "completed": 0, "failed": 1},
            {"data": [{"code": "expired", "message": "expired"}]},
        ),
        (
            "cancelled",
            {"total": 1, "completed": 0, "failed": 1},
            {"data": [{"code": "cancelled", "message": "cancelled"}]},
        ),
        (
            "failed",
            {"total": 1, "completed": 0, "failed": 0},
            {"data": [{"code": "invalid", "message": "invalid"}]},
        ),
        ("failed", {"total": 1, "completed": 0, "failed": 1}, None),
    ],
)
def test_zero_aggregate_cost_requires_proven_preexecution_failure(
    status: str,
    counts: dict[str, int],
    errors: dict[str, Any] | None,
) -> None:
    prices = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "scripts/bonafide/configs/labeling/prices-2026-08-16-coarse-v2.json"
        ).read_text()
    )
    event = {
        "request_id": "request-1",
        "validation_status": "missing",
        "usage": module.openai_usage(None).model_dump(mode="json"),
    }
    snapshot = {
        "status": status,
        "request_counts": counts,
        "errors": errors,
        "usage": {
            "input_tokens": 0,
            "input_tokens_details": {
                "cached_tokens": 0,
                "cache_write_tokens": 0,
            },
            "output_tokens": 0,
        },
    }
    _total, complete, basis = module._price_events(
        events=[event],
        snapshot=snapshot,
        prices=prices,
        aggregate_fallback_long_context_impossible=True,
    )
    assert complete is False
    assert basis == "cost_incomplete_per_request_and_aggregate_usage"


def test_missing_upload_receipt_recovery_requires_zero_batches_then_reuploads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    shard_root = run / "shards" / "shard-000"
    shard_root.mkdir(parents=True)
    (shard_root / "input.jsonl").write_text("{}\n")
    intent = {"campaign_run_sha256": "c" * 64}
    monkeypatch.setattr(module, "_campaign", lambda *_: (intent, {"shards": []}))
    submission_intent = module._hashed(
        {
            "schema_version": module.SUBMISSION_SCHEMA,
            "generation": "primary",
            "input_sha256": module.file_sha256(shard_root / "input.jsonl"),
        },
        "submission_intent_sha256",
    )
    module.atomic_write_json(shard_root / "submission-intent.json", submission_intent)
    uploaded = {
        "schema_version": module.UPLOAD_SCHEMA,
        "provider": "openai",
        "input_file_id": "file-reuploaded",
        "purpose": "batch",
    }

    def creator(file_id: str, *, metadata: dict[str, str]) -> dict[str, Any]:
        return {**_provider(metadata), "input_file_id": file_id}

    recovered = recover_shard_submission(
        run_root=run,
        shard_id="shard-000",
        discoverer=lambda **_: [],
        uploader=lambda _: uploaded,
        creator=creator,
    )
    assert (
        recovered["recovered_by"] == "safe_reupload_after_zero_batch_metadata_matches"
    )
    assert recovered["provider_upload_response_sha256"] == module.file_sha256(
        shard_root / "provider-upload-response.json"
    )
    assert (shard_root / "orphan-upload-state.json").is_file()
    orphan = json.loads((shard_root / "orphan-upload-state.json").read_text())
    assert orphan["failure_receipt_present"] is False
    _, verified = module._submission(run, "shard-000")
    assert verified["submission_sha256"] == recovered["submission_sha256"]


def test_recovery_missing_upload_receipt_uses_same_safe_reupload_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    shard_root = run / "recovery-000" / "shards" / "shard-000"
    shard_root.mkdir(parents=True)
    (shard_root / "input.jsonl").write_text("{}\n")
    intent = {"campaign_run_sha256": "c" * 64}
    binding = {"input_sha256": module.file_sha256(shard_root / "input.jsonl")}
    monkeypatch.setattr(
        module,
        "_recovery_binding",
        lambda *_: (intent, {}, binding, shard_root),
    )
    submission_intent = module._hashed(
        {
            "schema_version": module.SUBMISSION_SCHEMA,
            "generation": "recovery-000",
            "input_sha256": binding["input_sha256"],
        },
        "submission_intent_sha256",
    )
    failure = module._hashed(
        {
            "schema_version": module.SUBMISSION_SCHEMA,
            "submission_intent_sha256": submission_intent["submission_intent_sha256"],
            "upload_receipt_persisted": False,
        },
        "submission_failure_sha256",
    )
    module.atomic_write_json(shard_root / "submission-intent.json", submission_intent)
    module.atomic_write_json(shard_root / "submission-failure.json", failure)
    uploaded = {
        "schema_version": module.UPLOAD_SCHEMA,
        "provider": "openai",
        "input_file_id": "file-recovery-reuploaded",
        "purpose": "batch",
    }

    def creator(file_id: str, *, metadata: dict[str, str]) -> dict[str, Any]:
        return {**_provider(metadata), "input_file_id": file_id}

    recovered = recover_recovery_submission(
        run_root=run,
        shard_id="shard-000",
        discoverer=lambda **_: [],
        uploader=lambda _: uploaded,
        creator=creator,
    )
    assert (
        recovered["recovered_by"] == "safe_reupload_after_zero_batch_metadata_matches"
    )
    assert recovered["provider_upload_response_sha256"] == module.file_sha256(
        shard_root / "provider-upload-response.json"
    )


@pytest.mark.parametrize(
    ("recovery_budget", "queue_limit", "message"),
    [
        (1.5, 1_000, "actual-or-reserved recovery cost"),
        (10.0, 150, "queued-input-token capacity"),
    ],
)
def test_recovery_submit_reserves_cross_shard_budget_and_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recovery_budget: float,
    queue_limit: int,
    message: str,
) -> None:
    run = tmp_path / "run"
    recovery_root = run / "recovery-000"
    shard_rows = []
    for shard_id in ("shard-000", "shard-001"):
        shard_root = recovery_root / "shards" / shard_id
        shard_root.mkdir(parents=True)
        (shard_root / "input.jsonl").write_text("{}\n")
        shard_rows.append(
            {
                "shard_id": shard_id,
                "input_sha256": module.file_sha256(shard_root / "input.jsonl"),
                "request_count": 1,
                "direct_v4_cost_forecast_usd": 1.0,
                "strict_no_cache_full_output_exposure_usd": 10.0,
                "queued_input_tokens_empirical_forecast": 100,
            }
        )
    (recovery_root / "shards" / "shard-000" / "submission-intent.json").write_text(
        "{}\n"
    )
    intent = {
        "campaign_run_sha256": "c" * 64,
        "maximum_concurrent_shards": 2,
        "provider_queued_input_token_limit": queue_limit,
    }
    manifest = module._hashed(
        {
            "schema_version": "adag.process-witness.coarse-production-recovery.v1",
            "campaign_run_sha256": intent["campaign_run_sha256"],
            "strict_no_cache_full_output_exposure_usd": 20.0,
            "shards": shard_rows,
        },
        "recovery_manifest_sha256",
    )
    authorization = module._hashed(
        {
            "schema_version": "adag.process-witness.coarse-production-recovery-authorization.v1",
            "campaign_run_sha256": intent["campaign_run_sha256"],
            "recovery_manifest_sha256": manifest["recovery_manifest_sha256"],
            "recovery_forecast_budget_usd": recovery_budget,
            "forecast_budget_authorization_note": "test recovery forecast",
            "forecast_budget_is_hard_spend_cap": False,
            "strict_worst_case_exposure_usd": 20.0,
            "acknowledged_strict_worst_case_exposure_usd": 20.0,
            "strict_exposure_acknowledgement_note": "test recovery strict exposure",
        },
        "recovery_authorization_sha256",
    )
    module.atomic_write_json(recovery_root / "manifest.json", manifest)
    module.atomic_write_json(recovery_root / "authorization.json", authorization)
    target_root = recovery_root / "shards" / "shard-001"
    monkeypatch.setattr(
        module,
        "_recovery_binding",
        lambda *_: (intent, {}, shard_rows[1], target_root),
    )
    monkeypatch.setattr(module, "_attempted_primary_forecast", lambda *_: (1.0, True))
    with pytest.raises(ValueError, match=message):
        submit_recovery_shard(
            run_root=run,
            shard_id="shard-001",
            uploader=lambda _: (_ for _ in ()).throw(AssertionError("uploaded")),
        )


def test_recovery_authorization_self_hash_is_not_enough() -> None:
    intent = {"campaign_run_sha256": "c" * 64}
    manifest = module._hashed(
        {
            "campaign_run_sha256": intent["campaign_run_sha256"],
            "strict_no_cache_full_output_exposure_usd": 20.0,
        },
        "recovery_manifest_sha256",
    )
    incomplete = module._hashed(
        {
            "schema_version": "test-authorization",
            "campaign_run_sha256": intent["campaign_run_sha256"],
            "recovery_manifest_sha256": manifest["recovery_manifest_sha256"],
            "recovery_forecast_budget_usd": 1.0,
        },
        "recovery_authorization_sha256",
    )
    with pytest.raises(ValueError, match="semantic drift"):
        module._validate_recovery_authorization(
            authorization=incomplete,
            manifest=manifest,
            intent=intent,
        )


def test_actual_cost_cannot_exceed_separately_acknowledged_strict_exposure() -> None:
    module._enforce_actual_within_strict_exposure(
        actual_cost_usd=10.0,
        strict_exposure_usd=10.0,
        generation="primary",
    )
    with pytest.raises(ValueError, match="recovery actual cost exceeds"):
        module._enforce_actual_within_strict_exposure(
            actual_cost_usd=10.01,
            strict_exposure_usd=10.0,
            generation="recovery",
        )


def test_batch_aggregate_above_long_context_threshold_is_priced_per_request() -> None:
    prices = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "scripts/bonafide/configs/labeling/prices-2026-08-16-coarse-v2.json"
        ).read_text()
    )
    usage = {
        "input_tokens": 200_000,
        "uncached_input_tokens": 94_838,
        "cache_read_tokens": 0,
        "cache_write_tokens": 105_162,
        "output_tokens": 10,
        "reasoning_tokens": 0,
    }
    events: list[dict[str, Any]] = [
        {
            "request_id": f"request-{index}",
            "validation_status": "success",
            "usage": usage,
        }
        for index in range(2)
    ]
    snapshot = {
        "status": "completed",
        "usage": {
            "input_tokens": 400_000,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    total, complete, basis = module._price_events(
        events=events, snapshot=snapshot, prices=prices
    )
    assert complete is True
    assert basis == "per_request_usage_reconciled_to_batch_aggregate"
    assert all(
        event["long_context_price_multiplier_applied"] is False for event in events
    )
    assert total == pytest.approx(sum(event["cost"]["total_cost"] for event in events))


def test_aggregate_fallback_rejects_usage_below_known_request_rows() -> None:
    prices = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "scripts/bonafide/configs/labeling/prices-2026-08-16-coarse-v2.json"
        ).read_text()
    )
    events = [
        {
            "request_id": "known",
            "validation_status": "success",
            "usage": {
                "input_tokens": 100,
                "uncached_input_tokens": 50,
                "cache_read_tokens": 20,
                "cache_write_tokens": 30,
                "output_tokens": 10,
                "reasoning_tokens": 0,
            },
        },
        {
            "request_id": "missing",
            "validation_status": "missing",
            "usage": module.openai_usage(None).model_dump(mode="json"),
        },
    ]
    snapshot = {
        "status": "completed",
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {
                "cached_tokens": 2,
                "cache_write_tokens": 3,
            },
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    _, complete, basis = module._price_events(
        events=events,
        snapshot=snapshot,
        prices=prices,
        aggregate_fallback_long_context_impossible=True,
    )
    assert complete is False
    assert basis == "failed_closed_batch_aggregate_usage_below_known_rows"


def test_aggregate_fallback_rejects_derived_uncached_below_known_rows() -> None:
    prices = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "scripts/bonafide/configs/labeling/prices-2026-08-16-coarse-v2.json"
        ).read_text()
    )
    events = [
        {
            "request_id": "known",
            "validation_status": "success",
            "usage": {
                "input_tokens": 100,
                "uncached_input_tokens": 100,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "output_tokens": 10,
                "reasoning_tokens": 0,
            },
        },
        {
            "request_id": "missing",
            "validation_status": "missing",
            "usage": module.openai_usage(None).model_dump(mode="json"),
        },
    ]
    snapshot = {
        "status": "completed",
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": {
                "cached_tokens": 100,
                "cache_write_tokens": 0,
            },
            "output_tokens": 10,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    _, complete, basis = module._price_events(
        events=events,
        snapshot=snapshot,
        prices=prices,
        aggregate_fallback_long_context_impossible=True,
    )
    assert complete is False
    assert basis == "failed_closed_batch_aggregate_usage_below_known_rows"


def test_parser_preserves_typed_provider_error_message() -> None:
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
    event = _parse_row(
        {
            "custom_id": "request-1",
            "error": {"code": "invalid_request", "message": "exact provider detail"},
        },
        request,
    )
    assert event["provider_error_code"] == "invalid_request"
    assert event["error_message"] == "exact provider detail"


def test_batch_snapshot_preserves_raw_model_dump_and_errors() -> None:
    class Batch:
        id = "batch-1"
        input_file_id = "file-1"
        endpoint = "/v1/responses"
        completion_window = "24h"
        metadata: ClassVar[dict[str, str]] = {
            "campaign": "c",
            "shard": "s",
            "generation": "primary",
        }
        status = "failed"
        output_file_id = None
        error_file_id = None
        request_counts = None
        model = None
        usage = None
        errors: ClassVar[dict[str, list[dict[str, str]]]] = {
            "data": [{"code": "invalid", "message": "bad input"}]
        }

        def model_dump(self, *, mode: str) -> dict[str, Any]:
            assert mode == "json"
            return {
                "id": self.id,
                "status": self.status,
                "errors": self.errors,
                "created_at": 1,
            }

    snapshot = module._production_provider_batch_dict(Batch())
    assert snapshot["errors"] == Batch.errors
    assert snapshot["provider_model_dump"]["created_at"] == 1


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
