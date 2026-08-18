from __future__ import annotations

import json
from pathlib import Path

import circuits.analysis.bonafide.coarse_sampling_openai_batch_continuation_v1 as module
import pytest


def _line(request_id: str, payload: str) -> bytes:
    return (
        json.dumps(
            {
                "custom_id": request_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": {"input": payload},
            },
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _simple_attempt_manifest(*, reservation: float = 1.0) -> dict[str, object]:
    return {
        "continuation_manifest_sha256": "m" * 64,
        "maximum_concurrent_attempts": 1,
        "provider_queued_input_token_limit": 40_000_000,
        "warning_spend_threshold_usd": 20.0,
        "hard_campaign_stop_usd": 40.0,
        "calibration_known_priced_cost_usd": 4.0,
        "attempts": [
            {
                "attempt_id": "primary-tranche-000",
                "generation": "continuation-primary",
                "input_relative_path": "attempts/primary-tranche-000/input.jsonl",
                "input_sha256": "i" * 64,
                "request_count": 1,
                "request_ids_in_order": ["request-0"],
                "queued_input_tokens_empirical_forecast": 10,
                "calibrated_cost_reservation_usd": reservation,
            }
        ],
    }


def _provider_snapshot(
    *, metadata: dict[str, str], input_file_id: str = "file-1"
) -> dict[str, object]:
    return {
        "schema_version": "adag.labeling.provider-batch.v1",
        "provider": "openai",
        "batch_id": "batch-1",
        "input_file_id": input_file_id,
        "endpoint": "/v1/responses",
        "completion_window": "24h",
        "metadata": metadata,
        "status": "validating",
    }


def _write_ambiguous_submission_state(
    root: Path, *, manifest: dict[str, object], retain_create: bool
) -> dict[str, object]:
    attempt_root = root / "attempts/primary-tranche-000"
    attempt_root.mkdir(parents=True)
    metadata = {
        "campaign": str(manifest["continuation_manifest_sha256"])[:40],
        "shard": "primary-tranche-000",
        "generation": "continuation-primary",
    }
    intent = module._hashed(
        {
            "schema_version": module.production_v1.SUBMISSION_SCHEMA,
            "status": "intent_persisted_before_provider_calls",
            "metadata": metadata,
        },
        "submission_intent_sha256",
    )
    upload = {
        "schema_version": module.production_v1.UPLOAD_SCHEMA,
        "provider": "openai",
        "input_file_id": "file-1",
        "purpose": "batch",
    }
    (attempt_root / "submission-intent.json").write_text(json.dumps(intent))
    (attempt_root / "provider-upload-response.json").write_text(json.dumps(upload))
    failure = module._hashed(
        {
            "schema_version": module.production_v1.SUBMISSION_SCHEMA,
            "status": "failed_closed_indeterminate_provider_state",
            "continuation_manifest_sha256": manifest[
                "continuation_manifest_sha256"
            ],
            "submission_intent_sha256": intent["submission_intent_sha256"],
            "metadata": metadata,
            "provider_upload_response_sha256": module.file_sha256(
                attempt_root / "provider-upload-response.json"
            ),
        },
        "submission_failure_sha256",
    )
    (attempt_root / "submission-failure.json").write_text(json.dumps(failure))
    provider = _provider_snapshot(metadata=metadata)
    if retain_create:
        (attempt_root / "provider-create-response.json").write_text(
            json.dumps(provider)
        )
    return provider


def test_prepare_continuation_retries_only_failed_and_repacks_remaining_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_root = tmp_path / "bundle"
    calibration_root = tmp_path / "calibration"
    run_root = tmp_path / "continuation"
    (bundle_root / "batch-shards").mkdir(parents=True)
    (calibration_root / "shards/shard-005").mkdir(parents=True)
    rows = {
        "ok-005": _line("ok-005", "calibration success"),
        "bad-a": _line("bad-a", "calibration invalid"),
        "bad-b": _line("bad-b", "calibration provider error"),
        "r0-a": _line("r0-a", "first response"),
        "r0-b": _line("r0-b", "first response replica"),
        "r1-a": _line("r1-a", "second response"),
    }
    (bundle_root / "batch-shards/shard-005.jsonl").write_bytes(
        rows["ok-005"] + rows["bad-a"] + rows["bad-b"]
    )
    (bundle_root / "batch-shards/shard-000.jsonl").write_bytes(
        rows["r0-a"] + rows["r0-b"] + rows["r1-a"]
    )
    events = [
        {"request_id": "ok-005", "validation_status": "success"},
        {"request_id": "bad-a", "validation_status": "invalid_output"},
        {"request_id": "bad-b", "validation_status": "provider_error"},
    ]
    (calibration_root / "shards/shard-005/events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
    )
    (calibration_root / "shards/shard-005/collection.json").write_text(
        json.dumps(
            {
                "collection_sha256": "collection",
                "events_sha256": module.file_sha256(
                    calibration_root / "shards/shard-005/events.jsonl"
                ),
                "request_count": 3,
                "success_count": 1,
                "failure_count": 2,
                "known_priced_cost_usd": 1.25,
                "cost_complete": False,
            }
        ),
        encoding="utf-8",
    )
    request_index = [
        {
            "request_id": request_id,
            "shard_id": shard_id,
            "response_id": response_id,
            "window_id": f"w-{request_id}",
            "window_index": index,
            "replica_index": index % 3,
            "body_sha256": "b" * 64,
            "focal_unit_ids": [f"u-{request_id}"],
        }
        for index, (request_id, shard_id, response_id) in enumerate(
            (
                ("ok-005", "shard-005", "response-cal"),
                ("bad-a", "shard-005", "response-cal"),
                ("bad-b", "shard-005", "response-cal"),
                ("r0-a", "shard-000", "response-0"),
                ("r0-b", "shard-000", "response-0"),
                ("r1-a", "shard-000", "response-1"),
            )
        )
    ]
    bundle = {
        "manifest": {"manifest_sha256": "m" * 64},
        "config": {
            "empirical_calibration": {
                "source_input_tokens": 100,
                "source_provider_body_utf8_bytes": 100,
                "source_actual_cost_usd": 1.0,
                "source_request_count": 10,
            },
            "provider": {"input_token_overhead_per_request": 10},
        },
        "cost_plan": {"cost_plan_sha256": "c" * 64},
        "shards": [
            {
                "shard_id": "shard-000",
                "path": "batch-shards/shard-000.jsonl",
                "request_ids_in_order": ["r0-a", "r0-b", "r1-a"],
            },
            {
                "shard_id": "shard-005",
                "path": "batch-shards/shard-005.jsonl",
                "request_ids_in_order": ["ok-005", "bad-a", "bad-b"],
            },
        ],
        "request_index": request_index,
    }
    calibration_intent = {
        "campaign_run_sha256": "i" * 64,
        "bundle_manifest_sha256": "m" * 64,
        "authorized_primary_shard_ids": ["shard-005"],
        "provider_queued_input_token_limit": 40_000_000,
    }
    monkeypatch.setattr(module, "load_production_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(
        module,
        "_validated_calibration_source",
        lambda **_: (calibration_intent, bundle),
    )
    monkeypatch.setattr(module, "_copy_calibration_evidence", lambda **_: "tree-hash")
    monkeypatch.setattr(
        module,
        "_cost_metrics",
        lambda **kwargs: {
            "direct_v4_cost_forecast_usd": len(kwargs["rows"]) / 10,
            "strict_no_cache_full_output_exposure_usd": len(kwargs["rows"]),
            "calibrated_cost_reservation_usd": len(kwargs["rows"]) / 5,
        },
    )
    monkeypatch.setattr(
        module,
        "_reconcile_inherited_calibration_cost",
        lambda **_: {
            "cost_reconciliation_sha256": "r" * 64,
            "cost_complete": True,
        },
    )
    monkeypatch.setattr(
        module,
        "_current_source_revision",
        lambda **_: {"git_head": "h" * 40, "files": []},
    )
    monkeypatch.setattr(module, "_load_continuation", lambda _: ({}, bundle))

    manifest = module.prepare_continuation(
        bundle_root=bundle_root,
        calibration_run_root=calibration_root,
        run_root=run_root,
        provider_queued_input_token_limit=40_000_000,
        tranche_empirical_queue_cap=100,
        maximum_concurrent_attempts=1,
        authorized_forecast_budget_usd=20.0,
        warning_spend_threshold_usd=20.0,
        hard_campaign_stop_usd=40.0,
        authorization_note="user authorized full labeling below USD 100",
        calibration_observed_input_tokens=101,
        calibration_forecast_input_tokens=100,
    )

    assert manifest["calibration_failure_count"] == 2
    assert manifest["calibration_success_count"] == 1
    assert manifest["calibration_queue_actual_to_forecast_ratio"] == 1.01
    assert manifest["queue_rejection_policy"] == "receipt_bound_stop_no_further_submissions"
    assert manifest["inherited_failure_request_ids"] == ["bad-a", "bad-b"]
    assert not (run_root / "attempts/calibration-recovery-000").exists()
    tranche_bytes = b"".join(
        (run_root / item["input_relative_path"]).read_bytes()
        for item in manifest["primary_tranches"]
    )
    assert tranche_bytes == rows["r0-a"] + rows["r0-b"] + rows["r1-a"]
    assert all(
        item["queued_input_tokens_empirical_forecast"] <= 100
        for item in manifest["primary_tranches"]
    )
    memberships = {
        request_id: item["attempt_id"]
        for item in manifest["primary_tranches"]
        for request_id in item["request_ids_in_order"]
    }
    assert memberships["r0-a"] == memberships["r0-b"]
    assert set(memberships) == {"r0-a", "r0-b", "r1-a"}
    assert manifest["inherited_calibration_tree_sha256"] == "tree-hash"


def test_prepare_continuation_refuses_unknown_or_successful_recovery_membership(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="exactly the non-successful"):
        module._validate_calibration_failure_ids(
            ordered_primary_ids=["ok", "bad"],
            events=[
                {"request_id": "ok", "validation_status": "success"},
                {"request_id": "bad", "validation_status": "invalid_output"},
            ],
            recovery_ids=["ok"],
        )


def test_queue_gate_uses_conservative_reserved_sum() -> None:
    manifest = {
        "provider_queued_input_token_limit": 40_000_000,
        "maximum_concurrent_attempts": 1,
        "attempts": [
            {"attempt_id": "a", "queued_input_tokens_empirical_forecast": 30_000_000},
            {"attempt_id": "b", "queued_input_tokens_empirical_forecast": 30_000_000},
        ],
    }
    with pytest.raises(ValueError, match="concurrency"):
        module._validate_candidate_admission(
            manifest=manifest,
            candidate_id="b",
            active_attempt_ids=["a"],
            known_actual_cost_usd=4.0,
            active_cost_reservations_usd=0.0,
        )


def test_spend_admission_permits_exact_stop_and_rejects_above_or_at_known_stop() -> None:
    manifest = {
        "provider_queued_input_token_limit": 40_000_000,
        "maximum_concurrent_attempts": 1,
        "hard_campaign_stop_usd": 40.0,
        "attempts": [
            {
                "attempt_id": "candidate",
                "queued_input_tokens_empirical_forecast": 30_000_000,
                "calibrated_cost_reservation_usd": 4.0,
            }
        ],
    }
    module._validate_candidate_admission(
        manifest=manifest,
        candidate_id="candidate",
        active_attempt_ids=[],
        known_actual_cost_usd=36.0,
        active_cost_reservations_usd=0.0,
    )
    with pytest.raises(ValueError, match="hard stop"):
        module._validate_candidate_admission(
            manifest=manifest,
            candidate_id="candidate",
            active_attempt_ids=[],
            known_actual_cost_usd=36.000_001,
            active_cost_reservations_usd=0.0,
        )
    with pytest.raises(ValueError, match="hard stop"):
        module._validate_candidate_admission(
            manifest=manifest,
            candidate_id="candidate",
            active_attempt_ids=[],
            known_actual_cost_usd=40.0,
            active_cost_reservations_usd=0.0,
        )


def test_calibrated_reservation_uses_larger_estimator_with_margin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [{"body": {"input": "x"}} for _ in range(2)]
    body_bytes = sum(module._provider_body_bytes(row) for row in rows)
    bundle = {
        "config": {
            "empirical_calibration": {
                "source_actual_cost_usd": 1.0,
                "source_request_count": 10,
            }
        }
    }
    monkeypatch.setattr(module, "load_price_snapshot", lambda _: {})
    monkeypatch.setattr(
        module, "_strict_exposure_for_provider_bodies", lambda **_: 99.0
    )
    metrics = module._cost_metrics(
        bundle_root=tmp_path, bundle=bundle, rows=rows
    )
    raw = (
        module.CALIBRATION_ALL_INPUT_AND_CACHE_COST_USD
        * body_bytes
        / module.CALIBRATION_BODY_BYTES
        + module.CALIBRATION_OUTPUT_COST_USD
        * len(rows)
        / module.CALIBRATION_REQUEST_COUNT
    )
    scaled_direct = 0.2 * (
        module.CALIBRATION_KNOWN_COST_USD
        / module.CALIBRATION_DIRECT_FORECAST_USD
    )
    assert metrics["calibrated_cost_reservation_usd"] == pytest.approx(
        max(raw, scaled_direct) * 1.25
    )
    assert metrics["strict_no_cache_full_output_exposure_usd"] == 99.0


def test_submit_persists_calibrated_cost_gate_before_provider_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    attempt_root = run_root / "attempts/primary-tranche-000"
    attempt_root.mkdir(parents=True)
    input_path = attempt_root / "input.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")
    binding = {
        "attempt_id": "primary-tranche-000",
        "generation": "continuation-primary",
        "input_relative_path": "attempts/primary-tranche-000/input.jsonl",
        "input_sha256": module.file_sha256(input_path),
        "request_count": 1,
        "queued_input_tokens_empirical_forecast": 30_000_000,
        "calibrated_cost_reservation_usd": 4.0,
    }
    manifest = {
        "continuation_manifest_sha256": "m" * 64,
        "attempts": [binding],
        "maximum_concurrent_attempts": 1,
        "provider_queued_input_token_limit": 40_000_000,
        "warning_spend_threshold_usd": 20.0,
        "hard_campaign_stop_usd": 40.0,
    }
    monkeypatch.setattr(module, "_load_continuation", lambda _: (manifest, {}))
    monkeypatch.setattr(module, "_cost_state", lambda *a, **k: (36.0, 0.0))
    uploads: list[Path] = []

    def uploader(path: Path) -> dict[str, object]:
        uploads.append(path)
        return {
            "schema_version": module.production_v1.UPLOAD_SCHEMA,
            "provider": "openai",
            "input_file_id": "file-1",
            "purpose": "batch",
        }

    def creator(input_file_id: str, *, metadata: dict[str, str]) -> dict[str, object]:
        return {
            "schema_version": "adag.labeling.provider-batch.v1",
            "provider": "openai",
            "batch_id": "batch-1",
            "input_file_id": input_file_id,
            "endpoint": "/v1/responses",
            "completion_window": "24h",
            "metadata": metadata,
            "status": "validating",
        }

    module.submit_attempt(
        run_root=run_root,
        attempt_id="primary-tranche-000",
        uploader=uploader,
        creator=creator,
    )
    assert uploads == [input_path]
    submission_intent = json.loads(
        (attempt_root / "submission-intent.json").read_text()
    )
    assert submission_intent["candidate_calibrated_cost_reservation_usd"] == 4.0
    assert submission_intent["projected_warning_threshold_reached"] is True

    blocked_root = tmp_path / "blocked"
    blocked_attempt = blocked_root / "attempts/primary-tranche-000"
    blocked_attempt.mkdir(parents=True)
    blocked_input = blocked_attempt / "input.jsonl"
    blocked_input.write_text("{}\n", encoding="utf-8")
    blocked_binding = {
        **binding,
        "input_sha256": module.file_sha256(blocked_input),
    }
    blocked_manifest = {**manifest, "attempts": [blocked_binding]}
    monkeypatch.setattr(
        module, "_load_continuation", lambda _: (blocked_manifest, {})
    )
    monkeypatch.setattr(module, "_cost_state", lambda *a, **k: (36.01, 0.0))
    with pytest.raises(ValueError, match="hard stop"):
        module.submit_attempt(
            run_root=blocked_root,
            attempt_id="primary-tranche-000",
            uploader=lambda _: (_ for _ in ()).throw(AssertionError("uploaded")),
            creator=creator,
        )
    assert not (blocked_attempt / "submission-intent.json").exists()


def test_deferred_recovery_unions_inherited_and_new_failures_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    bundle_root = tmp_path / "bundle"
    (bundle_root / "batch-shards").mkdir(parents=True)
    (run_root / "inherited-calibration-run/shards/shard-005").mkdir(parents=True)
    primary_root = run_root / "attempts/primary-tranche-000"
    primary_root.mkdir(parents=True)
    source_rows = {
        "ok-old": _line("ok-old", "ok old"),
        "bad-old": _line("bad-old", "bad old"),
        "ok-new": _line("ok-new", "ok new"),
        "bad-new": _line("bad-new", "bad new"),
    }
    (bundle_root / "batch-shards/shard-005.jsonl").write_bytes(
        source_rows["ok-old"] + source_rows["bad-old"]
    )
    (bundle_root / "batch-shards/shard-000.jsonl").write_bytes(
        source_rows["ok-new"] + source_rows["bad-new"]
    )
    inherited_events = [
        {"request_id": "ok-old", "validation_status": "success"},
        {"request_id": "bad-old", "validation_status": "provider_error"},
    ]
    (run_root / "inherited-calibration-run/shards/shard-005/events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in inherited_events), encoding="utf-8"
    )
    primary_events = [
        {"request_id": "ok-new", "validation_status": "success"},
        {"request_id": "bad-new", "validation_status": "invalid_output"},
    ]
    (primary_root / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in primary_events), encoding="utf-8"
    )
    collection = module._hashed(
        {
            "cost_complete": True,
            "events_sha256": module.file_sha256(primary_root / "events.jsonl"),
        },
        "collection_sha256",
    )
    (primary_root / "collection.json").write_text(json.dumps(collection))
    manifest = {
        "continuation_manifest_sha256": "m" * 64,
        "bundle_root": str(bundle_root),
        "calibration_events_sha256": "e" * 64,
        "inherited_failure_request_ids": ["bad-old"],
        "tranche_empirical_queue_cap": 30_000_000,
        "attempts": [{"attempt_id": "primary-tranche-000"}],
    }
    bundle = {
        "shards": [
            {"shard_id": "shard-000", "path": "batch-shards/shard-000.jsonl"},
            {"shard_id": "shard-005", "path": "batch-shards/shard-005.jsonl"},
        ],
        "request_index": [
            {"request_id": request_id}
            for request_id in ("ok-old", "bad-old", "ok-new", "bad-new")
        ],
    }
    monkeypatch.setattr(module, "_load_continuation", lambda _: (manifest, bundle))
    monkeypatch.setattr(
        module,
        "_validate_inherited_calibration_evidence",
        lambda **_: ({}, inherited_events),
    )
    monkeypatch.setattr(
        module,
        "_validate_collected_attempt",
        lambda **_: (collection, primary_events),
    )
    monkeypatch.setattr(module, "_queue_tokens", lambda **_: 10)
    captured: list[str] = []

    def bind(**kwargs: object) -> dict[str, object]:
        rows = kwargs["exact_rows"]
        assert isinstance(rows, list)
        captured.extend(row[0] for row in rows)
        return {
            "attempt_id": "failed-only-recovery-000",
            "generation": "failed-only-recovery",
            "request_ids_in_order": list(captured),
            "queued_input_tokens_empirical_forecast": 10,
        }

    monkeypatch.setattr(module, "_attempt_binding", bind)
    recovery = module.prepare_failed_only_recovery(run_root=run_root)
    assert captured == ["bad-old", "bad-new"]
    assert recovery["request_count"] == 2
    assert recovery["successful_requests_rerun"] == 0
    assert recovery["continuation_primary_failure_count"] == 1


def test_inherited_credit_failures_are_reconciled_as_exact_zero_usage(
    tmp_path: Path,
) -> None:
    calibration_root = tmp_path / "calibration"
    raw_root = calibration_root / "shards/shard-005/raw"
    raw_root.mkdir(parents=True)
    usage = {
        "input_tokens": 11,
        "cache_read_tokens": 7,
        "cache_write_tokens": 1,
        "uncached_input_tokens": 3,
        "output_tokens": 5,
        "reasoning_tokens": 2,
    }
    events = [
        {
            "request_id": f"successful-{index}",
            "validation_status": "success",
            "usage": usage,
        }
        for index in range(6_439)
    ]
    events.extend(
        {
            "request_id": f"credit-failure-{index}",
            "validation_status": "provider_error",
            "provider_error_code": "credit_balance_exhausted",
            "usage": {
                "input_tokens": None,
                "output_tokens": None,
            },
        }
        for index in range(8)
    )
    snapshot = {
        "status": "completed",
        "request_counts": {"total": 6_447, "completed": 6_439, "failed": 8},
        "usage": {
            "input_tokens": 6_439 * 11,
            "input_tokens_details": {"cached_tokens": 6_439 * 7},
            "output_tokens": 6_439 * 5,
            "output_tokens_details": {"reasoning_tokens": 6_439 * 2},
        },
    }
    (raw_root / "provider-snapshot.json").write_text(json.dumps(snapshot))

    reconciliation = module._reconcile_inherited_calibration_cost(
        calibration_run_root=calibration_root, events=events
    )

    assert reconciliation["credit_failures_zero_provider_usage"] is True
    assert reconciliation["cost_complete"] is True
    assert reconciliation["adopted_actual_cost_usd"] == pytest.approx(
        module.CALIBRATION_KNOWN_COST_USD
    )

    snapshot["usage"]["input_tokens"] += 1  # type: ignore[index,operator]
    (raw_root / "provider-snapshot.json").write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="zero-usage cost reconciliation"):
        module._reconcile_inherited_calibration_cost(
            calibration_run_root=calibration_root, events=events
        )


@pytest.mark.parametrize(
    "failure_status",
    ["provider_error", "invalid_output", "missing", "ambiguous_output"],
)
def test_calibration_recovery_selects_every_non_success_status(
    failure_status: str,
) -> None:
    events = [
        {"request_id": "ok", "validation_status": "success"},
        {"request_id": "bad", "validation_status": failure_status},
    ]
    module._validate_calibration_failure_ids(
        ordered_primary_ids=["ok", "bad"],
        events=events,
        recovery_ids=["bad"],
    )


@pytest.mark.parametrize("matches", [[], [{"batch_id": "a"}, {"batch_id": "b"}]])
def test_ambiguous_submission_recovery_requires_exactly_one_match_and_never_creates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    matches: list[dict[str, str]],
) -> None:
    manifest = _simple_attempt_manifest()
    _write_ambiguous_submission_state(
        tmp_path, manifest=manifest, retain_create=False
    )
    monkeypatch.setattr(module, "_load_continuation", lambda _: (manifest, {}))
    discovery_calls = []

    def discoverer(**kwargs: object) -> dict[str, object]:
        discovery_calls.append(kwargs)
        return {
            "exhaustive": True,
            "page_count": 2,
            "total_scanned": len(matches),
            "snapshots": matches,
        }

    with pytest.raises(ValueError, match="exactly one metadata-matched Batch"):
        module.recover_attempt_submission(
            run_root=tmp_path,
            attempt_id="primary-tranche-000",
            discoverer=discoverer,
        )

    attempt_root = tmp_path / "attempts/primary-tranche-000"
    assert len(discovery_calls) == 1
    assert not (attempt_root / "submission.json").exists()
    assert not (attempt_root / "provider-create-response.json").exists()


def test_ambiguous_submission_repairs_from_retained_create_without_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _simple_attempt_manifest()
    provider = _write_ambiguous_submission_state(
        tmp_path, manifest=manifest, retain_create=True
    )
    monkeypatch.setattr(module, "_load_continuation", lambda _: (manifest, {}))

    receipt = module.recover_attempt_submission(
        run_root=tmp_path,
        attempt_id="primary-tranche-000",
        discoverer=lambda **_: (_ for _ in ()).throw(
            AssertionError("retained create must avoid discovery")
        ),
    )

    assert receipt["recovered_by"] == "immediate_retained_create_snapshot"
    assert receipt["provider_response"] == provider
    assert (tmp_path / "attempts/primary-tranche-000/submission.json").is_file()


def test_provider_discovery_exhausts_all_pages_before_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wanted = {"campaign": "campaign", "shard": "shard", "generation": "primary"}

    class Batch:
        def __init__(self, batch_id: str, metadata: dict[str, str]) -> None:
            self.id = batch_id
            self.metadata = metadata

    class Page:
        def __init__(self, data: list[Batch]) -> None:
            self.data = data

    class Pages:
        def iter_pages(self) -> object:
            return iter(
                [
                    Page([Batch("unrelated", {"campaign": "different"})]),
                    Page([Batch("matched-on-second-page", wanted)]),
                ]
            )

    class Batches:
        def list(self, *, limit: int) -> Pages:
            assert limit == 100
            return Pages()

    class Client:
        batches = Batches()

    monkeypatch.setattr(module.production_v1, "_openai_client", lambda: Client())
    monkeypatch.setattr(
        module,
        "_production_provider_batch_dict",
        lambda batch: {"batch_id": batch.id, "metadata": batch.metadata},
    )

    discovery = module._discover_batches_by_metadata(wanted)

    assert discovery["exhaustive"] is True
    assert discovery["page_count"] == 2
    assert discovery["total_scanned"] == 2
    assert discovery["snapshots"] == [
        {"batch_id": "matched-on-second-page", "metadata": wanted}
    ]


def test_existing_collection_repairs_missing_cost_status_without_redownload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _simple_attempt_manifest()
    attempt_root = tmp_path / "attempts/primary-tranche-000"
    attempt_root.mkdir(parents=True)
    collection = module._hashed(
        {
            "cost_complete": True,
            "known_priced_cost_usd": 16.0,
        },
        "collection_sha256",
    )
    (attempt_root / "collection.json").write_text(json.dumps(collection))
    monkeypatch.setattr(module, "_load_continuation", lambda _: (manifest, {}))
    monkeypatch.setattr(
        module,
        "_validate_collected_attempt",
        lambda **_: (collection, []),
    )

    observed = module.collect_attempt(
        run_root=tmp_path,
        attempt_id="primary-tranche-000",
        downloader=lambda _: (_ for _ in ()).throw(AssertionError("redownloaded")),
    )

    assert observed == collection
    cost = json.loads((tmp_path / "cost-status/receipt-0000.json").read_text())
    assert cost["known_actual_cost_usd"] == 20.0
    assert cost["warning_threshold_reached"] is True


def test_warning_is_sticky_starting_at_exactly_twenty_dollars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _simple_attempt_manifest()
    tmp_path.mkdir(exist_ok=True)
    states = iter([(20.0, 0.0), (19.0, 0.0)])
    monkeypatch.setattr(module, "_cost_state", lambda *a, **k: next(states))

    first = module._append_cost_status(
        run_root=tmp_path, manifest=manifest, trigger="exact-threshold"
    )
    second = module._append_cost_status(
        run_root=tmp_path, manifest=manifest, trigger="sticky-threshold"
    )

    assert first["warning_threshold_reached"] is True
    assert second["warning_threshold_reached"] is True
    assert second["previous_cost_status_sha256"] == first["cost_status_sha256"]


def test_failed_only_recovery_validates_raw_source_evidence_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    inherited = run_root / "inherited-calibration-run/shards/shard-005"
    primary = run_root / "attempts/primary-tranche-000"
    inherited.mkdir(parents=True)
    primary.mkdir(parents=True)
    (inherited / "events.jsonl").write_text(
        json.dumps({"request_id": "old-failure", "validation_status": "provider_error"})
        + "\n"
    )
    collection = module._hashed(
        {"cost_complete": True, "events_sha256": "irrelevant"},
        "collection_sha256",
    )
    (primary / "collection.json").write_text(json.dumps(collection))
    manifest = {
        "continuation_manifest_sha256": "m" * 64,
        "bundle_root": str(tmp_path / "bundle"),
        "inherited_failure_request_ids": ["old-failure"],
        "tranche_empirical_queue_cap": 100,
        "attempts": [{"attempt_id": "primary-tranche-000"}],
    }
    bundle = {"request_index": [], "shards": []}
    monkeypatch.setattr(module, "_load_continuation", lambda _: (manifest, bundle))
    monkeypatch.setattr(
        module,
        "_validate_inherited_calibration_evidence",
        lambda **_: (
            {},
            [{"request_id": "old-failure", "validation_status": "provider_error"}],
        ),
    )
    monkeypatch.setattr(
        module,
        "_validate_collected_attempt",
        lambda **_: (_ for _ in ()).throw(ValueError("raw provider evidence drift")),
    )

    with pytest.raises(ValueError, match="raw provider evidence drift"):
        module.prepare_failed_only_recovery(run_root=run_root)
    assert not (run_root / "failed-only-recovery").exists()
    assert not (run_root / "attempts/failed-only-recovery-000").exists()


def test_failed_only_recovery_over_cap_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    bundle_root = tmp_path / "bundle"
    inherited = run_root / "inherited-calibration-run/shards/shard-005"
    primary = run_root / "attempts/primary-tranche-000"
    inherited.mkdir(parents=True)
    primary.mkdir(parents=True)
    (bundle_root / "batch-shards").mkdir(parents=True)
    (inherited / "events.jsonl").write_text(
        json.dumps({"request_id": "old-failure", "validation_status": "provider_error"})
        + "\n"
    )
    (bundle_root / "batch-shards/shard-005.jsonl").write_bytes(
        _line("old-failure", "old")
    )
    (bundle_root / "batch-shards/shard-000.jsonl").write_bytes(
        _line("new-failure", "new")
    )
    manifest = {
        "continuation_manifest_sha256": "m" * 64,
        "bundle_root": str(bundle_root),
        "calibration_events_sha256": "e" * 64,
        "inherited_failure_request_ids": ["old-failure"],
        "tranche_empirical_queue_cap": 5,
        "attempts": [
            {
                "attempt_id": "primary-tranche-000",
                "generation": "continuation-primary",
            }
        ],
    }
    bundle = {
        "request_index": [
            {"request_id": "old-failure"},
            {"request_id": "new-failure"},
        ],
        "shards": [
            {"shard_id": "shard-005", "path": "batch-shards/shard-005.jsonl"},
            {"shard_id": "shard-000", "path": "batch-shards/shard-000.jsonl"},
        ],
    }
    primary_events = [
        {"request_id": "new-failure", "validation_status": "missing"}
    ]
    (primary / "collection.json").write_text(
        json.dumps(
            module._hashed(
                {"cost_complete": True, "events_sha256": "e" * 64},
                "collection_sha256",
            )
        )
    )
    monkeypatch.setattr(module, "_load_continuation", lambda _: (manifest, bundle))
    monkeypatch.setattr(
        module,
        "_validate_inherited_calibration_evidence",
        lambda **_: (
            {},
            [{"request_id": "old-failure", "validation_status": "provider_error"}],
        ),
    )
    monkeypatch.setattr(
        module,
        "_validate_collected_attempt",
        lambda **_: ({"cost_complete": True}, primary_events),
    )

    monkeypatch.setattr(module, "_queue_tokens", lambda **_: 6)
    monkeypatch.setattr(
        module,
        "_attempt_binding",
        lambda **_: (_ for _ in ()).throw(AssertionError("materialized over-cap")),
    )
    with pytest.raises(ValueError, match="exceeds queue cap"):
        module.prepare_failed_only_recovery(run_root=run_root)

    assert not (run_root / "failed-only-recovery").exists()
    assert not (run_root / "attempts/failed-only-recovery-000").exists()


def test_finalizer_rejects_less_than_exactly_three_replica_votes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    destination = tmp_path / "final"
    primary = {
        "attempt_id": "primary-tranche-000",
        "generation": "continuation-primary",
    }
    recovery = {
        "attempt_id": "failed-only-recovery-000",
        "generation": "failed-only-recovery",
    }
    manifest = {
        "continuation_manifest_sha256": "m" * 64,
        "bundle_root": str(tmp_path / "bundle"),
        "bundle_manifest_sha256": "b" * 64,
        "calibration_known_priced_cost_usd": 4.0,
        "warning_spend_threshold_usd": 20.0,
        "hard_campaign_stop_usd": 40.0,
        "attempts": [primary],
    }
    request_rows = [
        {
            "request_id": f"request-{replica}",
            "replica_index": replica,
        }
        for replica in range(3)
    ]
    bundle = {
        "manifest": {"manifest_sha256": "b" * 64},
        "request_index": request_rows,
        "units": [{"unit_id": "unit-0", "assignment_route": "openai_pending"}],
        "config": {"claim_boundary": "test"},
    }

    def event(request_id: str, replica: int, *, decisions: bool) -> dict[str, object]:
        return {
            "request_id": request_id,
            "replica_index": replica,
            "validation_status": "success",
            "decisions": (
                [{"unit_id": "unit-0", "label": "active_task_process"}]
                if decisions
                else []
            ),
        }

    inherited_failure = {
        "request_id": "request-0",
        "replica_index": 0,
        "validation_status": "provider_error",
        "decisions": None,
    }
    primary_events = [
        event("request-1", 1, decisions=True),
        event("request-2", 2, decisions=False),
    ]
    recovery_events = [event("request-0", 0, decisions=True)]
    monkeypatch.setattr(module, "_load_continuation", lambda _: (manifest, bundle))
    monkeypatch.setattr(module, "_all_attempts", lambda *_: [primary, recovery])
    monkeypatch.setattr(
        module,
        "_validate_inherited_calibration_evidence",
        lambda **_: (
            {"collection_sha256": "c0", "events_sha256": "e0"},
            [inherited_failure],
        ),
    )

    def validate_attempt(**kwargs: object) -> tuple[dict[str, object], list[dict[str, object]]]:
        binding = kwargs["binding"]
        assert isinstance(binding, dict)
        is_recovery = binding["generation"] == "failed-only-recovery"
        return (
            {
                "collection_sha256": "cr" if is_recovery else "cp",
                "events_sha256": "er" if is_recovery else "ep",
                "known_priced_cost_usd": 0.1,
            },
            recovery_events if is_recovery else primary_events,
        )

    monkeypatch.setattr(module, "_validate_collected_attempt", validate_attempt)
    monkeypatch.setattr(
        module,
        "load_production_bundle",
        lambda *_, **__: {"units": bundle["units"]},
    )

    with pytest.raises(ValueError, match="exactly three votes"):
        module.finalize_continuation(
            run_root=run_root,
            destination=destination,
        )
    assert not destination.exists()


def _minimal_frozen_bank(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], list[Path]]:
    continuation_root = root / "continuation-evidence"
    (root / "campaign-bundle").mkdir(parents=True)
    (continuation_root / "failed-only-recovery").mkdir(parents=True)
    (
        continuation_root
        / "inherited-calibration-run/shards/shard-005"
    ).mkdir(parents=True)
    (continuation_root / "cost-status").mkdir()
    (continuation_root / "attempts/failed-only-recovery-000").mkdir(parents=True)
    for filename in ("effective-events.jsonl", "proposals.jsonl", "sampling-groups.jsonl"):
        (root / filename).write_bytes(b"")
    (
        continuation_root
        / "inherited-calibration-run/shards/shard-005/events.jsonl"
    ).write_bytes(b"")
    (continuation_root / "attempts/failed-only-recovery-000/input.jsonl").write_bytes(
        b""
    )
    continuation = module._hashed(
        {
            "schema_version": module.CONTINUATION_SCHEMA,
            "status": "prepared_offline_no_provider_calls",
            "bundle_root": "/original/source/is/intentionally/unavailable",
            "bundle_manifest_sha256": "b" * 64,
            "attempts": [],
        },
        "continuation_manifest_sha256",
    )
    (continuation_root / "continuation-manifest.json").write_text(
        json.dumps(continuation)
    )
    recovery = module._hashed(
        {
            "continuation_manifest_sha256": continuation[
                "continuation_manifest_sha256"
            ],
            "attempt": {
                "attempt_id": "failed-only-recovery-000",
                "generation": "failed-only-recovery",
                "input_relative_path": "attempts/failed-only-recovery-000/input.jsonl",
            },
        },
        "recovery_manifest_sha256",
    )
    (continuation_root / "failed-only-recovery/manifest.json").write_text(
        json.dumps(recovery)
    )
    cost = module._hashed(
        {
            "continuation_manifest_sha256": continuation[
                "continuation_manifest_sha256"
            ],
            "previous_cost_status_sha256": None,
            "known_actual_cost_usd": 4.0,
            "active_calibrated_reservations_usd": 0.0,
            "warning_threshold_reached": False,
            "hard_stop_crossed_after_inflight_attempt": False,
        },
        "cost_status_sha256",
    )
    (continuation_root / "cost-status/receipt-0000.json").write_text(json.dumps(cost))
    bundle = {
        "manifest": {"manifest_sha256": "b" * 64},
        "request_index": [],
        "units": [],
        "shards": [],
    }
    loaded_paths: list[Path] = []

    def load_bundle(path: Path, **_: object) -> dict[str, object]:
        loaded_paths.append(path)
        return bundle

    monkeypatch.setattr(module.production_v1, "_validate_readonly_modes", lambda _: None)
    monkeypatch.setattr(module, "load_production_bundle", load_bundle)
    monkeypatch.setattr(
        module, "_validate_inherited_calibration_evidence", lambda **_: ({}, [])
    )
    monkeypatch.setattr(module, "_all_attempts", lambda *_: [])
    inventory = module.production_v1._write_evidence_inventory(root)
    result = module._hashed(
        {
            "schema_version": module.FINAL_SCHEMA,
            "status": "frozen_sampling_proposals_not_semantic_truth",
            "continuation_manifest_sha256": continuation[
                "continuation_manifest_sha256"
            ],
            "bundle_manifest_sha256": "b" * 64,
            "request_count": 0,
            "actual_total_cost_usd": 4.0,
            "cost_complete": True,
            "final_cost_status_sha256": cost["cost_status_sha256"],
            "hard_stop_crossed_after_inflight_attempt": False,
            "effective_events_sha256": module.file_sha256(
                root / "effective-events.jsonl"
            ),
            "proposals_sha256": module.file_sha256(root / "proposals.jsonl"),
            "sampling_groups_sha256": module.file_sha256(
                root / "sampling-groups.jsonl"
            ),
            "evidence_inventory_sha256": inventory["evidence_inventory_sha256"],
        },
        "proposal_bank_manifest_sha256",
    )
    (root / "manifest.json").write_text(json.dumps(result))
    return result, loaded_paths


def test_strict_loader_is_source_root_independent_and_rejects_evidence_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "frozen"
    root.mkdir()
    _result, loaded_paths = _minimal_frozen_bank(root, monkeypatch)

    loaded = module.load_frozen_continuation_proposal_bank(root)
    assert loaded["manifest"]["actual_total_cost_usd"] == 4.0
    assert loaded_paths == [root / "campaign-bundle"]

    (root / "effective-events.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError, match="evidence file drift"):
        module.load_frozen_continuation_proposal_bank(root)
