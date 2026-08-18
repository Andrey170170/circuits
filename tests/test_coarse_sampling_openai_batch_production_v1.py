from __future__ import annotations

import json
from pathlib import Path

import circuits.analysis.bonafide.coarse_sampling_openai_batch_production_v1 as module
import pytest
from circuits.analysis.bonafide.coarse_sampling_openai_batch_production_v1 import (
    _parse_row,
    initialize_campaign_run,
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

    def fail(*args, **kwargs):
        raise TimeoutError("create response lost")

    with pytest.raises(RuntimeError, match="indeterminate"):
        submit_shard(run_root=run, shard_id="shard-000", submitter=fail)
    assert (shard_root / "submission-intent.json").is_file()
    failure = json.loads((shard_root / "submission-failure.json").read_text())
    assert failure["automatic_retry_permitted"] is False


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
