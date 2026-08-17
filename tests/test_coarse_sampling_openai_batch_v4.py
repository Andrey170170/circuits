from __future__ import annotations

import json
from pathlib import Path

import circuits.analysis.bonafide.coarse_sampling_openai_batch_v4 as module
import pytest
from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_annotation_v3 import ARM_ZERO_SHOT
from circuits.analysis.bonafide.coarse_sampling_openai_batch_v4 import (
    collect_v4_batch,
    parse_v4_batch_row,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _do_not_chmod_temporary_test_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_readonly_tree", lambda _: None)


def _request(window: int, arm: str, replica: int) -> dict:
    body = {"model": "gpt-5.6-luna", "input": [], "store": False}
    return {
        "request_id": f"request-{arm}-{window}-{replica}",
        "arm_id": arm,
        "window_index": window,
        "replica_index": replica,
        "body_sha256": canonical_sha256(body),
        "repeat_of_request_id": (None if replica == 0 else f"request-{arm}-{window}-0"),
        "focal_unit_ids": [
            f"unit-{window}-{offset}" for offset in range(2 if window < 9 else 1)
        ],
        "provider_body": body,
    }


def _row(request: dict) -> dict:
    decisions = {
        "decisions": [
            {
                "unit_id": unit_id,
                "tag": "active_task_work",
                "confidence": "high",
                "boundary_concerns": [],
                "boundary_note": "",
            }
            for unit_id in request["focal_unit_ids"]
        ]
    }
    body = {
        "id": f"response-{request['request_id']}",
        "model": "gpt-5.6-luna",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": json.dumps(decisions)}],
            }
        ],
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 60, "cache_write_tokens": 10},
            "output_tokens": 40,
            "output_tokens_details": {"reasoning_tokens": 15},
        },
    }
    return {
        "custom_id": request["request_id"],
        "response": {
            "status_code": 200,
            "request_id": f"provider-{request['request_id']}",
            "body": body,
        },
        "error": None,
    }


def test_parser_preserves_arm_replica_and_exact_decisions() -> None:
    request = _request(3, ARM_ZERO_SHOT, 2)
    event = parse_v4_batch_row(_row(request), request)
    assert event["validation_status"] == "success"
    assert event["arm_id"] == ARM_ZERO_SHOT
    assert event["replica_index"] == 2
    assert event["window_index"] == 3
    assert event["usage"]["uncached_input_tokens"] == 30
    assert [item["unit_id"] for item in event["decisions"]] == request["focal_unit_ids"]


def test_collect_requires_all_45_requests_and_24_unique_arm_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = [
        _request(window, arm, replica)
        for arm in (ARM_ZERO_SHOT,)
        for window in range(15)
        for replica in range(3)
    ]
    price = ROOT / "scripts/bonafide/configs/labeling/prices-2026-08-16-coarse-v2.json"
    loaded = {
        "requests": requests,
        "cost_plan": {
            "price_snapshot_relative_path": str(price.relative_to(ROOT)),
            "price_snapshot_sha256": file_sha256(price),
        },
    }
    intent = {
        "run_intent_sha256": "r" * 64,
        "qualification_root": str(ROOT),
        "maximum_authorized_cost_usd": 20.0,
    }
    submission = {
        "submission_sha256": "s" * 64,
        "provider_response": {
            "batch_id": "batch-v4",
            "input_file_id": "input-v4",
        },
    }
    monkeypatch.setattr(module, "_load_run", lambda _: (intent, loaded, submission))
    monkeypatch.setattr(
        module, "_validate_provider_snapshot", lambda *args, **kwargs: None
    )
    content = "".join(json.dumps(_row(request)) + "\n" for request in requests).encode()

    def download(*args, **kwargs):
        return {
            "status": "completed",
            "output_file_id": "output-v4",
            "error_file_id": None,
        }, {"output": {"file_id": "output-v4", "content": content}}

    run = tmp_path / "run"
    run.mkdir()
    manifest = collect_v4_batch(run_root=run, downloader=download)
    assert manifest["status"] == "complete"
    assert manifest["success_count"] == 45
    assert manifest["unique_arm_target_coverage"] == 24
    assert manifest["arm_success_counts"] == {
        ARM_ZERO_SHOT: 45,
    }
    assert manifest["qualification_decisions_ready"] is True
