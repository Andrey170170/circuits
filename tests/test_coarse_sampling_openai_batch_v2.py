from __future__ import annotations

import json
from pathlib import Path

import circuits.analysis.bonafide.coarse_sampling_openai_batch_v2 as module
import pytest
from circuits.analysis.bonafide.canonical import canonical_sha256
from circuits.analysis.bonafide.coarse_sampling_annotation_v2 import (
    ARM_FULL_UNIT,
    ARM_TARGET_ONLY,
)
from circuits.analysis.bonafide.coarse_sampling_openai_batch_v2 import (
    collect_v2_batch,
    parse_v2_batch_row,
    submit_v2_batch,
)
from circuits.labeling.schema import CostEstimate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _do_not_chmod_temporary_test_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_readonly_tree", lambda _: None)


def _request(index: int = 0, *, arm_id: str = ARM_TARGET_ONLY) -> dict:
    focal = [f"unit-{index}-{offset}" for offset in range(6)]
    body = {"model": "gpt-5.6-luna", "input": [], "store": False}
    return {
        "request_id": f"request-{arm_id}-{index}",
        "arm_id": arm_id,
        "body_sha256": canonical_sha256(body),
        "source_v1_request_id": f"v1-{index}",
        "repeat_of_request_id": None,
        "focal_unit_ids": focal,
        "provider_body": body,
    }


def _decisions(request: dict) -> dict:
    return {
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


def _row(request: dict, *, status: str = "completed") -> dict:
    text = json.dumps(_decisions(request))
    body = {
        "id": f"response-{request['request_id']}",
        "model": "gpt-5.6-luna",
        "status": status,
        "output": [
            {
                "type": "message",
                "status": status,
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 60, "cache_write_tokens": 10},
            "output_tokens": 40,
            "output_tokens_details": {"reasoning_tokens": 15},
        },
    }
    if status != "completed":
        body["incomplete_details"] = {"reason": "max_output_tokens"}
    return {
        "custom_id": request["request_id"],
        "response": {"status_code": 200, "request_id": "provider-1", "body": body},
        "error": None,
    }


def test_parser_accepts_only_exact_completed_target_coverage() -> None:
    request = _request()
    event = parse_v2_batch_row(_row(request), request)
    assert event["validation_status"] == "success"
    assert [item["unit_id"] for item in event["decisions"]] == request["focal_unit_ids"]
    assert event["usage"]["cache_read_tokens"] == 60
    assert event["usage"]["cache_write_tokens"] == 10
    assert event["usage"]["uncached_input_tokens"] == 30

    missing = _row(request)
    parsed = json.loads(missing["response"]["body"]["output"][0]["content"][0]["text"])
    parsed["decisions"].pop()
    missing["response"]["body"]["output"][0]["content"][0]["text"] = json.dumps(parsed)
    assert parse_v2_batch_row(missing, request)["validation_status"] == "invalid_output"


def test_parser_preserves_incomplete_and_provider_error_without_partial_acceptance() -> (
    None
):
    request = _request()
    incomplete = parse_v2_batch_row(_row(request, status="incomplete"), request)
    assert incomplete["validation_status"] == "incomplete"
    assert incomplete["decisions"] is None
    error = {
        "custom_id": request["request_id"],
        "response": None,
        "error": {"code": "batch_expired", "message": "not executed"},
    }
    failed = parse_v2_batch_row(error, request)
    assert failed["validation_status"] == "provider_error"
    assert failed["decisions"] is None


def _loaded_qualification(tmp_path: Path, requests: list[dict]) -> dict:
    price = ROOT / "scripts/bonafide/configs/labeling/prices-2026-08-16-coarse-v2.json"
    return {
        "manifest": {"manifest_sha256": "a" * 64},
        "config": {"provider": {"api_key_env": "OPENAI_API_KEY"}},
        "requests": requests,
        "cost_plan": {
            "cost_plan_sha256": "b" * 64,
            "projected_upper_bound_usd": 0.5,
            "price_snapshot_path": str(price),
        },
    }


def _all_arm_requests() -> list[dict]:
    requests = []
    repeat_windows = {0, 5, 7, 9}
    for arm in (ARM_TARGET_ONLY, ARM_FULL_UNIT):
        for window in range(12):
            request = _request(window, arm_id=arm)
            requests.append(request)
            if window in repeat_windows:
                requests.append(
                    {
                        **request,
                        "request_id": f"{request['request_id']}-repeat",
                        "repeat_of_request_id": request["request_id"],
                    }
                )
    assert len(requests) == 32
    return requests


def test_submit_persists_indeterminate_failure_and_forbids_automatic_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification = tmp_path / "qualification"
    qualification.mkdir()
    (qualification / "batch-input.jsonl").write_text("{}\n")
    loaded = _loaded_qualification(tmp_path, [_request()])
    monkeypatch.setattr(module, "load_v2_qualification", lambda _: loaded)
    monkeypatch.setattr(
        module,
        "_source_revision",
        lambda: {"git_commit": "1" * 40, "tracked_worktree_clean": True},
    )

    def fail(*args, **kwargs):
        raise TimeoutError("provider state unknown")

    run = tmp_path / "run"
    with pytest.raises(RuntimeError, match="automatic retry is forbidden"):
        submit_v2_batch(
            qualification_root=qualification,
            run_root=run,
            maximum_authorized_cost_usd=1.0,
            authorization_note="test authorization",
            submitter=fail,
        )
    failure = json.loads((run / "submission-failure.json").read_text())
    assert failure["status"] == "failed_closed_indeterminate_provider_state"
    assert failure["automatic_retry_permitted"] is False
    with pytest.raises(FileExistsError, match="already exists"):
        submit_v2_batch(
            qualification_root=qualification,
            run_root=run,
            maximum_authorized_cost_usd=1.0,
            authorization_note="test authorization",
            submitter=fail,
        )


def test_collect_retains_raw_files_and_fails_closed_on_missing_custom_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request()
    loaded = _loaded_qualification(tmp_path, [request])
    intent = {
        "run_intent_sha256": "r" * 64,
        "maximum_authorized_cost_usd": 1.0,
    }
    provider = {
        "batch_id": "batch-1",
        "input_file_id": "file-input",
    }
    submission = {
        "submission_sha256": "s" * 64,
        "provider_response": provider,
    }
    monkeypatch.setattr(module, "_load_run", lambda _: (intent, loaded, submission))
    monkeypatch.setattr(
        module, "_validate_provider_snapshot", lambda *args, **kwargs: None
    )
    snapshot = {"status": "completed", "output_file_id": "file-output"}
    raw = {
        "output": {
            "file_id": "file-output",
            "content": (
                json.dumps(
                    {
                        "custom_id": "unknown-request",
                        "response": None,
                        "error": {"code": "bad"},
                    }
                )
                + "\n"
            ).encode(),
        }
    }

    def download(*args, **kwargs):
        return snapshot, raw

    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(ValueError, match="does not exactly cover"):
        collect_v2_batch(run_root=run, downloader=download)
    assert (run / "raw/output.jsonl").read_bytes() == raw["output"]["content"]
    manifest = json.loads((run / "collection-manifest.json").read_text())
    assert manifest["status"] == "failed_closed_row_coverage"
    assert manifest["missing_custom_ids"] == [request["request_id"]]


def test_collect_complete_requires_all_32_physical_and_144_unique_arm_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = _all_arm_requests()
    loaded = _loaded_qualification(tmp_path, requests)
    intent = {
        "run_intent_sha256": "r" * 64,
        "maximum_authorized_cost_usd": 1.0,
    }
    submission = {
        "submission_sha256": "s" * 64,
        "provider_response": {
            "batch_id": "batch-complete",
            "input_file_id": "file-input",
        },
    }
    monkeypatch.setattr(module, "_load_run", lambda _: (intent, loaded, submission))
    monkeypatch.setattr(
        module, "_validate_provider_snapshot", lambda *args, **kwargs: None
    )
    content = "".join(
        json.dumps(_row(request)) + "\n" for request in reversed(requests)
    ).encode()
    snapshot = {
        "status": "completed",
        "output_file_id": "file-output-complete",
        "error_file_id": None,
    }

    def download(*args, **kwargs):
        return snapshot, {
            "output": {"file_id": "file-output-complete", "content": content}
        }

    run = tmp_path / "complete-run"
    run.mkdir()
    manifest = collect_v2_batch(run_root=run, downloader=download)
    assert manifest["status"] == "complete"
    assert manifest["success_count"] == 32
    assert manifest["exact_target_coverage"] is True
    assert manifest["unique_arm_target_coverage"] == 144
    assert manifest["arm_success_counts"] == {
        ARM_TARGET_ONLY: 16,
        ARM_FULL_UNIT: 16,
    }
    assert manifest["usage_totals"]["input_tokens"] == 3200
    assert manifest["usage_totals"]["uncached_input_tokens"] == 960
    assert manifest["usage_totals"]["cache_read_tokens"] == 1920
    assert manifest["usage_totals"]["cache_write_tokens"] == 320
    assert manifest["usage_totals"]["reasoning_tokens"] == 480
    assert manifest["actual_total_cost_usd"] is not None


def test_collect_does_not_qualify_complete_decisions_without_priceable_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = _all_arm_requests()
    loaded = _loaded_qualification(tmp_path, requests)
    intent = {
        "run_intent_sha256": "r" * 64,
        "maximum_authorized_cost_usd": 1.0,
    }
    submission = {
        "submission_sha256": "s" * 64,
        "provider_response": {
            "batch_id": "batch-unpriceable",
            "input_file_id": "file-input",
        },
    }
    monkeypatch.setattr(module, "_load_run", lambda _: (intent, loaded, submission))
    monkeypatch.setattr(
        module, "_validate_provider_snapshot", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        module,
        "estimate_cost",
        lambda *args, **kwargs: CostEstimate(
            price_snapshot_id="test", complete=False, missing_components=["usage:input"]
        ),
    )
    content = "".join(json.dumps(_row(request)) + "\n" for request in requests).encode()

    def download(*args, **kwargs):
        return {
            "status": "completed",
            "output_file_id": "file-output",
            "error_file_id": None,
        }, {"output": {"file_id": "file-output", "content": content}}

    run = tmp_path / "unpriceable-run"
    run.mkdir()
    manifest = collect_v2_batch(run_root=run, downloader=download)
    assert manifest["success_count"] == 32
    assert manifest["exact_target_coverage"] is True
    assert manifest["status"] == "failed_closed_unpriceable_usage"
    assert manifest["qualification_decisions_ready"] is False
    assert manifest["actual_total_cost_usd"] is None


def test_collect_resumes_same_intent_after_download_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request()
    loaded = _loaded_qualification(tmp_path, [request])
    intent = {
        "run_intent_sha256": "r" * 64,
        "maximum_authorized_cost_usd": 1.0,
    }
    submission = {
        "submission_sha256": "s" * 64,
        "provider_response": {
            "batch_id": "batch-resume",
            "input_file_id": "file-input",
        },
    }
    monkeypatch.setattr(module, "_load_run", lambda _: (intent, loaded, submission))
    monkeypatch.setattr(
        module, "_validate_provider_snapshot", lambda *args, **kwargs: None
    )
    calls = 0

    def download(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("transient download failure")
        return {
            "status": "completed",
            "output_file_id": "file-output",
            "error_file_id": None,
        }, {
            "output": {
                "file_id": "file-output",
                "content": (json.dumps(_row(request)) + "\n").encode(),
            }
        }

    run = tmp_path / "resume-run"
    run.mkdir()
    with pytest.raises(TimeoutError, match="transient"):
        collect_v2_batch(run_root=run, downloader=download)
    original_intent = (run / "collection-intent.json").read_bytes()
    manifest = collect_v2_batch(run_root=run, downloader=download)
    assert calls == 2
    assert (run / "collection-intent.json").read_bytes() == original_intent
    assert manifest["status"] == "failed_closed_provider_results"
