from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.analysis.bonafide.coarse_sampling_annotation import (
    cost_plan,
    load_coarse_config,
)
from circuits.analysis.bonafide.coarse_sampling_openai_run import (
    load_offline_qualification,
    run_direct_qualification,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts/bonafide/configs/process_witness_coarse_openai_v1.json"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, values: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values)
    )


def _qualification_root(tmp_path: Path) -> Path:
    root = tmp_path / "qualification"
    root.mkdir()
    price_path = root / "prices.json"
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
    _write_json(price_path, prices)
    requests = []
    for index in range(16):
        body = {
            "model": "gpt-5.6-luna",
            "input": [{"role": "user", "content": f"label unit-{index}"}],
            "max_output_tokens": 100,
            "store": False,
        }
        requests.append(
            {
                "request_id": f"request-{index}",
                "body_sha256": canonical_sha256(body),
                "repeat_of_request_id": None,
                "focal_unit_ids": [f"unit-{index}"],
                "provider_body": body,
            }
        )
    _write_jsonl(root / "requests.jsonl", requests)
    plan = cost_plan(requests, load_coarse_config(CONFIG), prices)
    plan.update(
        {
            "price_snapshot_path": str(price_path),
            "price_snapshot_sha256": file_sha256(price_path),
        }
    )
    plan["cost_plan_sha256"] = canonical_sha256(plan)
    _write_json(root / "cost-plan.json", plan)
    files = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in (root / "requests.jsonl", root / "cost-plan.json")
    ]
    for name in ("human-review-template.jsonl", "units.jsonl", "windows.json"):
        path = root / name
        path.write_text("")
        files.append(
            {"path": name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        )
    manifest = {
        "schema_version": "adag.process-witness.coarse-qualification-bundle.v1",
        "status": "prepared_offline_no_provider_calls",
        "network_calls_made": 0,
        "config_path": str(CONFIG),
        "config_sha256": file_sha256(CONFIG),
        "files": files,
        "request_bindings_in_order": [
            {
                "request_id": request["request_id"],
                "body_sha256": request["body_sha256"],
                "repeat_of_request_id": None,
            }
            for request in requests
        ],
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    _write_json(root / "manifest.json", manifest)
    return root


class _FakeResponse:
    def __init__(self, unit_id: str, index: int):
        self.id = f"response-{index}"
        self.model = "gpt-5.6-luna-2026-08-01"
        self.status = "completed"
        self.incomplete_details = None
        self.output_text = json.dumps(
            {
                "decisions": [
                    {
                        "unit_id": unit_id,
                        "tag": "active_task_work",
                        "confidence": "high",
                        "boundary_concerns": [],
                        "boundary_note": "",
                    }
                ]
            }
        )
        self.usage = SimpleNamespace(
            input_tokens=50,
            output_tokens=25,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
            output_tokens_details=SimpleNamespace(reasoning_tokens=5),
        )

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {
            "id": self.id,
            "model": self.model,
            "status": self.status,
            "output_text": self.output_text,
            "usage": {
                "input_tokens": 50,
                "output_tokens": 25,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 5},
            },
        }


class _FakeResponses:
    def __init__(self) -> None:
        self.index = 0

    def create(self, **body: object) -> _FakeResponse:
        content = body["input"][0]["content"]  # type: ignore[index]
        unit_id = str(content).split()[-1]
        response = _FakeResponse(unit_id, self.index)
        self.index += 1
        return response


def test_direct_run_logs_every_receipt_and_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification = _qualification_root(tmp_path)
    loaded = load_offline_qualification(qualification)
    assert len(loaded["requests"]) == 16
    monkeypatch.setattr(
        "circuits.analysis.bonafide.coarse_sampling_openai_run._source_revision",
        lambda: {"git_commit": "a" * 40, "tracked_worktree_clean": True},
    )
    monkeypatch.setattr(
        "circuits.analysis.bonafide.coarse_sampling_openai_run._readonly_tree",
        lambda _: None,
    )
    client = SimpleNamespace(responses=_FakeResponses())
    output = tmp_path / "run"
    manifest = run_direct_qualification(
        qualification_root=qualification,
        output_root=output,
        maximum_authorized_cost_usd=1.0,
        authorization_note="test authorization",
        client=client,
    )
    assert manifest["status"] == "complete"
    assert manifest["event_count"] == 16
    assert manifest["success_count"] == 16
    assert manifest["actual_total_cost_usd"] > 0
    assert len(list((output / "intents").glob("*.json"))) == 16
    assert len(list((output / "raw").glob("*.json"))) == 16
    assert len(list((output / "records").glob("*.json"))) == 16
    events = [
        json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()
    ]
    assert all(event["provider_request_id"] for event in events)
    assert all(event["usage"]["input_tokens"] == 50 for event in events)


def test_direct_run_refuses_cost_above_authorization(tmp_path: Path) -> None:
    qualification = _qualification_root(tmp_path)
    with pytest.raises(ValueError, match="exceeds authorization"):
        run_direct_qualification(
            qualification_root=qualification,
            output_root=tmp_path / "run",
            maximum_authorized_cost_usd=0.001,
            authorization_note="too low",
            client=SimpleNamespace(responses=_FakeResponses()),
        )


def test_transport_error_freezes_unknown_cost_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification = _qualification_root(tmp_path)
    monkeypatch.setattr(
        "circuits.analysis.bonafide.coarse_sampling_openai_run._source_revision",
        lambda: {"git_commit": "a" * 40, "tracked_worktree_clean": True},
    )
    monkeypatch.setattr(
        "circuits.analysis.bonafide.coarse_sampling_openai_run._readonly_tree",
        lambda _: None,
    )

    class FailingResponses:
        def create(self, **_: object) -> None:
            raise TimeoutError("unknown provider state")

    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="no automatic retry"):
        run_direct_qualification(
            qualification_root=qualification,
            output_root=output,
            maximum_authorized_cost_usd=1.0,
            authorization_note="test authorization",
            client=SimpleNamespace(responses=FailingResponses()),
        )
    manifest = json.loads((output / "run-manifest.json").read_text())
    assert manifest["status"] == "failed_closed_no_resume"
    assert manifest["cost_complete"] is False
    assert manifest["actual_total_cost_usd"] is None
    assert manifest["known_priced_cost_usd"] == 0
    assert manifest["event_count"] == 1
